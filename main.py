import argparse
import gc
import os
import sys
import textwrap
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.dataset import Dataset as PolygraphDataset
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.processor import Logger
from lm_polygraph.utils.builder_enviroment_stat_calculator import BuilderEnvironmentStatCalculator
from lm_polygraph.defaults.register_default_stat_calculators import register_default_stat_calculators
from lm_polygraph.ue_metrics import PredictionRejectionArea
from lm_polygraph.estimators import *

SEED = 3407

MODELS = {
    "LFM2-350M":      "LiquidAI/LFM2-350M",
    "LFM2-1.2B":      "LiquidAI/LFM2-1.2B",
    "MedGemma-4B-it": "google/medgemma-4b-it",
    "Gemma3-4B-it":   "google/gemma-3-4b-it",
    # Ancora di replica a 7B: e' uno dei backbone usati da Vashurin et al., e
    # serve a verificare che la nostra pipeline riproduca i loro risultati nel
    # loro stesso regime di scala. Senza questo controllo, qualunque scostamento
    # osservato sui modelli piccoli sarebbe attribuibile alla pipeline invece
    # che alla scala.
    "Mistral-7B-it":  "mistralai/Mistral-7B-Instruct-v0.2",
}

# Numero di parametri (in miliardi), usato per l'asse x della figura
# Kendall tau vs scala.
MODEL_PARAMS_B = {
    "LFM2-350M": 0.35,
    "LFM2-1.2B": 1.2,
    "MedGemma-4B-it": 4.0,
    "Gemma3-4B-it": 4.0,
    "Mistral-7B-it": 7.0,
}

# Modello di riferimento per il confronto di ranking (Kendall tau): e' il
# modello nel regime di scala del paper, quindi il metro su cui misurare
# quanto il ranking dei metodi UQ "trasferisce" scendendo di scala.
REPLICATION_ANCHOR_MODEL = "Mistral-7B-it"

# Richiedono di accettare la licenza su huggingface.co + un HF_TOKEN esportato.
GATED_MODELS = {"MedGemma-4B-it", "Gemma3-4B-it", "Mistral-7B-it"}

# Modelli instruction-tuned: si aspettano un chat template esplicito (marcatori
# di turno tipo <start_of_turn>/<end_of_turn> o [INST]). Dando loro lo stesso
# prompt a completamento usato per i modelli base LFM2 non generano
# correttamente (chiudono subito il turno emettendo EOS). I loro prompt vengono
# costruiti con tokenizer.apply_chat_template().
CHAT_TEMPLATE_MODELS = {"MedGemma-4B-it", "Gemma3-4B-it", "Mistral-7B-it"}

# Modelli che NON possono essere quantizzati a 4-bit: la famiglia Gemma3
# produce logit NaN sotto bitsandbytes nf4 (verificato escludendo prima
# backend di attenzione, formato del prompt e quantizzazione del solo lm_head),
# quindi vanno caricati in bf16. Nota: questo insieme e' volutamente distinto
# da CHAT_TEMPLATE_MODELS, con cui coincideva prima dell'aggiunta di Mistral:
# Mistral-7B ha bisogno del chat template ma DEVE restare quantizzato, perche'
# in bf16 occuperebbe ~15GB e i modelli 4B in bf16 gia' toccavano 23.3GB dei
# 23.6GB disponibili sulla 3090.
NO_QUANT_MODELS = {"MedGemma-4B-it", "Gemma3-4B-it"}


def uses_quantization(model_name):
    """4-bit nf4 per default; bf16 solo per i modelli che sotto quantizzazione
    producono NaN (vedi NO_QUANT_MODELS)."""
    return model_name not in NO_QUANT_MODELS


def print_banner():
    print("=" * 70)
    print("UQ BENCHMARK -- Sezione 5.1 Vashurin et al. (arXiv:2406.15627)")
    print("Selective QA: CoQA, TriviaQA, MMLU, GSM8k")
    print(f"Modelli: {', '.join(MODELS)}")
    print(f"Ancora di replica a 7B: {REPLICATION_ANCHOR_MODEL}")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {torch.cuda.get_device_name(i)} - {props.total_memory / 1e9:.1f} GB")
    print("=" * 70)


def log_gpu_mem(tag):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[GPU MEM] {tag}: allocated={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB")


# Loader dei dataset, metriche di qualita' e registri di configurazione vivono
# in dataset_prep.py, cosi' che main.py resti concentrato sull'esecuzione del
# benchmark e sulla produzione delle figure.
from dataset_prep import (
    DATASETS,
    SEVERITY_DATASETS,
    LINGUISTIC_EXPRESSIONS,
    VERBALIZED_CONFIDENCE_REGEX,
    build_verbalized_content,
    format_prompt,
    format_chat_prompt,
)


FIGURE_A_MODELS_NOTE = (
    "Figura A ~ Vashurin et al. Fig. 2 (white-box, full access: StableLM v2 12b / "
    "Mistral v0.2 7b base), Mean PRR aggregato su CoQA/TriviaQA/MMLU/GSM8k -> sostituita da "
    "LFM2-350M / LFM2-1.2B / MedGemma-4B-it / Gemma3-4B-it, piu' Mistral-7B-Instruct-v0.2 come "
    "ancora di replica nel regime di scala del paper. Barre d'errore: intervalli bootstrap al 95%. "
    "Da leggere insieme a accuracy_table.csv: un PRR basso su un modello con accuracy vicina al "
    "caso non indica un metodo debole ma una misura presa in un regime degenere."
)
FIGURE_B_MODELS_NOTE = (
    "Figura B ~ Vashurin et al. Fig. 3 (black-box/reflexive: StableLM v2 12b Chat / "
    "Mistral v0.2 7b Instruct / GPT-4o-mini), Mean PRR aggregato su CoQA/TriviaQA/MMLU/GSM8k -> "
    "sostituita dagli stessi nostri modelli. Nel nostro caso l'accesso e' sempre white-box, ma i "
    "metodi di questo sottoinsieme usano solo il testo generato (piu', per Semantic Entropy, la "
    "log-probabilita' di sequenza), quindi il valore coincide con quello ottenibile in accesso "
    "ristretto. Barre d'errore: intervalli bootstrap al 95%."
)

# ---------------------------------------------------------------------------
# Mappatura 1:1 con le righe (etichette esatte) delle Figure 2/3 e della
# Tabella 6 di Vashurin et al. (arXiv:2406.15627, TACL 2025), cosi' come
# incollate da Filo. Ogni voce e':
#   - "factory": callable che istanzia il nostro estimator equivalente,
#     "alias:<paper_label>" se il metodo e' identico a un altro gia' incluso
#     (nessuna seconda istanza, solo etichetta duplicata nelle tabelle finali),
#     None se il metodo NON e' incluso (vedi "reason").
#   - "figure": "A" (solo Fig. 2, white-box full-access, stesso set di
#     metodi della Tabella 6), "B" (solo Fig. 3, reflexive/black-box),
#     "AB" (in entrambe le figure: i metodi a diversita' campionaria che
#     restano identici in entrambi gli scenari).
# Questa lista e' la SINGOLA fonte di verita': build_estimators() e le
# tabelle finali derivano entrambe da qui, quindi "inclusi + esclusi con
# motivo" == esattamente le righe delle 3 immagini, senza aggiunte ne' omissioni.
# ---------------------------------------------------------------------------
PAPER_METHODS = [
    # --- solo Figura A / Tabella 6 (white-box, full access) ---
    {"paper_label": "CCP", "figure": "A", "factory": lambda: ClaimConditionedProbability()},
    {"paper_label": "Maximum Sequence Probability", "figure": "A", "factory": lambda: MaximumSequenceProbability()},
    {"paper_label": "SAR", "figure": "A", "factory": lambda: SAR()},
    {"paper_label": "Perplexity", "figure": "A", "factory": lambda: Perplexity()},
    {"paper_label": "TokenSAR", "figure": "A", "factory": lambda: TokenSAR()},
    {"paper_label": "SentenceSAR", "figure": "A", "factory": lambda: SentenceSAR()},
    {"paper_label": "Semantic Entropy", "figure": "A", "factory": lambda: SemanticEntropy()},
    {"paper_label": "Mean Token Entropy", "figure": "A", "factory": lambda: MeanTokenEntropy()},
    {"paper_label": "Monte Carlo Sequence Entropy", "figure": "A", "factory": lambda: MonteCarloSequenceEntropy()},
    {"paper_label": "Monte Carlo Normalized Sequence Entropy", "figure": "A", "factory": lambda: MonteCarloNormalizedSequenceEntropy()},
    {"paper_label": "Pointwise Mutual Information", "figure": "A", "factory": lambda: MeanPointwiseMutualInformation()},
    {"paper_label": "P(True)", "figure": "A", "factory": lambda: PTrue()},
    {"paper_label": "Conditional Pointwise Mutual Information", "figure": "A", "factory": lambda: MeanConditionalPointwiseMutualInformation()},
    {"paper_label": "Fisher-Rao", "figure": "A", "factory": lambda: FisherRao()},
    {"paper_label": "Renyi Divergence", "figure": "A", "factory": lambda: RenyiNeg()},

    # --- Figura A + B (diversita' campionaria, presenti in entrambi gli scenari) ---
    {"paper_label": "EigValLaplacian NLI Score Entail.", "figure": "AB", "factory": lambda: EigValLaplacian(similarity_score="NLI_score", affinity="entail")},
    {"paper_label": "EigValLaplacian Jaccard Score", "figure": "AB", "factory": lambda: EigValLaplacian(similarity_score="Jaccard_score")},
    {"paper_label": "DegMat NLI Score Entail.", "figure": "AB", "factory": lambda: DegMat(similarity_score="NLI_score", affinity="entail")},
    {"paper_label": "DegMat Jaccard Score", "figure": "AB", "factory": lambda: DegMat(similarity_score="Jaccard_score")},
    {"paper_label": "Eccentricity NLI Score Entail.", "figure": "AB", "factory": lambda: Eccentricity(similarity_score="NLI_score", affinity="entail")},
    {"paper_label": "Eccentricity Jaccard Score", "figure": "AB", "factory": lambda: Eccentricity(similarity_score="Jaccard_score")},
    {"paper_label": "Lexical Similarity Rouge-L", "figure": "AB", "factory": lambda: LexicalSimilarity(metric="rougeL")},
    {"paper_label": "Lexical Similarity BLEU", "figure": "AB", "factory": lambda: LexicalSimilarity(metric="BLEU")},
    {"paper_label": "NumSet", "figure": "AB", "factory": lambda: NumSemSets()},

    # --- solo Figura B (reflexive / black-box) ---
    {"paper_label": "BB Semantic Entropy", "figure": "B", "factory": "alias:Semantic Entropy",
     "reason": "Stessa classe SemanticEntropy() del white-box (lm-polygraph non ha una classe black-box "
               "separata): nessuna istanza in piu', il valore viene solo duplicato in tabella con questa etichetta."},
    {"paper_label": "Label Prob.", "figure": "B", "factory": lambda: LabelProb()},
    {"paper_label": "BB P(True)", "figure": "B", "factory": lambda: PTrueEmpirical()},

    # --- esclusi: density-based, richiedono dati di training separati ---
    {"paper_label": "Mahalanobis Distance - Decoder", "figure": "A", "factory": None,
     "reason": "Richiede fit di un modello di densita' su embeddings del train set "
               "(TrainingStatisticExtractionCalculator + EmbeddingsCalculator, non registrati di default "
               "in register_default_stat_calculators): forward pass extra per modello, costo di tempo "
               "cluster e rischio OOM su MedGemma/Gemma3 in bf16 (gia' a 23.5/24GB). Escluso su richiesta esplicita (2026-08-12)."},
    {"paper_label": "RDE - Decoder", "figure": "A", "factory": None,
     "reason": "Stesso motivo di Mahalanobis Distance: richiede training data ed embeddings extra. Escluso su richiesta esplicita (2026-08-12)."},
    {"paper_label": "Relative Mahalanobis Distance - Decoder", "figure": "A", "factory": None,
     "reason": "Stesso motivo di Mahalanobis Distance: richiede training data ed embeddings extra. Escluso su richiesta esplicita (2026-08-12)."},
    {"paper_label": "HUQ-MD - Decoder", "figure": "A", "factory": None,
     "reason": "Non esiste come classe in lm-polygraph (verificato nel sorgente del repo IINemo/lm-polygraph: "
               "nessun file/import con 'HUQ' in src/lm_polygraph/estimators/) -- andrebbe implementato da zero "
               "leggendo il paper originale del metodo, fuori scope."},

    # --- esclusi: verbalized/linguistic, incompatibili con la pipeline a generazione condivisa ---
    {"paper_label": "Verbalized 1S top-k", "figure": "B", "factory": None,
     "reason": "Estrae la confidenza dalla STESSA generazione greedy condivisa con gli altri 26 stimatori, "
               "che oggi contiene solo la risposta breve/lettera richiesta dal task (nessun testo di confidenza). "
               "Per renderlo utile dovremmo cambiare prompt/lunghezza di generazione per tutti gli stimatori."},
    {"paper_label": "Verbalized 1S top-1", "figure": "B", "factory": None,
     "reason": "Stesso motivo di Verbalized 1S top-k (legge la confidenza dalla generazione condivisa)."},
    {"paper_label": "Verbalized 2S top-k", "figure": "B", "factory": None,
     "reason": "Richiede come 'prima risposta' un campionamento top-k; la nostra risposta condivisa e' "
               "sempre greedy/deterministica, non produciamo varianti top-k della risposta principale."},
    {"paper_label": "Verbalized 2S CoT", "figure": "B", "factory": None,
     "reason": "Richiede una risposta con ragionamento chain-of-thought come primo turno; incompatibile "
               "con i prompt CoQA/TriviaQA/MMLU (risposta breve richiesta) -- solo GSM8k ha gia' CoT, ma "
               "e' un solo dataset su quattro e non e' comunque il formato standard atteso da questo estimator."},
    {"paper_label": "Verbalized 2S top-1", "figure": "B", "factory": None,
     "reason": "Verificato in lm_polygraph.utils.model.WhiteboxModel.tokenize(): il turno di follow-up "
               "chat (necessario per questo estimator) viene formattato solo se il modello e' istanziato "
               "con instruct=True; noi usiamo sempre instruct=False perche' applichiamo gia' il chat "
               "template a mano al prompt principale (vedi format_chat_prompt). Attivare instruct=True "
               "applicherebbe il chat template due volte a tutta la generazione condivisa degli altri "
               "26 stimatori su MedGemma/Gemma3, corrompendola."},
    {"paper_label": "Linguistic 1S", "figure": "B", "factory": None,
     "reason": "Stesso motivo di Verbalized 1S: legge un'espressione verbale di confidenza dalla "
               "generazione condivisa, che non la contiene."},
]


def build_estimators():
    """Istanzia solo i metodi con una factory valida in PAPER_METHODS (esclude
    alias e voci senza factory). Vedi PAPER_METHODS per la mappatura completa
    e i motivi di ogni esclusione."""
    estimators = [m["factory"]() for m in PAPER_METHODS if callable(m["factory"])]
    n_alias = sum(1 for m in PAPER_METHODS if isinstance(m["factory"], str) and m["factory"].startswith("alias:"))
    n_excluded = sum(1 for m in PAPER_METHODS if m["factory"] is None)
    print(f"Totale stimatori: {len(estimators)} (+{n_alias} alias, {n_excluded} esclusi con motivo, "
          f"su {len(PAPER_METHODS)} metodi mappati dalle Figure 2/3 e Tabella 6 del paper)")
    return estimators


def write_excluded_methods_report(results_dir):
    """Scrive un riepilogo leggibile dei metodi del paper esclusi (con motivo)
    e degli alias, per confronto rapido senza dover leggere PAPER_METHODS nel sorgente."""
    lines = ["# Metodi UQ: mappatura con le Figure 2/3 e Tabella 6 (arXiv:2406.15627)\n"]
    lines.append(FIGURE_A_MODELS_NOTE + "\n")
    lines.append(FIGURE_B_MODELS_NOTE + "\n\n")
    lines.append("## Alias (stesso valore di un altro metodo gia' incluso)\n")
    for m in PAPER_METHODS:
        if isinstance(m["factory"], str) and m["factory"].startswith("alias:"):
            lines.append(f"- **{m['paper_label']}** (Fig. {m['figure']}): {m.get('reason', '')}\n")
    lines.append("\n## Esclusi\n")
    for m in PAPER_METHODS:
        if m["factory"] is None:
            lines.append(f"- **{m['paper_label']}** (Fig. {m['figure']}): {m.get('reason', '')}\n")
    lines.append(f"\nTotale metodi mappati: {len(PAPER_METHODS)}. "
                  f"Inclusi: {sum(1 for m in PAPER_METHODS if callable(m['factory']))}. "
                  f"Alias: {sum(1 for m in PAPER_METHODS if isinstance(m['factory'], str))}. "
                  f"Esclusi: {sum(1 for m in PAPER_METHODS if m['factory'] is None)}.\n")
    path = os.path.join(results_dir, "excluded_methods.md")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"Report metodi esclusi salvato: {path}")
    return path


class TimedEstimator:
    """Wrapper trasparente attorno a un Estimator di lm-polygraph: misura il
    tempo speso in __call__ (con torch.cuda.synchronize() per un timing GPU
    accurato) e lo accumula in timing_dict[str(estimator)], usato per la
    tabella di efficienza per metodo/modello. Preserva __str__/level/
    stats_dependencies cosi' che UEManager lo tratti in modo identico
    all'estimator originale (stesse chiavi in man.metrics)."""

    def __init__(self, estimator, timing_dict):
        self._estimator = estimator
        self._timing_dict = timing_dict
        self.level = estimator.level
        self.stats_dependencies = estimator.stats_dependencies

    def __str__(self):
        return str(self._estimator)

    def __call__(self, stats):
        # torch.cuda.synchronize() prima di entrambe le letture del cronometro:
        # le operazioni GPU sono asincrone, senza sincronizzazione si
        # misurerebbe l'accodamento e non l'esecuzione. perf_counter invece di
        # time per coerenza con il timing delle fasi in TimedUEManager.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = self._estimator(stats)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        key = str(self._estimator)
        self._timing_dict[key] = self._timing_dict.get(key, 0.0) + elapsed
        return result


# ---------------------------------------------------------------------------
# Timing end-to-end.
#
# UEManager calcola UNA VOLTA SOLA le statistiche condivise di ogni domanda
# (risposta greedy, K campioni, matrice NLI tra i campioni) e poi tutti gli
# stimatori leggono da quel magazzino comune. Cronometrare soltanto la
# chiamata dell'estimator, come faceva la versione precedente, misura quindi
# solo l'ultimo passo -- l'aritmetica su statistiche gia' pronte -- e produce
# un risultato paradossale: DegMat sembra quasi gratuito (millisecondi di
# algebra su una matrice gia' calcolata) mentre il suo costo vero, le K
# generazioni piu' i forward del modello NLI, e' ammortizzato nel magazzino
# condiviso e non attribuito a nessuno; all'opposto BB P(True) sembra
# costosissimo solo perche' la sua chiamata extra al modello non e' condivisa.
#
# Quel numero risponde a "quanto costa aggiungere questa tecnica se sto gia'
# calcolando tutto il resto?". La domanda del deployment on-device e' un'altra:
# "quanto costa questa tecnica se e' l'unica che faccio girare?". Servono
# entrambi, quindi qui si cronometrano anche le fasi di raccolta delle
# statistiche e si attribuiscono a ciascun metodo quelle che gli servono
# davvero, in base alle sue stats_dependencies dichiarate.
# ---------------------------------------------------------------------------

# Statistiche che implicano il campionamento di K generazioni aggiuntive.
_SAMPLING_STATS_PREFIXES = ("sample_", "blackbox_sample_")
# Statistiche che implicano i forward del modello NLI (DeBERTa).
_NLI_STATS_MARKERS = ("semantic_matrix", "semantic_classes")


def classify_stat_calculator(name):
    """Assegna un calcolatore di statistiche a una fase di costo, in base al
    suo nome di classe."""
    lowered = name.lower()
    if "sampling" in lowered:
        return "campionamento_K_generazioni"
    if "semantic" in lowered or "deberta" in lowered or "nli" in lowered:
        return "forward_NLI"
    if "greedy" in lowered:
        return "generazione_greedy"
    return "altro"


def estimator_phase_needs(estimator):
    """Quali fasi costose servono davvero a uno stimatore, dedotte dalle
    statistiche che dichiara di volere. La generazione greedy serve sempre,
    perche' e' la risposta di cui si stima l'incertezza."""
    deps = [str(d) for d in getattr(estimator, "stats_dependencies", [])]
    needs_sampling = any(d.startswith(_SAMPLING_STATS_PREFIXES) for d in deps)
    needs_nli = any(any(m in d for m in _NLI_STATS_MARKERS) for d in deps)
    return {"generazione_greedy": True,
            "campionamento_K_generazioni": needs_sampling,
            "forward_NLI": needs_nli}


class TimedUEManager(UEManager):
    """UEManager che cronometra anche ogni calcolatore di statistiche, non solo
    gli stimatori.

    Reimplementa `calculate` delegando al metodo del padre un calcolatore alla
    volta: cosi' la gestione degli errori resta quella della libreria (non
    viene duplicata qui, dove potrebbe divergere a un aggiornamento) e si
    ottiene comunque il tempo per singola fase. `torch.cuda.synchronize()`
    prima di ogni lettura del cronometro e' obbligatorio: le operazioni GPU
    sono asincrone e senza sincronizzazione si misurerebbe il tempo di
    ACCODAMENTO dell'operazione invece della sua esecuzione, sottostimando i
    tempi anche di ordini di grandezza."""

    def __init__(self, *args, stat_timing_dict=None, **kwargs):
        self._stat_timing = stat_timing_dict if stat_timing_dict is not None else {}
        super().__init__(*args, **kwargs)

    def calculate(self, batch_stats, calculators, inp_texts):
        for stat_calculator in calculators:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            batch_stats = super().calculate(batch_stats, [stat_calculator], inp_texts)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            key = type(stat_calculator).__name__
            self._stat_timing[key] = self._stat_timing.get(key, 0.0) + (time.perf_counter() - start)
        return batch_stats


def load_whitebox_model(model_id, cache_dir, hf_token=None, attn_implementation="eager", use_quantization=True):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        device_map="auto",
        token=hf_token,
        cache_dir=cache_dir,
        attn_implementation=attn_implementation,
    )
    if use_quantization:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    hf_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    hf_model.eval()

    model = WhiteboxModel(hf_model, tokenizer, model_path=model_id)
    return model


def build_manager(model, dataset, estimators, cache_dir, max_rejection, max_new_tokens,
                  generation_metric, stat_timing_dict=None):
    available_stat_calculators = register_default_stat_calculators(
        model_type="Whitebox",
        language="en",
        hf_cache=cache_dir,
        output_attentions=False,
        output_hidden_states=True,
        blackbox_supports_logprobs=False,
        deberta_batch_size=10,
    )
    builder_env_stat_calc = BuilderEnvironmentStatCalculator(model=model)

    man = TimedUEManager(
        data=dataset,
        model=model,
        estimators=estimators,
        builder_env_stat_calc=builder_env_stat_calc,
        available_stat_calculators=available_stat_calculators,
        generation_metrics=[generation_metric],
        ue_metrics=[PredictionRejectionArea(max_rejection=max_rejection)],
        processors=[Logger()],
        ignore_exceptions=True,
        max_new_tokens=max_new_tokens,
        stat_timing_dict=stat_timing_dict,
    )
    return man


def extract_prr_table(man, model_name):
    """Converte man.metrics (dict con chiavi (livello, estimator_name,
    generation_metric_name, ue_metric_name) -> valore) in un DataFrame lungo."""
    rows = []
    for key, value in man.metrics.items():
        try:
            *rest, ue_metric_name = key
            estimator_name = key[1] if len(key) > 1 else str(key)
            rows.append({
                "model": model_name,
                "key": key,
                "estimator": estimator_name,
                "ue_metric": ue_metric_name,
                "value": value,
            })
        except TypeError:
            rows.append({"model": model_name, "key": str(key), "estimator": str(key), "ue_metric": None, "value": value})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analisi a livello di singola istanza.
#
# UEManager espone, oltre ai valori aggregati in man.metrics, anche i valori
# per-istanza: man.gen_metrics[(livello, nome_metrica)] contiene la qualita'
# della generazione domanda per domanda, e man.estimations[(livello, nome_
# stimatore)] il punteggio di incertezza domanda per domanda. Sono gli stessi
# array che eval_ue() usa internamente per calcolare il PRR, e servono qui per
# tre cose che il solo valore aggregato non permette: l'accuracy di base del
# modello, gli intervalli di confidenza bootstrap e il silent failure rate.
# ---------------------------------------------------------------------------

# Soglia oltre la quale una risposta e' considerata corretta. Per le metriche
# binarie (AccuracyMetric, GSM8kAccuracy) qualunque valore in (0,1) equivale;
# per AlignScore, che e' continua, 0.5 e' una scelta convenzionale da
# dichiarare in tesi -- il silent failure rate va letto rispetto a questa
# soglia, non come una verita' assoluta.
CORRECTNESS_THRESHOLD = 0.5


def extract_instance_level(man):
    """Estrae dal manager gli array per-istanza. Ritorna
    (qualita', nome_metrica_qualita', {nome_stimatore: punteggi_incertezza})."""
    quality, quality_name = None, None
    for (level, name), values in man.gen_metrics.items():
        if level == "sequence":
            quality = np.asarray(values, dtype=float)
            quality_name = name
            break

    scores = {}
    for (level, name), values in man.estimations.items():
        if level == "sequence":
            scores[name] = np.asarray(values, dtype=float)
    return quality, quality_name, scores


def _prr_from_arrays(ue_metric, ue, quality):
    """Riproduce esattamente il calcolo che UEManager.eval_ue() fa per il PRR,
    cosi' che i valori bootstrap siano confrontabili con quello aggregato:
    le istanze con qualita' NaN vengono scartate, e i NaN nel punteggio di
    incertezza vengono sostituiti con -1e7 (vedi _delete_nans nel sorgente di
    lm-polygraph).

    ATTENZIONE, questo e' un dettaglio con conseguenze reali sui metodi
    verbalized: un punteggio NaN significa "confidenza non estraibile dal
    testo", ma -1e7 e' il valore piu' BASSO possibile di incertezza, quindi
    quelle istanze vengono trattate come le piu' confidenti in assoluto e non
    vengono mai scartate dal PRR. Un modello che non riesce a produrre il
    formato richiesto viene cosi' premiato invece che penalizzato: e' il
    motivo per cui il parse-failure rate va sempre riportato accanto al PRR
    dei metodi verbalized."""
    clipped = np.nan_to_num(ue, nan=-1e7, neginf=-1e7, posinf=1e7)
    keep = ~np.isnan(quality)
    clipped, q = clipped[keep], quality[keep]
    if len(q) == 0:
        return np.nan
    # PredictionRejectionArea normalizza la qualita' min-max: se tutte le
    # istanze hanno lo stesso valore il denominatore e' zero e il risultato
    # non e' definito (tipicamente accade quando il modello sbaglia tutto o
    # indovina tutto in un ricampionamento bootstrap sfortunato).
    if np.nanmax(q) == np.nanmin(q):
        return np.nan
    try:
        return float(ue_metric(clipped, q))
    except Exception:
        return np.nan


def bootstrap_prr_ci(ue, quality, max_rejection, n_resamples, seed, alpha=0.05):
    """Intervallo di confidenza bootstrap percentile sul PRR.

    Il benchmark e' calcolato su un campione di domande, non sul dataset
    intero: ripetendolo con altre domande ogni PRR verrebbe leggermente
    diverso. Il bootstrap stima quanto, senza bisogno di nuove run GPU:
    ricampiona con reimmissione le stesse istanze gia' calcolate, ricalcola il
    PRR su ogni ricampionamento e usa i percentili dei valori ottenuti come
    barre d'errore.

    Il ricampionamento e' fatto sugli INDICI delle istanze e applicato insieme
    a punteggio e qualita', perche' l'unita' che varia tra un esperimento e
    l'altro e' la domanda: separare i due array distruggerebbe
    l'accoppiamento e sottostimerebbe l'incertezza."""
    rng = np.random.default_rng(seed)
    ue_metric = PredictionRejectionArea(max_rejection=max_rejection)
    n = len(quality)
    if n == 0:
        return np.nan, np.nan
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        v = _prr_from_arrays(ue_metric, ue[idx], quality[idx])
        if np.isfinite(v):
            values.append(v)
    # Se piu' della meta' dei ricampionamenti e' degenere (es. accuracy
    # costante), l'intervallo non e' affidabile e viene riportato come NaN
    # invece di un numero falsamente preciso.
    if len(values) < n_resamples // 2:
        return np.nan, np.nan
    return (float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))))


def bootstrap_paired_diff_ci(ue_a, ue_b, quality, max_rejection, n_resamples, seed, alpha=0.05):
    """Intervallo di confidenza bootstrap APPAIATO sulla differenza
    PRR(A) - PRR(B).

    Serve per poter scrivere "A batte B" in modo difendibile. Guardare se due
    barre d'errore separate si sovrappongono e' troppo conservativo: A e B
    sono valutati sulle STESSE domande, quindi i loro punteggi sono correlati
    (entrambi faticano sulle domande difficili). Qui, a ogni ricampionamento,
    si ricalcola la differenza sullo stesso campione, e alla fine si guarda
    l'intervallo delle differenze: se non contiene zero, il vantaggio e'
    reale."""
    rng = np.random.default_rng(seed)
    ue_metric = PredictionRejectionArea(max_rejection=max_rejection)
    n = len(quality)
    if n == 0:
        return np.nan, np.nan, np.nan
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        q = quality[idx]
        va = _prr_from_arrays(ue_metric, ue_a[idx], q)
        vb = _prr_from_arrays(ue_metric, ue_b[idx], q)
        if np.isfinite(va) and np.isfinite(vb):
            diffs.append(va - vb)
    if len(diffs) < n_resamples // 2:
        return np.nan, np.nan, np.nan
    return (float(np.mean(diffs)),
            float(np.percentile(diffs, 100 * alpha / 2)),
            float(np.percentile(diffs, 100 * (1 - alpha / 2))))


def silent_failure_rate(ue, quality, quantile=0.10):
    """Frazione delle risposte SBAGLIATE che finisce nel decile piu'
    confidente del metodo.

    E' la metrica che interessa davvero in ambito clinico: non "quanto bene
    ordina in media" (il PRR) ma "quanti errori passano completamente
    inosservati, presentati con la massima sicurezza". Tutti gli stimatori di
    lm-polygraph restituiscono INCERTEZZA (valore alto = piu' incerto), quindi
    il decile piu' confidente e' quello con i valori piu' bassi."""
    keep = ~np.isnan(quality) & ~np.isnan(ue)
    if keep.sum() == 0:
        return np.nan
    u, q = ue[keep], quality[keep]
    wrong = q < CORRECTNESS_THRESHOLD
    if wrong.sum() == 0:
        return np.nan
    cutoff = np.quantile(u, quantile)
    return float((wrong & (u <= cutoff)).sum() / wrong.sum())


def compute_instance_level_stats(man, model_name, dataset_name, paper_label_by_str,
                                 max_rejection, n_bootstrap, seed):
    """Costruisce, per ogni stimatore, una riga con: PRR ricalcolato dagli
    array per-istanza, intervallo di confidenza bootstrap, accuracy di base
    del modello su quel dataset, parse-failure rate e silent failure rate.

    Ritorna anche il DataFrame "wide" con i valori grezzi per-istanza (una
    riga per domanda, una colonna per stimatore piu' la qualita'), salvato su
    disco per poter rifare bootstrap e test appaiati in seguito senza
    rieseguire nulla sulla GPU."""
    quality, quality_name, scores = extract_instance_level(man)
    if quality is None or len(quality) == 0:
        return None, None

    ue_metric = PredictionRejectionArea(max_rejection=max_rejection)
    mean_quality = float(np.nanmean(quality))
    n_instances = int(len(quality))

    rows = []
    per_instance = {"quality": quality}
    for est_name, ue in scores.items():
        if len(ue) != len(quality):
            print(f"ATTENZIONE: {est_name} ha {len(ue)} valori ma la qualita' ne ha "
                  f"{len(quality)}; salto le statistiche per-istanza di questo stimatore.")
            continue
        per_instance[est_name] = ue
        ci_low, ci_high = bootstrap_prr_ci(ue, quality, max_rejection, n_bootstrap, seed)
        rows.append({
            "model": model_name,
            "dataset": dataset_name,
            "estimator": est_name,
            "paper_label": paper_label_by_str.get(est_name, est_name),
            "prr": _prr_from_arrays(ue_metric, ue, quality),
            "prr_ci_low": ci_low,
            "prr_ci_high": ci_high,
            "quality_metric": quality_name,
            "mean_quality": mean_quality,
            "n_instances": n_instances,
            # Frazione di istanze in cui lo stimatore non ha prodotto un
            # numero. Per i metodi verbalized coincide col parse-failure rate
            # (confidenza non estraibile dal testo generato); per gli altri
            # metodi dovrebbe essere zero.
            "nan_rate": float(np.isnan(ue).mean()),
            "silent_failure_rate": silent_failure_rate(ue, quality),
        })

    stats_df = pd.DataFrame(rows)
    per_instance_df = pd.DataFrame(per_instance)
    per_instance_df.insert(0, "dataset", dataset_name)
    per_instance_df.insert(0, "model", model_name)
    return stats_df, per_instance_df


def run_model_on_dataset(model, model_name, dataset_name, examples, cfg, args, paper_label_by_str,
                         use_chat_template, estimators_factory=None, content_transform=None,
                         max_new_tokens_override=None):
    """Esegue gli stimatori UQ su un singolo modello gia' caricato contro un
    singolo dataset gia' preparato (prompt + reference).

    Ritorna (prr_df, timing_df, stats_df, per_instance_df), oppure quattro
    None se la combinazione fallisce. E' l'unico punto in cui viene eseguito
    UEManager: la pipeline principale e tutte le sezioni di confronto
    (severita', quantizzazione, verbalized) passano da qui, cosi' che
    qualunque correzione valga per tutte.

    I tre parametri opzionali servono alla sezione verbalized, che ha bisogno
    di un set di stimatori diverso, di un prompt che chieda esplicitamente la
    confidenza e di piu' token per generarla:
    - `estimators_factory`: callable che ritorna la lista di stimatori da
      usare al posto di build_estimators().
    - `content_transform`: callable applicata al testo del prompt prima di
      formattarlo.
    - `max_new_tokens_override`: sostituisce il max_new_tokens del dataset."""
    print(f"\n--- {model_name} su {dataset_name} ---")
    ds_start = time.time()
    df, timing_df, stats_df, per_instance_df = None, None, None, None
    try:
        contents = [ex["content"] for ex in examples]
        if content_transform is not None:
            contents = [content_transform(c) for c in contents]

        if use_chat_template:
            prompts = [format_chat_prompt(model.tokenizer, c) for c in contents]
        else:
            prompts = [format_prompt(c, cfg["plain_suffix"]) for c in contents]
        references = [ex["reference"] for ex in examples]

        model_dataset = PolygraphDataset(prompts, references, batch_size=args.batch_size)

        max_new_tokens = max_new_tokens_override or cfg["max_new_tokens"]
        base_estimators = estimators_factory() if estimators_factory else build_estimators()

        # Memoria di picco misurata per singola combinazione: su un telefono il
        # vincolo stringente e' spesso la RAM prima del tempo, quindi entra
        # nella tabella dei costi insieme ai tempi.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        timing_dict = {}
        stat_timing_dict = {}
        estimators = [TimedEstimator(e, timing_dict) for e in base_estimators]
        phase_needs = {str(e): estimator_phase_needs(e) for e in base_estimators}
        man = build_manager(
            model, model_dataset, estimators, args.cache_dir, args.max_rejection,
            max_new_tokens, cfg["generation_metric_factory"](),
            stat_timing_dict=stat_timing_dict,
        )
        man()
        log_gpu_mem(f"{model_name}/{dataset_name} done")
        peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else np.nan

        stats_df, per_instance_df = compute_instance_level_stats(
            man, model_name, dataset_name, paper_label_by_str,
            args.max_rejection, args.n_bootstrap, SEED,
        )
        if stats_df is not None and not stats_df.empty:
            acc = stats_df["mean_quality"].iloc[0]
            qname = stats_df["quality_metric"].iloc[0]
            print(f"{model_name}/{dataset_name}: {qname} medio = {acc:.3f} "
                  f"su {stats_df['n_instances'].iloc[0]} istanze.")

        df = extract_prr_table(man, model_name)
        df["dataset"] = dataset_name
        n_ok = df["value"].notna().sum() if "value" in df.columns else 0
        print(f"{model_name}/{dataset_name}: {n_ok}/{len(df)} righe metrica con valore.")

        # Tempo per fase, sommando i calcolatori di statistiche che ricadono
        # nella stessa fase di costo.
        phase_seconds = {}
        for calc_name, seconds in stat_timing_dict.items():
            phase_seconds[classify_stat_calculator(calc_name)] = (
                phase_seconds.get(classify_stat_calculator(calc_name), 0.0) + seconds
            )
        n_inst = max(len(examples), 1)

        timing_rows = []
        for est_str, seconds in timing_dict.items():
            needs = phase_needs.get(est_str, {"generazione_greedy": True})
            # Costo pieno standalone: se questa fosse l'unica tecnica in
            # esecuzione, dovrebbe pagarsi da sola la generazione greedy piu'
            # le fasi che le servono, oltre alla propria aritmetica.
            full = seconds
            for phase, needed in needs.items():
                if needed:
                    full += phase_seconds.get(phase, 0.0)
            timing_rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "estimator": est_str,
                "paper_label": paper_label_by_str.get(est_str, est_str),
                # Costo marginale: solo l'aritmetica dello stimatore su
                # statistiche gia' pronte (utile a chi ne calcola molti insieme).
                "seconds": seconds,
                "seconds_marginal_per_instance": seconds / n_inst,
                # Costo pieno: quello che conta per la scelta on-device.
                "seconds_full_standalone": full,
                "seconds_full_per_instance": full / n_inst,
                "needs_sampling": needs.get("campionamento_K_generazioni", False),
                "needs_nli": needs.get("forward_NLI", False),
                "peak_memory_gb": peak_mem_gb,
                "n_instances": n_inst,
            })
        timing_df = pd.DataFrame(timing_rows)

        for phase, seconds in sorted(phase_seconds.items(), key=lambda kv: -kv[1]):
            print(f"  [fase] {phase}: {seconds:.1f}s totali "
                  f"({seconds / n_inst:.3f}s per istanza)")
        del man
    except Exception:
        print(f"!!! {model_name}/{dataset_name} fallito, salto alla prossima combinazione.")
        traceback.print_exc()
        df, timing_df, stats_df, per_instance_df = None, None, None, None
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Tempo {model_name}/{dataset_name}: {time.time() - ds_start:.1f}s")
    return df, timing_df, stats_df, per_instance_df


def _pivot_with_ci(df, labels, group_order, value_col="value"):
    """Pivot metodo x modello dei valori PRR, piu' i due pivot paralleli con
    gli estremi dell'intervallo di confidenza (se presenti nel DataFrame).
    Ritorna (pivot, err_low, err_high), dove i due err sono gia' espressi
    come DISTANZA dal valore centrale, che e' il formato richiesto da
    matplotlib per xerr; se gli intervalli non ci sono, ritorna (pivot, None,
    None)."""
    subset = df[df["paper_label"].isin(labels)]
    if subset.empty:
        return None, None, None
    pivot = subset.pivot_table(index="paper_label", columns="model", values=value_col, aggfunc="first")
    pivot = pivot.reindex(columns=group_order)

    if "prr_ci_low" not in subset.columns or "prr_ci_high" not in subset.columns:
        return pivot, None, None

    low = subset.pivot_table(index="paper_label", columns="model", values="prr_ci_low", aggfunc="first")
    high = subset.pivot_table(index="paper_label", columns="model", values="prr_ci_high", aggfunc="first")
    low = low.reindex(index=pivot.index, columns=group_order)
    high = high.reindex(index=pivot.index, columns=group_order)
    # matplotlib vuole le semi-ampiezze, non gli estremi assoluti.
    err_low = (pivot - low).clip(lower=0)
    err_high = (high - pivot).clip(lower=0)
    return pivot, err_low, err_high


def _apply_label_margin(fig, index, cap=0.55, floor=0.22, per_char=0.011, **kwargs):
    """Margine sinistro proporzionale alla label piu' lunga (in caratteri):
    col figsize fisso il margine di default di matplotlib (~0.125) tronca
    etichette lunghe tipo "Monte Carlo Normalized Sequence Entropy"."""
    max_label_len = max(len(str(lbl)) for lbl in index)
    left_margin = min(cap, max(floor, max_label_len * per_char))
    fig.subplots_adjust(left=left_margin, **kwargs)


def plot_prr_bars(df, labels, group_order, title, note, out_path, xlabel, add_mean=True):
    """Bar chart orizzontale generico: una barra per elemento di group_order
    (colonna 'model' del df, che puo' contenere sia nomi di modelli veri sia
    etichette di varianti come "quantized"/"full precision"), una riga per
    etichetta 'paper_label' in labels. Se add_mean=True aggiunge anche una
    barra "Mean".

    Se il DataFrame contiene le colonne prr_ci_low/prr_ci_high (prodotte dal
    bootstrap), disegna anche le barre d'errore al 95%. La barra "Mean" non ne
    ha: e' una media di PRR calcolati su dataset diversi, non una statistica
    ricampionata, quindi un intervallo li' sarebbe fuorviante."""
    pivot, err_low, err_high = _pivot_with_ci(df, labels, group_order)
    if pivot is None:
        print(f"!!! Nessun dato per {out_path}, salto.")
        return None

    row_mean = pivot.mean(axis=1)
    order = row_mean.sort_values(ascending=True).index  # barh: ultima disegnata = in cima
    pivot = pivot.loc[order]

    xerr = None
    if err_low is not None:
        err_low, err_high = err_low.loc[order], err_high.loc[order]
        # Forma richiesta da pandas.plot(kind="barh"): un array
        # (n_colonne, 2, n_righe) con le semi-ampiezze basso/alto.
        xerr = np.stack([
            np.stack([err_low[c].to_numpy(dtype=float), err_high[c].to_numpy(dtype=float)])
            for c in pivot.columns
        ])
        xerr = np.nan_to_num(xerr, nan=0.0)

    if add_mean:
        pivot = pivot.copy()
        pivot["Mean"] = row_mean.loc[order]
        if xerr is not None:
            zeros = np.zeros((1, 2, len(pivot)))
            xerr = np.concatenate([xerr, zeros], axis=0)

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.4)))
    pivot.plot(kind="barh", ax=ax, width=0.8, xerr=xerr,
               error_kw={"elinewidth": 0.7, "ecolor": "0.3"})
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8)
    _apply_label_margin(fig, pivot.index, bottom=0.16)
    fig.text(0.01, 0.02, "\n".join(textwrap.wrap(note, width=115)), fontsize=7, va="bottom")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato: {out_path}")
    return pivot


def plot_severity_grid(df, labels, group_order, out_path, accuracy_by_cell=None):
    """Griglia 2x2 severita' (righe) x formato della risposta (colonne).

    Motivazione: nel confronto precedente MedQA vs MedicationQA la severita' e
    il formato variavano INSIEME, quindi un calo del PRR poteva dipendere da
    entrambi. Qui ogni colonna fissa il formato (e quindi la metrica di
    qualita'), cosi' il confronto verticale dentro una colonna isola la sola
    severita'.

    L'asse x e' condiviso all'interno di ciascuna colonna -- non tra colonne,
    perche' AccuracyMetric (binaria) e AlignScore (continua) non sono
    commensurabili e forzare la stessa scala suggerirebbe un confronto che non
    e' lecito fare."""
    cells = {}
    for ds_name, cfg in SEVERITY_DATASETS.items():
        ds_subset = df[(df["dataset"] == ds_name) & (df["paper_label"].isin(labels))]
        if ds_subset.empty:
            continue
        pivot = ds_subset.pivot_table(index="paper_label", columns="model", values="value", aggfunc="first")
        pivot = pivot.reindex(columns=group_order)
        pivot["Mean"] = pivot.mean(axis=1)
        cells[ds_name] = pivot

    if not cells:
        print(f"!!! Nessun dato per {out_path}, salto.")
        return

    # Ordine comune dei metodi in tutti e quattro i pannelli, cosi' che la
    # stessa riga corrisponda sempre allo stesso metodo.
    combined = sum(c["Mean"] for c in cells.values()) / len(cells)
    common_order = combined.sort_values(ascending=True).index

    severities = ["alta", "bassa"]
    formats = ["MCQ", "libera"]
    by_position = {}
    for ds_name, cfg in SEVERITY_DATASETS.items():
        by_position[(cfg["severity"], cfg["answer_format"])] = ds_name

    fig, axes = plt.subplots(2, 2, figsize=(17, max(9, len(common_order) * 0.45)), sharey=True)
    for r, sev in enumerate(severities):
        # Scala x condivisa per colonna (stesso formato = stessa metrica).
        for c, fmt in enumerate(formats):
            ax = axes[r][c]
            ds_name = by_position.get((sev, fmt))
            if ds_name is None or ds_name not in cells:
                ax.set_visible(False)
                continue
            pivot = cells[ds_name].reindex(common_order)
            pivot.plot(kind="barh", ax=ax, width=0.8, legend=False)
            title = f"{ds_name} -- severita' {sev}, formato {fmt}"
            if accuracy_by_cell and ds_name in accuracy_by_cell:
                title += f"\n{accuracy_by_cell[ds_name]}"
            ax.set_title(title, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Mean PRR (raw, max_rejection=0.5)", fontsize=8)

    for c, fmt in enumerate(formats):
        col_axes = [axes[r][c] for r in range(2)
                    if by_position.get((severities[r], fmt)) in cells]
        if len(col_axes) == 2:
            lo = min(a.get_xlim()[0] for a in col_axes)
            hi = max(a.get_xlim()[1] for a in col_axes)
            for a in col_axes:
                a.set_xlim(lo, hi)

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][-1].legend(handles, legend_labels, loc="lower right", fontsize=7)
    fig.suptitle("PRR per metodo UQ: griglia severita' clinica x formato della risposta")
    _apply_label_margin(fig, common_order, cap=0.42, floor=0.18, per_char=0.008,
                        bottom=0.08, top=0.91, wspace=0.05, hspace=0.22)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato: {out_path}")


# Modello scelto per il confronto quantizzato vs non-quantizzato (vedi
# --run_quant_comparison in main()). MedGemma-4B-it e Gemma3-4B-it sono
# esclusi a priori: producono logit NaN sotto bitsandbytes 4-bit (vedi
# NO_QUANT_MODELS) e per questo vengono SEMPRE caricati in bf16, quindi un
# confronto quantizzato/non-quantizzato su di loro non e' fattibile. Tra i
# modelli rimasti scegliamo LFM2-1.2B: la quantizzazione comprime
# maggiormente un modello con piu' parametri, quindi un eventuale effetto
# sulla qualita' delle stime di incertezza e' piu' probabilmente misurabile
# rispetto al modello da 350M.
QUANT_COMPARE_MODEL_DEFAULT = "LFM2-1.2B"


def plot_pareto_frontier(df, out_path,
                         cost_col="sec_full_per_instance", value_col="value",
                         label_col="paper_label"):
    """Scatter costo pieno per istanza (x, scala log) vs PRR medio (y), con
    evidenziata la frontiera di Pareto.

    Una tecnica e' DOMINATA se ne esiste un'altra insieme piu' economica e piu'
    affidabile: sceglierla non e' mai razionale. Le tecniche non dominate
    formano la frontiera, cioe' la curva dei migliori compromessi possibili: a
    ogni budget di calcolo la figura dice qual e' la migliore affidabilita'
    raggiungibile e con quale metodo. E' la figura che risponde alla domanda
    del deployment on-device, dove il budget e' fissato dall'hardware."""
    if df is None or df.empty:
        print(f"!!! Nessun dato per {out_path}, salto.")
        return None

    data = df.dropna(subset=[cost_col, value_col]).sort_values(cost_col)
    if data.empty:
        print(f"!!! Nessun punto valido per {out_path}, salto.")
        return None

    # Frontiera: scorrendo per costo crescente, un punto e' sulla frontiera se
    # supera il PRR massimo visto finora (nessuno piu' economico e' migliore).
    frontier_idx, best = [], -np.inf
    for idx, row in data.iterrows():
        if row[value_col] > best:
            frontier_idx.append(idx)
            best = row[value_col]
    frontier = data.loc[frontier_idx]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(data[cost_col], data[value_col], s=32, color="0.6",
               label="metodi dominati", zorder=2)
    ax.scatter(frontier[cost_col], frontier[value_col], s=64, color="tab:red",
               label="frontiera di Pareto", zorder=3)
    ax.plot(frontier[cost_col], frontier[value_col], color="tab:red",
            linewidth=1.0, linestyle="--", zorder=1)

    for _, row in data.iterrows():
        ax.annotate(str(row[label_col]), (row[cost_col], row[value_col]),
                    textcoords="offset points", xytext=(5, 3), fontsize=6.5)

    ax.set_xscale("log")
    ax.set_xlabel("Costo pieno standalone per istanza (secondi, scala log)")
    ax.set_ylabel("Mean PRR (raw, max_rejection=0.5)")
    ax.set_title("Frontiera di Pareto: affidabilita' dell'incertezza vs costo di calcolo")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.text(0.01, 0.01,
             "Costo pieno = generazione greedy + K campioni (se la tecnica li richiede) + forward NLI "
             "(se li richiede) + aritmetica dello stimatore,\nmediato sui modelli e sui dataset. Un "
             "metodo e' dominato se un altro e' insieme piu' economico e piu' affidabile.",
             fontsize=7, va="bottom")
    fig.subplots_adjust(bottom=0.16)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato: {out_path}")
    print("Metodi sulla frontiera di Pareto:")
    print(frontier[[label_col, cost_col, value_col]].to_string(index=False))
    return frontier


def run_dataset_section(section_name, datasets_cfg, args, hf_token, paper_label_by_str,
                        results_basename, models=None, estimators_factory=None,
                        content_transform=None, max_new_tokens_override=None,
                        min_datasets=1):
    """Esegue un insieme di dataset su un insieme di modelli, con checkpoint e
    ripresa, e ritorna (raw_df, stats_df) gia' mappati sulle etichette del
    paper.

    Estratta perche' la griglia di severita' e la sezione verbalized fanno
    esattamente la stessa cosa della pipeline principale, cambiando solo quali
    dataset, quali stimatori e con che prompt: duplicare il loop tre volte
    significherebbe dover ricordare di applicare ogni correzione futura in tre
    punti diversi."""
    models = models or MODELS
    print(f"\n--- {section_name} ---")

    examples_by_ds = {}
    for ds_name, cfg in datasets_cfg.items():
        n_test = args.n_test_samples if args.n_test_samples is not None else cfg["n_test"]
        print(f"Caricamento {ds_name} (n_test={n_test})...")
        try:
            examples = cfg["loader"](n_test, SEED, cache_dir=args.datasets_cache_dir)
            print(f"{ds_name}: {len(examples)} esempi caricati.")
            examples_by_ds[ds_name] = examples
        except Exception:
            print(f"!!! Caricamento di {ds_name} fallito, salto questo dataset.")
            traceback.print_exc()

    if len(examples_by_ds) < min_datasets:
        print(f"!!! Solo {len(examples_by_ds)} dataset caricati (ne servono almeno "
              f"{min_datasets}), salto la sezione.")
        return None, None

    final_path = os.path.join(args.results_dir, f"{results_basename}.csv")
    stats_path = os.path.join(args.results_dir, f"{results_basename}_instance_stats.csv")
    per_inst_path = os.path.join(args.results_dir, f"{results_basename}_per_instance.csv")

    metrics_dfs, stats_dfs, per_inst_dfs = [], [], []
    already_done = set()
    if os.path.exists(final_path):
        try:
            existing = pd.read_csv(final_path)
            already_done = set(zip(existing["dataset"], existing["model"]))
            metrics_dfs.append(existing)
            print(f"Combinazioni gia' completate in {results_basename}:", already_done)
        except Exception as e:
            print(f"{results_basename}.csv illeggibile ({e}) -- riparto senza skip.")
    if os.path.exists(stats_path):
        try:
            stats_dfs.append(pd.read_csv(stats_path))
        except Exception as e:
            print(f"{results_basename}_instance_stats.csv illeggibile ({e}).")

    for model_name, model_id in models.items():
        pending = [d for d in examples_by_ds if (d, model_name) not in already_done]
        if not pending:
            continue
        try:
            model = load_whitebox_model(model_id, args.cache_dir, hf_token=hf_token,
                                        use_quantization=uses_quantization(model_name))
        except Exception:
            print(f"!!! Caricamento di {model_name} fallito, salto i suoi dataset.")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        for dataset_name in pending:
            cfg = datasets_cfg[dataset_name]
            df, _, stats_df, per_inst_df = run_model_on_dataset(
                model, model_name, dataset_name, examples_by_ds[dataset_name], cfg, args,
                paper_label_by_str, use_chat_template=(model_name in CHAT_TEMPLATE_MODELS),
                estimators_factory=estimators_factory,
                content_transform=content_transform,
                max_new_tokens_override=max_new_tokens_override,
            )
            if df is not None:
                metrics_dfs.append(df)
                pd.concat(metrics_dfs, ignore_index=True).to_csv(final_path, index=False)
                print(f"Checkpoint {results_basename} salvato dopo {model_name}/{dataset_name}.")
            if stats_df is not None:
                stats_dfs.append(stats_df)
                pd.concat(stats_dfs, ignore_index=True).to_csv(stats_path, index=False)
            if per_inst_df is not None:
                per_inst_dfs.append(per_inst_df)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if not metrics_dfs:
        print(f"!!! Nessuna combinazione completata per {section_name}.")
        return None, None

    combined = pd.concat(metrics_dfs, ignore_index=True)
    raw_df = (combined[combined["ue_metric"] == "prr_0.5"].copy()
              if "ue_metric" in combined.columns else combined.copy())
    raw_df["paper_label"] = raw_df["estimator"].map(paper_label_by_str).fillna(raw_df["estimator"])

    stats_all = pd.concat(stats_dfs, ignore_index=True) if stats_dfs else None
    if stats_all is not None:
        # Porta gli intervalli di confidenza accanto ai valori PRR, cosi' che
        # le funzioni di plotting possano disegnare le barre d'errore.
        raw_df = raw_df.merge(
            stats_all[["model", "dataset", "estimator", "prr_ci_low", "prr_ci_high",
                       "mean_quality", "quality_metric", "nan_rate", "silent_failure_rate"]],
            on=["model", "dataset", "estimator"], how="left",
        )
    raw_df.to_csv(os.path.join(args.results_dir, f"{results_basename}_mapped.csv"), index=False)

    if per_inst_dfs:
        pd.concat(per_inst_dfs, ignore_index=True).to_csv(per_inst_path, index=False)

    return raw_df, stats_all


def aggregate_across_datasets(raw_df):
    """Media del PRR di ogni metodo sui dataset, con l'intervallo di confidenza
    propagato da quelli dei singoli dataset.

    Le figure aggregate del paper mediano il PRR su tutti i task; l'intervallo
    su quella media non puo' essere preso da un singolo dataset. Qui si
    converte ogni intervallo bootstrap in un errore standard (semi-ampiezza /
    1.96), si combinano come errori indipendenti -- se(media) = sqrt(somma dei
    quadrati) / k -- e si torna a un intervallo al 95%.

    L'ipotesi di indipendenza tra dataset e' ragionevole (sono campioni
    disgiunti da fonti diverse) ma va dichiarata: se i dataset fossero
    correlati, l'intervallo risultante sarebbe leggermente ottimistico."""
    if "prr_ci_low" not in raw_df.columns:
        return raw_df.groupby(["model", "paper_label"], as_index=False)["value"].mean()

    tmp = raw_df.copy()
    se = (tmp["prr_ci_high"] - tmp["prr_ci_low"]) / (2 * 1.96)
    tmp["_se2"] = se ** 2

    # Aggregazione tutta dentro .agg() invece che con .apply(): evita la
    # dipendenza dal parametro include_groups, che esiste solo da pandas 2.2.
    # "count" ignora i NaN, quindi conta solo i dataset con un intervallo
    # valido.
    agg = tmp.groupby(["model", "paper_label"], as_index=False).agg(
        value=("value", "mean"),
        n_datasets=("value", "size"),
        _se2_sum=("_se2", "sum"),
        _se_count=("_se2", "count"),
    )
    denom = agg["_se_count"].replace(0, np.nan)
    se_comb = np.sqrt(agg["_se2_sum"]) / denom
    agg["prr_ci_low"] = agg["value"] - 1.96 * se_comb
    agg["prr_ci_high"] = agg["value"] + 1.96 * se_comb
    return agg.drop(columns=["_se2_sum", "_se_count"])


def build_accuracy_table(stats_df, results_dir, filename="accuracy_table.csv"):
    """Tabella modello x dataset della qualita' media delle generazioni.

    Serve a rendere leggibile ogni figura PRR: il PRR misura quanto bene
    l'incertezza ordina risposte giuste e sbagliate, quindi ha senso solo se
    nel campione ci sono entrambe. Con accuracy vicina alla soglia del caso il
    modello sta tirando a indovinare e nessun segnale interno puo' predire
    l'esito, quindi il PRR crolla per TUTTI i metodi insieme e un valore basso
    non dice nulla sulla qualita' del metodo; con accuracy vicina al 100% ci
    sono pochissimi errori da trovare e la stima e' dominata dal rumore. Senza
    questa tabella accanto, un confronto di PRR tra modelli con accuracy molto
    diverse confronta misure prese in regimi diversi."""
    if stats_df is None or stats_df.empty:
        return None
    table = stats_df.pivot_table(index="model", columns="dataset",
                                 values="mean_quality", aggfunc="first")
    metrics = stats_df.groupby("dataset")["quality_metric"].first()
    table.columns = [f"{c} ({metrics.get(c, '?')})" for c in table.columns]
    path = os.path.join(results_dir, filename)
    table.to_csv(path)
    print(f"Salvato: {path}")
    print(table.round(3).to_string())
    return table


def compute_rank_transfer(stats_df, results_dir, anchor=REPLICATION_ANCHOR_MODEL):
    """Kendall tau tra il ranking dei metodi UQ del modello-ancora a 7B e
    quello di ogni altro modello, piu' la figura tau vs numero di parametri.

    Risponde alla domanda "le conclusioni del benchmark, costruite su modelli
    da 7-12B, sopravvivono scendendo a 0.35-4B?". Tau vale 1 se i due ranking
    coincidono, 0 se sono scorrelati, -1 se sono invertiti: un tau che cala al
    calare della scala significa che la classifica dei metodi non trasferisce,
    ed e' esattamente il risultato che giustifica un benchmark dedicato ai
    modelli piccoli."""
    if stats_df is None or stats_df.empty:
        return None
    if anchor not in stats_df["model"].unique():
        print(f"!!! Modello-ancora {anchor} assente dai risultati, salto Kendall tau.")
        return None

    # Media del PRR su tutti i dataset: un solo ranking per modello.
    agg = stats_df.groupby(["model", "paper_label"], as_index=False)["prr"].mean()
    pivot = agg.pivot_table(index="paper_label", columns="model", values="prr")
    if anchor not in pivot.columns:
        return None

    rows = []
    for model_name in pivot.columns:
        if model_name == anchor:
            continue
        pair = pivot[[anchor, model_name]].dropna()
        if len(pair) < 3:
            print(f"Solo {len(pair)} metodi in comune tra {anchor} e {model_name}, salto.")
            continue
        tau, p_value = kendalltau(pair[anchor].rank(ascending=False),
                                  pair[model_name].rank(ascending=False))
        rows.append({
            "model": model_name,
            "params_B": MODEL_PARAMS_B.get(model_name, np.nan),
            "kendall_tau_vs_anchor": tau,
            "p_value": p_value,
            "n_methods_compared": len(pair),
        })

    if not rows:
        return None
    tau_df = pd.DataFrame(rows).sort_values("params_B")
    path = os.path.join(results_dir, "rank_transfer_kendall_tau.csv")
    tau_df.to_csv(path, index=False)
    print(f"Salvato: {path}")
    print(tau_df.round(3).to_string(index=False))

    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(tau_df["params_B"], tau_df["kendall_tau_vs_anchor"], "o-", color="tab:blue")
        for _, r in tau_df.iterrows():
            ax.annotate(r["model"], (r["params_B"], r["kendall_tau_vs_anchor"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(1, color="0.7", linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel("Parametri del modello (miliardi, scala log)")
        ax.set_ylabel(f"Kendall tau del ranking dei metodi UQ vs {anchor}")
        ax.set_title("Trasferimento del ranking dei metodi UQ al calare della scala")
        ax.set_ylim(-1.05, 1.05)
        fig.text(0.01, 0.01,
                 f"tau = 1: stesso ordine di preferenza dei metodi del modello a 7B; tau = 0: "
                 f"ranking scorrelato.\nIl PRR e' mediato su tutti i dataset disponibili per "
                 f"ciascun modello.", fontsize=7, va="bottom")
        fig.subplots_adjust(bottom=0.18)
        chart_path = os.path.join(results_dir, "fig_rank_transfer.png")
        plt.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"Salvato: {chart_path}")
    except Exception:
        print("!!! figura rank transfer fallita:")
        traceback.print_exc()

    return tau_df


def build_verbalized_estimators(style):
    """Stimatori per la sezione verbalized. `style` sceglie come il modello
    deve dichiarare la confidenza: "numeric" (un numero tra 0 e 1, letto da
    Verbalized1S via regex) o "linguistic" (una classe testuale tipo "High",
    mappata a un valore da Linguistic1S)."""
    if style == "numeric":
        return [Verbalized1S(confidence_regex=VERBALIZED_CONFIDENCE_REGEX,
                             name_postfix="_numeric")]
    if style == "linguistic":
        return [Linguistic1S(expressions=LINGUISTIC_EXPRESSIONS,
                             name_postfix="_linguistic")]
    raise ValueError(f"Stile verbalized sconosciuto: {style}")


def main():
    parser = argparse.ArgumentParser(
        description="Selective QA UQ benchmark (lm-polygraph) -- replica Sezione 5.1 Vashurin et al."
    )
    parser.add_argument("--n_test_samples", type=int, default=None,
                         help="Se specificato, sovrascrive n_test per TUTTI i dataset (utile per smoke test rapidi).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_rejection", type=float, default=0.5)
    parser.add_argument("--results_dir", type=str, default=os.environ.get("RESULTS_DIR", "/workspace/results"))
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HOME", "/llms"))
    # Cache separata per i dataset (load_dataset), diversa da --cache_dir
    # (usata solo per i pesi dei modelli). /llms sembra essere una cache
    # condivisa a livello di cluster: il primo run con GSM8k ha fallito con
    # un PermissionError sul lock file "/llms/datasets/..." (probabilmente
    # gia' scritto da un altro utente/processo con permessi diversi). Uso
    # una directory privata sotto /workspace (il repo di Filo) per evitare
    # qualunque conflitto di permessi sulla cache condivisa.
    parser.add_argument("--datasets_cache_dir", type=str,
                         default=os.environ.get("HF_DATASETS_CACHE", "/workspace/hf_datasets_cache"))
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                         help="Numero di ricampionamenti bootstrap per gli intervalli di confidenza "
                              "sul PRR. Non costa GPU (ricampiona valori gia' calcolati), ma su molti "
                              "stimatori x dataset il costo CPU si somma: abbassarlo per prove rapide "
                              "(default: %(default)s).")
    parser.add_argument("--run_severity_grid", action="store_true",
                         help="Esegue la griglia 2x2 severita' x formato: MedQAbstain-LT/Safe (MCQ) e "
                              "MedicationQA/MedQuAD (risposta libera), su tutti i modelli "
                              "(fig_severity_grid.png + tabella silent failure rate).")
    parser.add_argument("--run_verbalized", action="store_true",
                         help="Esegue la sezione dei metodi verbalized (Verbalized1S, Linguistic1S) con "
                              "un prompt che chiede esplicitamente la confidenza e piu' token per "
                              "generarla, piu' la tabella dei parse-failure rate.")
    parser.add_argument("--verbalized_max_new_tokens", type=int, default=40,
                         help="max_new_tokens per la sezione verbalized: deve bastare a contenere la "
                              "risposta E la riga 'Confidence: ...' (default: %(default)s).")
    parser.add_argument("--run_quant_comparison", action="store_true",
                         help="In piu' rispetto alla pipeline principale, esegue anche il confronto "
                              "quantizzato (4-bit) vs non-quantizzato (bf16) su un modello "
                              "(--quant_compare_model) sui 4 dataset principali.")
    parser.add_argument("--quant_compare_model", type=str, default=QUANT_COMPARE_MODEL_DEFAULT,
                         choices=list(MODELS.keys()),
                         help="Modello su cui eseguire --run_quant_comparison (default: %(default)s). "
                              "MedGemma-4B-it/Gemma3-4B-it non sono utilizzabili: producono NaN sotto "
                              "quantizzazione 4-bit (vedi NO_QUANT_MODELS).")
    args = parser.parse_args()

    print_banner()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None and any(m in GATED_MODELS for m in MODELS):
        print("HF_TOKEN non impostato ma MODELS include modelli gated "
              f"({', '.join(m for m in MODELS if m in GATED_MODELS)}). "
              "Il download fallira' senza accettare la licenza + passare un token. "
              "Esporta HF_TOKEN nella shell prima di lanciare sbatch_script.sh.")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.datasets_cache_dir, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    write_excluded_methods_report(args.results_dir)

    # Nome-interno (str(estimator)) -> etichetta del paper, per rimappare i
    # risultati alle Figure 2/3 / Tabella 6 senza toccare extract_prr_table.
    paper_label_by_str = {
        str(m["factory"]()): m["paper_label"] for m in PAPER_METHODS if callable(m["factory"])
    }

    print("\n--- Caricamento dataset (Sezione 5.1: CoQA, TriviaQA, MMLU, GSM8k) ---")
    dataset_examples = {}
    for ds_name, cfg in DATASETS.items():
        n_test = args.n_test_samples if args.n_test_samples is not None else cfg["n_test"]
        print(f"Caricamento {ds_name} (n_test={n_test})...")
        try:
            examples = cfg["loader"](n_test, SEED, cache_dir=args.datasets_cache_dir)
            print(f"{ds_name}: {len(examples)} esempi caricati.")
            print(examples[0]["content"][:500])
            print("Riferimento:", examples[0]["reference"])
            dataset_examples[ds_name] = examples
        except Exception:
            # Un dataset rotto (es. problemi di cache/rete) non deve far
            # cadere l'intero run: lo saltiamo e continuiamo con gli altri.
            print(f"!!! Caricamento di {ds_name} fallito, questo dataset sara' saltato per tutti i modelli.")
            traceback.print_exc()

    if not dataset_examples:
        print("Nessun dataset caricato con successo -- niente da fare.")
        return

    final_path = os.path.join(args.results_dir, "results_final.csv")
    results_df_existing = None
    already_done_pairs = set()

    if os.path.exists(final_path):
        try:
            results_df_existing = pd.read_csv(final_path)
            already_done_pairs = set(zip(results_df_existing["dataset"], results_df_existing["model"]))
            print("Combinazioni dataset x modello gia' completate:", already_done_pairs)
        except Exception as e:
            print(f"results_final.csv presente ma illeggibile ({e}) -- riparto senza skip.")
            results_df_existing = None
    else:
        print("Nessun risultato precedente trovato -- primo avvio, eseguo tutte le combinazioni.")

    all_metrics_dfs = [results_df_existing] if results_df_existing is not None else []

    timing_final_path = os.path.join(args.results_dir, "estimator_timings.csv")
    all_timing_dfs = []
    if os.path.exists(timing_final_path):
        try:
            all_timing_dfs = [pd.read_csv(timing_final_path)]
        except Exception as e:
            print(f"estimator_timings.csv presente ma illeggibile ({e}) -- riparto senza skip.")

    # Statistiche a livello di istanza (accuracy, intervalli bootstrap, silent
    # failure rate) e valori grezzi per-istanza. Questi ultimi vengono salvati
    # su disco per poter rifare bootstrap, test appaiati e analisi di soglia
    # senza rieseguire nulla sulla GPU.
    all_stats_dfs = []
    all_per_instance_dfs = []

    # Loop esterno sui MODELLI (non sui dataset): caricare un modello da 4B
    # e' l'operazione piu' costosa, quindi lo facciamo una volta sola e gli
    # facciamo girare tutti e 4 i dataset prima di scaricarlo, invece di
    # ricaricarlo per ogni dataset.
    for model_name, model_id in MODELS.items():
        pending_datasets = [d for d in dataset_examples if (d, model_name) not in already_done_pairs]
        if not pending_datasets:
            print(f"\n=== Modello: {model_name} -- tutti i dataset gia' completati, salto. ===")
            continue

        print(f"\n=== Modello: {model_name} ({model_id}) -- dataset da eseguire: {pending_datasets} ===")
        model_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            model = load_whitebox_model(model_id, args.cache_dir, hf_token=hf_token,
                                         use_quantization=uses_quantization(model_name))
            log_gpu_mem(f"{model_name} loaded")
        except Exception:
            print(f"!!! Caricamento di {model_name} fallito, salto tutti i suoi dataset.")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        for dataset_name in pending_datasets:
            cfg = DATASETS[dataset_name]
            df, timing_df, stats_df, per_inst_df = run_model_on_dataset(
                model, model_name, dataset_name, dataset_examples[dataset_name], cfg, args,
                paper_label_by_str, use_chat_template=(model_name in CHAT_TEMPLATE_MODELS),
            )
            if df is not None:
                all_metrics_dfs.append(df)
            if timing_df is not None:
                all_timing_dfs.append(timing_df)
            if stats_df is not None:
                all_stats_dfs.append(stats_df)
            if per_inst_df is not None:
                all_per_instance_dfs.append(per_inst_df)

            if all_metrics_dfs:
                pd.concat(all_metrics_dfs, ignore_index=True).to_csv(
                    os.path.join(args.results_dir, "results_partial.csv"), index=False
                )
                print(f"Checkpoint salvato dopo {model_name}/{dataset_name}.")
            if all_timing_dfs:
                pd.concat(all_timing_dfs, ignore_index=True).to_csv(
                    os.path.join(args.results_dir, "estimator_timings_partial.csv"), index=False
                )
            if all_stats_dfs:
                pd.concat(all_stats_dfs, ignore_index=True).to_csv(
                    os.path.join(args.results_dir, "instance_level_stats.csv"), index=False
                )

        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Tempo totale {model_name} (tutti i dataset): {time.time() - model_start:.1f}s")

    if not all_metrics_dfs:
        print("Nessuna combinazione completata con successo -- niente da salvare/plottare.")
        return

    results_df = pd.concat(all_metrics_dfs, ignore_index=True)
    results_df.to_csv(final_path, index=False)
    print(results_df.head(20))

    if all_timing_dfs:
        timing_df = pd.concat(all_timing_dfs, ignore_index=True)
        timing_df.to_csv(timing_final_path, index=False)

    stats_all = pd.concat(all_stats_dfs, ignore_index=True) if all_stats_dfs else None
    if stats_all is not None:
        stats_all.to_csv(os.path.join(args.results_dir, "instance_level_stats.csv"), index=False)
    if all_per_instance_dfs:
        pd.concat(all_per_instance_dfs, ignore_index=True).to_csv(
            os.path.join(args.results_dir, "per_instance_scores.csv"), index=False
        )

    raw_df = results_df[results_df["ue_metric"] == "prr_0.5"].copy() if "ue_metric" in results_df.columns else results_df.copy()
    raw_df["paper_label"] = raw_df["estimator"].map(paper_label_by_str).fillna(raw_df["estimator"])

    if stats_all is not None:
        raw_df = raw_df.merge(
            stats_all[["model", "dataset", "estimator", "prr_ci_low", "prr_ci_high",
                       "mean_quality", "quality_metric", "silent_failure_rate"]],
            on=["model", "dataset", "estimator"], how="left",
        )

    # Aggiunge le righe alias (es. "BB Semantic Entropy" = stesso valore di
    # "Semantic Entropy") cosi' che le tabelle finali abbiano esattamente le
    # etichette delle Figure 2/3, senza ricalcolare nulla.
    alias_rows = []
    for m in PAPER_METHODS:
        if isinstance(m["factory"], str) and m["factory"].startswith("alias:"):
            target_label = m["factory"].split("alias:", 1)[1]
            src = raw_df[raw_df["paper_label"] == target_label]
            for _, row in src.iterrows():
                alias_rows.append({**row.to_dict(), "paper_label": m["paper_label"]})
    if alias_rows:
        raw_df = pd.concat([raw_df, pd.DataFrame(alias_rows)], ignore_index=True)

    raw_df.to_csv(os.path.join(args.results_dir, "results_paper_mapped.csv"), index=False)

    # Accuracy di base per modello e dataset: indispensabile per leggere ogni
    # figura PRR (vedi build_accuracy_table per il perche').
    build_accuracy_table(stats_all, args.results_dir)

    # Kendall tau: quanto il ranking dei metodi UQ del modello a 7B (regime di
    # scala del paper) si conserva scendendo di scala.
    if stats_all is not None:
        stats_labeled = stats_all.copy()
        stats_labeled["paper_label"] = (
            stats_labeled["estimator"].map(paper_label_by_str).fillna(stats_labeled["estimator"])
        )
        try:
            compute_rank_transfer(stats_labeled, args.results_dir)
        except Exception:
            print("!!! calcolo Kendall tau fallito:")
            traceback.print_exc()

    model_order = [m for m in MODELS.keys() if m in raw_df["model"].unique()]
    dataset_order = [d for d in DATASETS.keys() if d in raw_df["dataset"].unique()]

    # Mean PRR aggregato su tutti i task di selective QA, esattamente come da
    # didascalia originale delle Figure 2/3 del paper ("aggregated over all
    # selective QA tasks for each ... LLM separately"), con l'intervallo di
    # confidenza propagato dai singoli dataset.
    agg_df = aggregate_across_datasets(raw_df)

    figure_a_labels = [m["paper_label"] for m in PAPER_METHODS if m["figure"] in ("A", "AB") and m["factory"] is not None]
    figure_b_labels = [m["paper_label"] for m in PAPER_METHODS if m["figure"] in ("B", "AB") and m["factory"] is not None]

    def plot_paper_figure(labels, title, note, out_name):
        plot_prr_bars(
            agg_df, labels, model_order, title, note,
            os.path.join(args.results_dir, out_name),
            "Mean PRR (raw, max_rejection=0.5, aggregato su CoQA/TriviaQA/MMLU/GSM8k)",
        )

    try:
        plot_paper_figure(
            figure_a_labels,
            "Mean PRR aggregato su selective QA (~ Fig. 2 Vashurin et al., white-box full-access)",
            FIGURE_A_MODELS_NOTE,
            "fig_a_white_box.png",
        )
    except Exception:
        print("!!! fig_a_white_box.png fallito:")
        traceback.print_exc()

    try:
        plot_paper_figure(
            figure_b_labels,
            "Mean PRR aggregato su selective QA (~ Fig. 3 Vashurin et al., reflexive/black-box)",
            FIGURE_B_MODELS_NOTE,
            "fig_b_reflexive.png",
        )
    except Exception:
        print("!!! fig_b_reflexive.png fallito:")
        traceback.print_exc()

    # Tabella stile Tabella 6 del paper, UNA per ciascun nostro modello:
    # righe = metodi di Figura A, colonne = i 4 dataset (CoQA/TriviaQA/MMLU/
    # GSM8k) + Mean Rank + Mean PRR -- stessa identica struttura del paper
    # (che pero' la applica solo a StableLM 2 12B), qui replicata per ognuno
    # dei nostri 4 modelli.
    for model_name in model_order:
        try:
            subset = raw_df[(raw_df["model"] == model_name) & (raw_df["paper_label"].isin(figure_a_labels))]
            table6 = subset.pivot_table(index="paper_label", columns="dataset", values="value", aggfunc="first")
            table6 = table6.reindex(columns=dataset_order)
            ranks = table6.rank(axis=0, ascending=False)
            table6["Mean Rank"] = ranks.mean(axis=1)
            table6["Mean PRR"] = table6[dataset_order].mean(axis=1)
            table6 = table6.sort_values("Mean PRR", ascending=False)
            table6_path = os.path.join(args.results_dir, f"table6_style_{model_name}.csv")
            table6.to_csv(table6_path)
            print(f"Salvato: {table6_path}")
        except Exception:
            print(f"!!! table6_style_{model_name}.csv fallito:")
            traceback.print_exc()

    # Costi di calcolo: due tabelle e due figure. Vedi il blocco di commento
    # sopra TimedUEManager per il motivo per cui servono DUE nozioni di costo.
    try:
        if all_timing_dfs:
            timing_all = pd.concat(all_timing_dfs, ignore_index=True).drop_duplicates(
                subset=["model", "dataset", "estimator"], keep="last"
            )
            timing_all.to_csv(os.path.join(args.results_dir, "estimator_timings.csv"), index=False)

            # Tabella 1 -- costo marginale (aritmetica sola), il numero utile a
            # chi calcola molte tecniche insieme sulla stessa generazione.
            timing_pivot = timing_all.pivot_table(
                index="paper_label", columns="model", values="seconds", aggfunc="sum"
            ).reindex(columns=model_order)
            timing_pivot["Total"] = timing_pivot.sum(axis=1)
            timing_pivot = timing_pivot.sort_values("Total", ascending=False)
            timing_path = os.path.join(args.results_dir, "estimator_timing_table.csv")
            timing_pivot.to_csv(timing_path)
            print(f"Salvato: {timing_path}")

            # Tabella 2 -- costo pieno standalone per istanza + memoria di
            # picco: e' questa la tabella da citare per la raccomandazione
            # on-device.
            cost_table = timing_all.groupby("paper_label", as_index=False).agg(
                sec_marginal_per_instance=("seconds_marginal_per_instance", "mean"),
                sec_full_per_instance=("seconds_full_per_instance", "mean"),
                needs_sampling=("needs_sampling", "max"),
                needs_nli=("needs_nli", "max"),
                peak_memory_gb=("peak_memory_gb", "max"),
            ).sort_values("sec_full_per_instance", ascending=False)
            cost_path = os.path.join(args.results_dir, "estimator_cost_table.csv")
            cost_table.to_csv(cost_path, index=False)
            print(f"Salvato: {cost_path}")

            # Scala log sull'asse x: i tempi coprono diversi ordini di
            # grandezza, in scala lineare tutte le barre tranne una sarebbero
            # invisibili.
            plot_pivot = timing_pivot.drop(columns=["Total"]).loc[
                timing_pivot.sort_values("Total", ascending=True).index
            ]
            fig, ax = plt.subplots(figsize=(10, max(8, len(plot_pivot) * 0.4)))
            plot_pivot.plot(kind="barh", ax=ax, width=0.8, logx=True)
            ax.set_xlabel("Tempo di sola aritmetica dello stimatore, sommato su batch e dataset "
                          "(secondi, scala log)")
            ax.set_title("Costo MARGINALE per metodo UQ e modello")
            ax.legend(loc="lower right", fontsize=8)
            fig.text(0.01, 0.01,
                     "Costo marginale = solo il calcolo dello stimatore su statistiche gia' pronte. "
                     "NON e' il costo di eseguire la tecnica da sola: per quello vedi la frontiera "
                     "di Pareto e estimator_cost_table.csv.", fontsize=7, va="bottom")
            fig.subplots_adjust(bottom=0.14)
            timing_chart_path = os.path.join(args.results_dir, "estimator_timing_chart.png")
            plt.savefig(timing_chart_path, dpi=150)
            plt.close(fig)
            print(f"Salvato: {timing_chart_path}")

            # Frontiera di Pareto costo/affidabilita'.
            try:
                pareto_source = agg_df.groupby("paper_label", as_index=False)["value"].mean()
                pareto = pareto_source.merge(
                    cost_table[["paper_label", "sec_full_per_instance"]],
                    on="paper_label", how="inner",
                ).dropna(subset=["value", "sec_full_per_instance"])
                plot_pareto_frontier(
                    pareto,
                    os.path.join(args.results_dir, "fig_pareto_cost_quality.png"),
                )
            except Exception:
                print("!!! frontiera di Pareto fallita:")
                traceback.print_exc()
    except Exception:
        print("!!! tabelle/grafici dei costi falliti:")
        traceback.print_exc()


    # -------------------------------------------------------------------
    # Sezione extra 1: griglia 2x2 severita' clinica x formato della risposta.
    #
    # Sostituisce il precedente confronto MedQA vs MedicationQA, in cui
    # severita' e formato variavano insieme e non erano quindi separabili.
    # Le quattro celle sono MedQAbstain-LT/Safe (MCQ, stessa metrica) e
    # MedicationQA/MedQuAD (risposta libera, stessa metrica): il confronto di
    # severita' si legge a parita' di formato.
    # -------------------------------------------------------------------
    if args.run_severity_grid:
        try:
            severity_raw, severity_stats = run_dataset_section(
                "Griglia severita' x formato", SEVERITY_DATASETS, args, hf_token,
                paper_label_by_str, "results_severity_grid", min_datasets=2,
            )
            if severity_raw is not None:
                severity_model_order = [m for m in MODELS if m in severity_raw["model"].unique()]

                acc_table = build_accuracy_table(
                    severity_stats, args.results_dir, "accuracy_table_severity_grid.csv"
                )
                # Accuracy di ogni cella, mostrata nel titolo del pannello:
                # senza di essa un PRR basso non e' distinguibile da "modello
                # che tira a indovinare".
                accuracy_by_cell = {}
                if severity_stats is not None:
                    for ds_name, grp in severity_stats.groupby("dataset"):
                        per_model = ", ".join(
                            f"{m}: {v:.2f}" for m, v in
                            grp.groupby("model")["mean_quality"].first().items()
                        )
                        accuracy_by_cell[ds_name] = f"qualita' media -- {per_model}"

                plot_severity_grid(
                    severity_raw, figure_a_labels, severity_model_order,
                    os.path.join(args.results_dir, "fig_severity_grid.png"),
                    accuracy_by_cell=accuracy_by_cell,
                )

                # Silent failure rate: quanti degli errori finiscono nel decile
                # piu' confidente. E' il numero che conta davvero in clinica e
                # che il PRR medio non mostra.
                if severity_stats is not None:
                    sfr = severity_stats.pivot_table(
                        index="paper_label", columns=["dataset", "model"],
                        values="silent_failure_rate", aggfunc="first",
                    )
                    sfr_path = os.path.join(args.results_dir, "silent_failure_rate.csv")
                    sfr.to_csv(sfr_path)
                    print(f"Salvato: {sfr_path}")
                print("Griglia severita' completata.")
        except Exception:
            print("!!! Griglia severita' fallita:")
            traceback.print_exc()

    # -------------------------------------------------------------------
    # Sezione extra 2: metodi verbalized.
    #
    # Verbalized1S e Linguistic1S leggono la confidenza dichiarata dal modello
    # dentro la generazione greedy CONDIVISA con tutti gli altri stimatori:
    # per usarli servono un prompt che chieda la confidenza e piu' token per
    # generarla, cioe' due modifiche che cambierebbero i valori di ogni altro
    # metodo. Girano quindi qui, in una sezione separata con la propria
    # configurazione, e i loro numeri non vanno mescolati con quelli della
    # pipeline principale.
    #
    # L'interesse specifico per i modelli piccoli: verbalizzare la propria
    # incertezza e' un comportamento che in letteratura emerge con la scala,
    # quindi il parse-failure rate (quante volte il modello non produce
    # nemmeno il formato richiesto) e' esso stesso un risultato.
    # -------------------------------------------------------------------
    if args.run_verbalized:
        for style in ("numeric", "linguistic"):
            try:
                verb_raw, verb_stats = run_dataset_section(
                    f"Metodi verbalized ({style})", DATASETS, args, hf_token,
                    paper_label_by_str, f"results_verbalized_{style}",
                    estimators_factory=lambda s=style: build_verbalized_estimators(s),
                    content_transform=lambda c, s=style: build_verbalized_content(c, s),
                    max_new_tokens_override=args.verbalized_max_new_tokens,
                )
                if verb_raw is None:
                    continue

                build_accuracy_table(
                    verb_stats, args.results_dir, f"accuracy_table_verbalized_{style}.csv"
                )

                if verb_stats is not None:
                    # Parse-failure rate: frazione di istanze in cui la
                    # confidenza non e' estraibile dal testo generato. Va letto
                    # INSIEME al PRR, perche' lm-polygraph converte i punteggi
                    # NaN in -1e7, cioe' li tratta come massima confidenza: un
                    # modello che non rispetta il formato non viene penalizzato
                    # dal PRR, e senza questa tabella il suo risultato
                    # sembrerebbe migliore di quanto sia.
                    pf = verb_stats.pivot_table(
                        index=["estimator", "dataset"], columns="model",
                        values="nan_rate", aggfunc="first",
                    )
                    pf_path = os.path.join(args.results_dir, f"parse_failure_rate_{style}.csv")
                    pf.to_csv(pf_path)
                    print(f"Salvato: {pf_path}")
                    print(f"Parse-failure rate ({style}):")
                    print(pf.round(3).to_string())

                verb_model_order = [m for m in MODELS if m in verb_raw["model"].unique()]
                verb_labels = sorted(verb_raw["paper_label"].dropna().unique())
                verb_agg = verb_raw.groupby(["model", "paper_label"], as_index=False)["value"].mean()
                plot_prr_bars(
                    verb_agg, verb_labels, verb_model_order,
                    f"Mean PRR metodi verbalized ({style}), aggregato su CoQA/TriviaQA/MMLU/GSM8k",
                    "I metodi verbalized girano con un prompt e un max_new_tokens diversi dalla "
                    "pipeline principale, quindi questi valori NON sono confrontabili con quelli "
                    "delle Figure A/B. Da leggere sempre insieme al parse-failure rate: le istanze "
                    "in cui la confidenza non e' estraibile vengono trattate da lm-polygraph come "
                    "massima confidenza, non come errore.",
                    os.path.join(args.results_dir, f"fig_verbalized_{style}.png"),
                    "Mean PRR (raw, max_rejection=0.5)",
                )
                print(f"Sezione verbalized ({style}) completata.")
            except Exception:
                print(f"!!! Sezione verbalized ({style}) fallita:")
                traceback.print_exc()


    # -------------------------------------------------------------------
    # Confronto extra 2: quantizzato (4-bit) vs non-quantizzato (bf16), su
    # un solo modello scelto (--quant_compare_model, default LFM2-1.2B --
    # vedi QUANT_COMPARE_MODEL_DEFAULT per la motivazione della scelta),
    # aggregato sugli stessi 4 dataset della pipeline principale (riusa
    # dataset_examples gia' caricato, nessun ricaricamento).
    # -------------------------------------------------------------------
    if args.run_quant_comparison:
        quant_model_name = args.quant_compare_model
        print(f"\n--- Confronto extra: quantizzato vs non-quantizzato ({quant_model_name}) ---")
        if quant_model_name in CHAT_TEMPLATE_MODELS:
            print(f"!!! {quant_model_name} produce NaN sotto quantizzazione 4-bit (vedi CHAT_TEMPLATE_MODELS), "
                  "confronto non eseguibile su questo modello -- salto.")
        else:
            try:
                model_id = MODELS[quant_model_name]
                quant_final_path = os.path.join(args.results_dir, "results_quant_comparison.csv")
                quant_metrics_dfs = []
                quant_stats_dfs = []
                quant_already_done = set()
                if os.path.exists(quant_final_path):
                    try:
                        existing = pd.read_csv(quant_final_path)
                        quant_already_done = set(zip(existing["dataset"], existing["model"]))
                        quant_metrics_dfs.append(existing)
                        print("Combinazioni quant gia' completate:", quant_already_done)
                    except Exception as e:
                        print(f"results_quant_comparison.csv illeggibile ({e}) -- riparto senza skip.")

                variants = [("quantized (4-bit nf4)", True), ("full precision (bf16)", False)]
                for variant_label, use_quant in variants:
                    pending = [d for d in dataset_examples if (d, variant_label) not in quant_already_done]
                    if not pending:
                        continue
                    try:
                        model = load_whitebox_model(model_id, args.cache_dir, hf_token=hf_token, use_quantization=use_quant)
                    except Exception:
                        print(f"!!! Caricamento di {quant_model_name} ({variant_label}) fallito, salto questa variante.")
                        traceback.print_exc()
                        continue

                    for dataset_name in pending:
                        cfg = DATASETS[dataset_name]
                        # variant_label prende il posto del nome del modello
                        # nei risultati (cosi' le due varianti compaiono come
                        # due serie da confrontare), ma il chat template va
                        # deciso sul modello VERO: --quant_compare_model puo'
                        # essere anche un instruction-tuned.
                        df, _, quant_stats_df, _ = run_model_on_dataset(
                            model, variant_label, dataset_name, dataset_examples[dataset_name], cfg, args,
                            paper_label_by_str,
                            use_chat_template=(quant_model_name in CHAT_TEMPLATE_MODELS),
                        )
                        if df is not None:
                            quant_metrics_dfs.append(df)
                            pd.concat(quant_metrics_dfs, ignore_index=True).to_csv(quant_final_path, index=False)
                            print(f"Checkpoint quant salvato dopo {variant_label}/{dataset_name}.")
                        if quant_stats_df is not None:
                            quant_stats_dfs.append(quant_stats_df)

                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

                if quant_metrics_dfs:
                    quant_df = pd.concat(quant_metrics_dfs, ignore_index=True)
                    quant_raw = (
                        quant_df[quant_df["ue_metric"] == "prr_0.5"].copy()
                        if "ue_metric" in quant_df.columns else quant_df.copy()
                    )
                    quant_raw["paper_label"] = quant_raw["estimator"].map(paper_label_by_str).fillna(quant_raw["estimator"])

                    # Intervalli bootstrap accanto ai valori, cosi' che il
                    # grafico del confronto mostri se il divario 4-bit vs bf16
                    # e' piu' grande del rumore di campionamento.
                    if quant_stats_dfs:
                        quant_stats_all = pd.concat(quant_stats_dfs, ignore_index=True)
                        quant_stats_all.to_csv(
                            os.path.join(args.results_dir, "results_quant_comparison_instance_stats.csv"),
                            index=False,
                        )
                        quant_raw = quant_raw.merge(
                            quant_stats_all[["model", "dataset", "estimator",
                                             "prr_ci_low", "prr_ci_high", "mean_quality"]],
                            on=["model", "dataset", "estimator"], how="left",
                        )
                        build_accuracy_table(quant_stats_all, args.results_dir,
                                             "accuracy_table_quant_comparison.csv")
                    quant_raw.to_csv(os.path.join(args.results_dir, "results_quant_comparison_mapped.csv"), index=False)

                    quant_agg = quant_raw.groupby(["model", "paper_label"], as_index=False)["value"].mean()
                    variant_order = [v for v, _ in variants if v in quant_agg["model"].unique()]
                    plot_prr_bars(
                        quant_agg, figure_a_labels, variant_order,
                        f"Mean PRR per metodo UQ -- {quant_model_name}: 4-bit vs bf16 "
                        "(aggregato su CoQA/TriviaQA/MMLU/GSM8k)",
                        f"Confronto quantizzazione su {quant_model_name} (vedi commento su "
                        "QUANT_COMPARE_MODEL_DEFAULT nel codice per la motivazione della scelta: "
                        "MedGemma-4B-it/Gemma3-4B-it producono NaN sotto 4-bit e sono esclusi a priori).",
                        os.path.join(args.results_dir, f"fig_quant_comparison_{quant_model_name}.png"),
                        "Mean PRR (raw, max_rejection=0.5)",
                        add_mean=False,
                    )
                    print("Confronto quantizzazione completato.")
                else:
                    print("!!! Nessuna combinazione quant completata con successo.")
            except Exception:
                print("!!! Confronto quantizzazione fallito:")
                traceback.print_exc()

    print("\nBENCHMARK COMPLETATO.")


if __name__ == "__main__":
    main()
