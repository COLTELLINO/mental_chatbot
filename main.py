import argparse
import copy
import gc
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from lm_polygraph.ue_metrics.ue_metric import get_random_scores, normalize_metric
from lm_polygraph.estimators import *

from analysis_lib import silent_failure_rate as _silent_failure_rate

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
    # che alla scala. Licenza Apache 2.0, non gated: nessun token necessario.
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
# Mistral-7B-Instruct-v0.2 NON e' gated (verificato via API HuggingFace:
# "gated": false, licenza Apache 2.0), quindi non compare qui: si scarica
# liberamente anche senza token.
GATED_MODELS = {"MedGemma-4B-it", "Gemma3-4B-it"}

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


# Riduzione automatica del batch per i modelli che stanno stretti nei 24GB
# della 3090.
#
# I pesi da soli non saturano mai la scheda: il picco arriva durante il
# campionamento dei K sample richiesti dagli stimatori a diversita'
# campionaria (SemanticEntropy, DegMat, Eccentricity, SAR, ...), dove logit e
# KV-cache di batch_size*K sequenze stanno in memoria insieme, piu' i tensori
# del modello NLI che gira sulla STESSA GPU. Quanto spazio resta per tutto
# questo dipende da quanto ne hanno gia' preso i pesi.
#
# La soglia sotto e' tarata sui fallimenti osservati, non derivata da un
# calcolo -- va rivista se si cambia GPU o si aggiungono modelli:
#   run 2026-09-01: Mistral-7B 4-bit (~3.5GB di pesi), batch 4 -> OOM su tutti
#                   e 4 i dataset principali.
#   run 2026-09-02: Mistral a batch 1 passa; falliscono pero' Gemma3-4B e
#                   MedGemma-4B in bf16 (~8GB di pesi) e la variante bf16 di
#                   LFM2-1.2B (~2.4GB), tutte ancora a batch 4.
#   sempre riusciti a batch 4: LFM2-350M e LFM2-1.2B quantizzati (0.2-0.6GB).
# Il confine osservato cade quindi tra 0.6GB e 2.4GB di pesi.
HEAVY_WEIGHTS_GB = 2.0
BYTES_PER_PARAM = {True: 0.5, False: 2.0}  # nf4 ~4 bit, bf16 = 2 byte

DEBERTA_BATCH_SIZE_DEFAULT = 10
DEBERTA_BATCH_SIZE_HEAVY = 2

# Override espliciti, per i casi che la regola sopra non prende. Ha la
# precedenza su tutto.
MODEL_BATCH_SIZE = {}


def uses_quantization(model_name):
    """4-bit nf4 per default; bf16 solo per i modelli che sotto quantizzazione
    producono NaN (vedi NO_QUANT_MODELS)."""
    return model_name not in NO_QUANT_MODELS


def weights_gb(model_name, quantized):
    """GB occupati dai soli pesi. Serve a stimare quanta memoria resta per il
    campionamento, non a essere esatto."""
    return MODEL_PARAMS_B.get(model_name, 1.0) * BYTES_PER_PARAM[bool(quantized)]


def is_memory_heavy(model_name, quantized):
    return weights_gb(model_name, quantized) >= HEAVY_WEIGHTS_GB


def batch_size_for(model_name, default_batch_size, quantized=True):
    """Batch di generazione: il default da riga di comando, ridotto a 1 per i
    modelli i cui pesi non lasciano spazio al campionamento."""
    if model_name in MODEL_BATCH_SIZE:
        return MODEL_BATCH_SIZE[model_name]
    return 1 if is_memory_heavy(model_name, quantized) else default_batch_size


def deberta_batch_size_for(model_name, quantized=True):
    """Batch del modello NLI, ridotto insieme a quello di generazione."""
    if is_memory_heavy(model_name, quantized):
        return DEBERTA_BATCH_SIZE_HEAVY
    return DEBERTA_BATCH_SIZE_DEFAULT


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
    self_test_mcq_metric,
)

# GSM8k e' escluso dai metodi verbalized: con --verbalized_max_new_tokens=40 il
# ragionamento non finisce mai prima della riga di confidenza, e nella run del
# 13/09 il parse-failure rate su GSM8k era 100% per TUTTI i modelli. Non e' un
# risultato sui modelli ma un limite del budget di token; eseguirlo costava
# ore di GPU per una cella vuota.
VERBALIZED_DATASETS = {k: v for k, v in DATASETS.items() if k != "GSM8k"}


# Corrispondenza fra i modelli delle Figure 2/3 del paper e i nostri: finisce
# nel report excluded_methods.md (le figure le disegna make_figures.py).
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
     "reason": "ESCLUSIONE DI DESIGN, non limitazione pratica. I metodi density-based stimano "
               "l'incertezza come distanza dalla distribuzione degli embeddings del TRAINING SET, "
               "che va quindi conservata a disposizione al momento dell'inferenza: e' esattamente "
               "cio' che un deployment on-device non puo' fare, ed e' il vincolo che questo lavoro "
               "assume. Un metodo che richiede di spedire sul telefono le statistiche del training "
               "e' fuori scope per costruzione, indipendentemente da quanto sia accurato. "
               "(Secondariamente: richiederebbe anche TrainingStatisticExtractionCalculator + "
               "EmbeddingsCalculator, non registrati di default in register_default_stat_calculators, "
               "con forward pass extra per modello. Escluso su richiesta esplicita, 2026-08-12.)"},
    {"paper_label": "RDE - Decoder", "figure": "A", "factory": None,
     "reason": "Stesso motivo di Mahalanobis Distance: e' density-based, quindi richiede di tenere "
               "a disposizione le statistiche del training set in inferenza -- incompatibile con il "
               "vincolo on-device che questo lavoro assume. Escluso su richiesta esplicita (2026-08-12)."},
    {"paper_label": "Relative Mahalanobis Distance - Decoder", "figure": "A", "factory": None,
     "reason": "Stesso motivo di Mahalanobis Distance: e' density-based, quindi richiede di tenere "
               "a disposizione le statistiche del training set in inferenza -- incompatibile con il "
               "vincolo on-device che questo lavoro assume. Escluso su richiesta esplicita (2026-08-12)."},
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


# Calcolatori che caricano un modello ausiliario, per la tabella dei costi.
_NLI_CALCULATORS = {"SemanticMatrixCalculator", "SemanticClassesCalculator",
                    "GreedyAlternativesNLICalculator"}
_CROSS_ENCODER_CALCULATORS = {"CrossEncoderSimilarityMatrixCalculator"}
_EXTRA_FORWARD_CALCULATORS = {"GreedyLMProbsCalculator", "PromptCalculator"}


def default_stat_calculators(cache_dir, deberta_batch_size=None):
    """Descrizioni dei calcolatori di statistiche usate da UEManager (vedi
    build_manager). Servono anche a ricostruire le dipendenze di ogni metodo."""
    return register_default_stat_calculators(
        model_type="Whitebox",
        language="en",
        hf_cache=cache_dir,
        output_attentions=False,
        # Nessuno dei 26 stimatori usa gli hidden state (servirebbero solo ai
        # metodi density-based, esclusi per design): chiederli a generate()
        # conservava gli stati di tutti i layer per ogni token, ed era la voce
        # di memoria che portava i Gemma a ~24 GB di picco su MMLU, al limite
        # della RTX 3090. Se uno stimatore ne avesse bisogno, UEManager
        # fallirebbe subito all'avvio per statistica mancante, non in silenzio.
        output_hidden_states=False,
        blackbox_supports_logprobs=False,
        deberta_batch_size=deberta_batch_size or DEBERTA_BATCH_SIZE_DEFAULT,
    )


def estimator_calculator_closure(estimator, containers):
    """Tutti i calcolatori di statistiche di cui un metodo ha bisogno, anche
    INDIRETTAMENTE. Prima si guardavano solo le statistiche dichiarate dal
    metodo: Label Prob. chiede le classi semantiche, che pero' si calcolano dai
    K campioni; CCP usa l'NLI sulle alternative dei token; TokenSAR e SAR usano
    un cross-encoder sui campioni. Quei costi non venivano attribuiti, e CCP e
    TokenSAR risultavano economici quanto la Perplexity. La mappa statistica ->
    calcolatore segue la stessa regola di UEManager (vince l'ultimo registrato).
    La generazione greedy e' sempre inclusa: e' la risposta da valutare."""
    provider = {}
    for c in containers:
        for stat in c.stats:
            provider[stat] = c
    closure, todo = {"GreedyProbsCalculator"}, list(getattr(estimator, "stats_dependencies", []))
    visited = set()
    while todo:
        stat = todo.pop()
        if stat in visited:
            continue
        visited.add(stat)
        c = provider.get(stat)
        if c is None:
            continue
        closure.add(c.name)
        todo.extend(c.dependencies)
    return closure


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


# Implementazione dell'attenzione, per modello.
#
# Fino al 2026-09-09 era "eager" per TUTTI, ed era un errore costoso: eager
# calcola l'attenzione materializzando la matrice completa, che cresce col
# QUADRATO della lunghezza del contesto, mentre SDPA usa kernel fusi che non la
# materializzano mai. Serve solo agli stimatori che leggono le mappe di
# attenzione (RAUQ, AttentionScore), che non sono nella nostra lista: infatti
# passiamo output_attentions=False. Era quindi costo puro, in tempo e in
# memoria, senza alcun beneficio -- e il sospetto e' che sia la causa
# principale sia delle run da 30 ore sia degli OOM su CoQA, il dataset con i
# contesti piu' lunghi.
#
# Eccezione per la famiglia Gemma: le sue varianti usano il soft-capping dei
# logit di attenzione, che le implementazioni fuse non applicano correttamente
# in tutte le versioni di transformers, e la guida di HuggingFace raccomanda
# eager per quei modelli. Preferisco pagare il costo dove la correttezza e' in
# dubbio piuttosto che ottenere numeri veloci e sbagliati. DA VERIFICARE sulla
# model card della versione di transformers in uso (4.57.3).
ATTN_IMPLEMENTATION_DEFAULT = "sdpa"
MODEL_ATTN_IMPLEMENTATION = {
    "Gemma3-4B-it": "eager",
    "MedGemma-4B-it": "eager",
}


def attn_implementation_for(model_name):
    return MODEL_ATTN_IMPLEMENTATION.get(model_name, ATTN_IMPLEMENTATION_DEFAULT)


def load_whitebox_model(model_id, cache_dir, hf_token=None,
                        attn_implementation=ATTN_IMPLEMENTATION_DEFAULT, use_quantization=True):
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

    print(f"  caricamento {model_id}: attenzione={attn_implementation}, "
          f"{'4-bit nf4' if use_quantization else 'bf16'}")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    hf_model.eval()
    # Quando una sequenza di un batch si ferma prima delle altre (fine turno o
    # stringa di arresto), generate() riempie le posizioni successive con
    # pad_token_id. lm-polygraph taglia la generazione solo al primo EOS: se il
    # riempitivo fosse un token <pad> diverso da EOS, finirebbe dentro il testo
    # e dentro le log-probabilita' usate dagli stimatori. Con il riempitivo
    # uguale a EOS il taglio avviene esattamente dove la sequenza e' finita.
    # (Il padding a sinistra dei prompt usa tokenizer.pad_token e non cambia.)
    if tokenizer.eos_token_id is not None:
        hf_model.generation_config.pad_token_id = tokenizer.eos_token_id

    model = WhiteboxModel(hf_model, tokenizer, model_path=model_id)
    return model


def build_manager(model, dataset, estimators, cache_dir, max_rejection, max_new_tokens,
                  generation_metric, stat_timing_dict=None,
                  deberta_batch_size=DEBERTA_BATCH_SIZE_DEFAULT):
    available_stat_calculators = default_stat_calculators(cache_dir, deberta_batch_size)
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
        # Con ignore_exceptions=True lm-polygraph, a un errore su un batch
        # (tipicamente OOM su un prompt lungo), RIMUOVE lo stimatore coinvolto
        # per tutto il resto della cella e continua: il metodo sparisce dai
        # risultati senza che il job fallisca. Meglio fallire in modo visibile:
        # l'OOM viene gestito da run_model_on_dataset (riprova con batch 1) e
        # ogni altro errore dall'esecuzione a blocchi (run_cell_chunked).
        ignore_exceptions=False,
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
    confidente del metodo: quanti errori passano inosservati, presentati con
    la massima sicurezza. Implementazione in analysis_lib (condivisa con lo
    script di ricalcolo offline), con gestione corretta dei pareggi."""
    return _silent_failure_rate(ue, quality,
                                correctness_threshold=CORRECTNESS_THRESHOLD,
                                quantile=quantile)


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
    return stats_from_arrays(quality, quality_name, scores, model_name, dataset_name,
                             paper_label_by_str, max_rejection, n_bootstrap, seed)


def stats_from_arrays(quality, quality_name, scores, model_name, dataset_name,
                      paper_label_by_str, max_rejection, n_bootstrap, seed):
    """Stesse statistiche di compute_instance_level_stats, ma a partire dagli
    array per-istanza gia' estratti. Serve all'esecuzione a blocchi, che
    ricalcola tutto sulle istanze di tutti i blocchi riunite."""
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
                         max_new_tokens_override=None, quantized=None, weights_model_name=None,
                         stop_strings_key="stop_strings"):
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

        # `model_name` puo' essere un'etichetta di variante ("full precision
        # (bf16)") invece di un nome in MODELS: in quel caso il conteggio dei
        # parametri va cercato sotto il nome del modello vero, passato dal
        # chiamante in weights_model_name.
        weights_name = weights_model_name or model_name
        is_quantized = uses_quantization(weights_name) if quantized is None else quantized
        batch_size = batch_size_for(weights_name, args.batch_size, is_quantized)
        deberta_batch_size = deberta_batch_size_for(weights_name, is_quantized)
        if batch_size != args.batch_size or deberta_batch_size != DEBERTA_BATCH_SIZE_DEFAULT:
            print(f"  batch ridotto per {model_name} "
                  f"({weights_gb(weights_name, is_quantized):.1f}GB di pesi, "
                  f"{'4-bit' if is_quantized else 'bf16'}): generazione={batch_size} "
                  f"(default {args.batch_size}), NLI={deberta_batch_size} "
                  f"(default {DEBERTA_BATCH_SIZE_DEFAULT}).")
        max_new_tokens = max_new_tokens_override or cfg["max_new_tokens"]

        # Stringhe di arresto della generazione per questo dataset (vedi il
        # commento su STOP_FIRST_NEWLINE in dataset_prep.py). Valgono per la
        # generazione greedy e per i campioni, come nel protocollo ufficiale.
        stop_strings = cfg.get(stop_strings_key)
        model.generation_parameters.stop_strings = list(stop_strings) if stop_strings else None
        print(f"  stringhe di arresto ({stop_strings_key}): {stop_strings!r}")

        # Un OOM non deve far perdere la cella: si riprova una volta con batch
        # di generazione e di NLI a 1, che e' la configurazione di memoria
        # minima. Se fallisce anche cosi', l'errore risale e lo gestisce
        # l'esecuzione a blocchi.
        attempts = [(batch_size, deberta_batch_size)]
        if (batch_size, deberta_batch_size) != (1, 1):
            attempts.append((1, 1))
        for attempt_idx, (gen_bs, nli_bs) in enumerate(attempts):
            model_dataset = PolygraphDataset(prompts, references, batch_size=gen_bs)
            base_estimators = estimators_factory() if estimators_factory else build_estimators()

            # Memoria di picco misurata per singola combinazione: su un telefono il
            # vincolo stringente e' spesso la RAM prima del tempo, quindi entra
            # nella tabella dei costi insieme ai tempi.
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # L'istanza della metrica va tenuta: le metriche MCQ contano quante
            # risposte non sono estraibili, e quel numero distingue "il modello
            # sbaglia" da "il modello non produce il formato".
            generation_metric = cfg["generation_metric_factory"]()

            timing_dict = {}
            stat_timing_dict = {}
            estimators = [TimedEstimator(e, timing_dict) for e in base_estimators]
            containers = default_stat_calculators(args.cache_dir, nli_bs)
            closures = {str(e): estimator_calculator_closure(e, containers) for e in base_estimators}
            man = build_manager(
                model, model_dataset, estimators, args.cache_dir, args.max_rejection,
                max_new_tokens, generation_metric,
                stat_timing_dict=stat_timing_dict,
                deberta_batch_size=nli_bs,
            )
            try:
                man()
                break
            except torch.cuda.OutOfMemoryError:
                if attempt_idx == len(attempts) - 1:
                    raise
                print(f"  OOM su {model_name}/{dataset_name} con batch generazione={gen_bs}, "
                      f"NLI={nli_bs}: riprovo con batch 1/1.")
                del man
                gc.collect()
                torch.cuda.empty_cache()

        # Diagnostica: generazioni vuote. Con le stringhe di arresto un modello
        # che iniziasse la risposta andando a capo produrrebbe una risposta
        # vuota; va visto subito, non scoperto nelle figure.
        try:
            greedy_texts = man.stats.get("greedy_texts", [])
            n_empty = sum(1 for t in greedy_texts if not str(t).strip())
            if greedy_texts and n_empty / len(greedy_texts) > 0.05:
                print(f"  ATTENZIONE {model_name}/{dataset_name}: {n_empty}/{len(greedy_texts)} "
                      f"generazioni vuote.")
        except Exception:
            pass
        log_gpu_mem(f"{model_name}/{dataset_name} done")
        peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else np.nan

        stats_df, per_instance_df = compute_instance_level_stats(
            man, model_name, dataset_name, paper_label_by_str,
            args.max_rejection, args.n_bootstrap, SEED,
        )
        # Parse-failure rate della metrica di qualita', quando la metrica lo
        # espone (MCQAccuracyMetric). Va accanto all'accuracy, non nascosto
        # dentro di essa.
        n_parse_failures = getattr(generation_metric, "n_parse_failures", None)
        n_seen = getattr(generation_metric, "n_seen", 0) or 0
        answer_parse_failure_rate = (n_parse_failures / n_seen) if (n_parse_failures is not None and n_seen) else np.nan
        if stats_df is not None and not stats_df.empty:
            stats_df["answer_parse_failure_rate"] = answer_parse_failure_rate
            acc = stats_df["mean_quality"].iloc[0]
            qname = stats_df["quality_metric"].iloc[0]
            messaggio = (f"{model_name}/{dataset_name}: {qname} medio = {acc:.3f} "
                         f"su {stats_df['n_instances'].iloc[0]} istanze.")
            if np.isfinite(answer_parse_failure_rate):
                messaggio += (f" Risposte non estraibili: "
                              f"{answer_parse_failure_rate:.1%} ({n_parse_failures}/{n_seen}).")
            print(messaggio)

        # Campione di generazioni grezze, accodato a un unico CSV. Non deve mai
        # far fallire una combinazione: e' materiale diagnostico.
        try:
            esempi = getattr(generation_metric, "esempi", None)
            if esempi:
                righe = pd.DataFrame(esempi)
                righe.insert(0, "dataset", dataset_name)
                righe.insert(0, "model", model_name)
                campioni_path = os.path.join(args.results_dir, "sample_generations.csv")
                righe.to_csv(campioni_path, mode="a", index=False,
                             header=not os.path.exists(campioni_path))
        except Exception:
            print("  (salvataggio del campione di generazioni fallito, ignoro)")

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
            closure = closures.get(est_str, {"GreedyProbsCalculator"})
            # Costo pieno standalone: se questa fosse l'unica tecnica in
            # esecuzione, dovrebbe pagarsi da sola ogni calcolatore della sua
            # catena di dipendenze (generazione greedy, campioni, modelli
            # ausiliari, forward extra), oltre alla propria aritmetica. I
            # tempi sono quelli misurati calcolatore per calcolatore.
            full = seconds + sum(stat_timing_dict.get(c, 0.0) for c in closure)
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
                "needs_sampling": "SamplingGenerationCalculator" in closure,
                "needs_nli": bool(closure & _NLI_CALCULATORS),
                "needs_cross_encoder": bool(closure & _CROSS_ENCODER_CALCULATORS),
                "needs_extra_forward": bool(closure & _EXTRA_FORWARD_CALCULATORS),
                "calculators": ";".join(sorted(closure)),
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


# ---------------------------------------------------------------------------
# Esecuzione a blocchi con checkpoint.
#
# Con centinaia di istanze per dataset una singola cella modello x dataset puo'
# durare molte ore (Gemma su GSM8k: ~3 minuti per istanza). Il checkpoint della
# pipeline esiste solo a cella completata, quindi un job interrotto (limite di
# tempo SLURM, nodo perso, OOM) ripartiva da zero su quella cella, e una cella
# piu' lunga del limite di tempo non sarebbe mai finita.
#
# Qui ogni cella viene eseguita a blocchi di --chunk_size istanze, ciascuno
# con la STESSA funzione run_model_on_dataset di sempre, e ogni blocco
# completato viene salvato su disco. Alla fine i blocchi vengono riuniti e PRR,
# intervalli bootstrap e silent failure rate vengono ricalcolati sulle istanze
# di tutti i blocchi insieme: il PRR ricalcolato dagli array per-istanza
# coincide esattamente con quello di UEManager (verificato sulle 520 celle
# della run del 13/09, differenza massima 0.0), quindi il risultato e' identico
# a quello di un'unica esecuzione sull'intera cella.
#
# Se un blocco fallisce anche dopo il tentativo con batch 1 (vedi
# run_model_on_dataset), le sue istanze vengono rieseguite una alla volta e
# quelle che falliscono ancora vengono saltate e registrate. Oltre una soglia
# di istanze saltate la cella viene abbandonata in modo esplicito: un errore
# sistematico (un bug) non deve diventare una cella con meta' dati.
# ---------------------------------------------------------------------------

def _safe_name(x):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x))


def _part_from_result(result, n_instances):
    prr_df, timing_df, stats_df, per_inst_df = result
    if prr_df is None or stats_df is None or per_inst_df is None:
        return None
    rate = (stats_df["answer_parse_failure_rate"].iloc[0]
            if "answer_parse_failure_rate" in stats_df.columns else np.nan)
    meta = pd.DataFrame([{
        "n_instances": n_instances,
        "parse_failure_sum": rate * n_instances if np.isfinite(rate) else np.nan,
        "quality_metric": stats_df["quality_metric"].iloc[0],
    }])
    timing_df = timing_df if timing_df is not None else pd.DataFrame()
    return {"prr": prr_df, "timing": timing_df, "per_instance": per_inst_df, "meta": meta}


def _concat_parts(parts):
    return {
        "prr": parts[0]["prr"],
        "timing": pd.concat([p["timing"] for p in parts], ignore_index=True),
        "per_instance": pd.concat([p["per_instance"] for p in parts], ignore_index=True),
        "meta": pd.concat([p["meta"] for p in parts], ignore_index=True),
    }


def _merge_timing(timing_all, model_name, dataset_name):
    if timing_all is None or timing_all.empty:
        return None
    agg = timing_all.groupby("estimator", as_index=False, sort=False).agg(
        paper_label=("paper_label", "first"),
        seconds=("seconds", "sum"),
        seconds_full_standalone=("seconds_full_standalone", "sum"),
        needs_sampling=("needs_sampling", "first"),
        needs_nli=("needs_nli", "first"),
        peak_memory_gb=("peak_memory_gb", "max"),
        n_instances=("n_instances", "sum"),
        **{c: (c, "first") for c in ("needs_cross_encoder", "needs_extra_forward", "calculators")
           if c in timing_all.columns},
    )
    n = agg["n_instances"].clip(lower=1)
    agg["seconds_marginal_per_instance"] = agg["seconds"] / n
    agg["seconds_full_per_instance"] = agg["seconds_full_standalone"] / n
    agg.insert(0, "dataset", dataset_name)
    agg.insert(0, "model", model_name)
    cols = ["model", "dataset", "estimator", "paper_label", "seconds",
            "seconds_marginal_per_instance", "seconds_full_standalone",
            "seconds_full_per_instance", "needs_sampling", "needs_nli",
            "needs_cross_encoder", "needs_extra_forward", "calculators",
            "peak_memory_gb", "n_instances"]
    return agg[[c for c in cols if c in agg.columns]]


def run_cell_chunked(section_tag, model, model_name, dataset_name, examples, cfg, args,
                     paper_label_by_str, **kwargs):
    """Come run_model_on_dataset (stessi argomenti e stesso valore di ritorno),
    ma a blocchi con checkpoint. Vedi il commento sopra."""
    chunk_size = int(getattr(args, "chunk_size", 0) or 0)
    n = len(examples)
    if chunk_size <= 0 or n <= chunk_size:
        return run_model_on_dataset(model, model_name, dataset_name, examples, cfg, args,
                                    paper_label_by_str, **kwargs)

    chunk_dir = os.path.join(args.results_dir, "chunks", _safe_name(section_tag),
                             f"{_safe_name(model_name)}__{_safe_name(dataset_name)}__n{n}_c{chunk_size}")
    if getattr(args, "no_resume", False) and os.path.isdir(chunk_dir):
        shutil.rmtree(chunk_dir)
    os.makedirs(chunk_dir, exist_ok=True)

    # Il bootstrap dei singoli blocchi non serve (viene rifatto sull'insieme):
    # se ne fa uno minimo per non sprecare tempo.
    chunk_args = copy.copy(args)
    chunk_args.n_bootstrap = min(int(args.n_bootstrap), 20)

    n_chunks = math.ceil(n / chunk_size)
    max_skipped = max(5, int(0.02 * n))
    parts = []
    skipped_total = 0
    print(f"\n=== {model_name}/{dataset_name}: {n} istanze in {n_chunks} blocchi da {chunk_size} ===")
    for ci in range(n_chunks):
        lo, hi = ci * chunk_size, min(n, (ci + 1) * chunk_size)
        paths = {k: os.path.join(chunk_dir, f"chunk{ci:04d}_{k}.csv")
                 for k in ("prr", "timing", "per_instance", "meta")}
        skipped_path = os.path.join(chunk_dir, f"chunk{ci:04d}_skipped.csv")
        if all(os.path.exists(pth) for pth in paths.values()):
            try:
                part = {k: pd.read_csv(pth) for k, pth in paths.items()}
                parts.append(part)
                if os.path.exists(skipped_path):
                    skipped_total += len(pd.read_csv(skipped_path))
                print(f"  blocco {ci + 1}/{n_chunks} ripreso da checkpoint.")
                continue
            except Exception as e:
                print(f"  blocco {ci + 1}/{n_chunks}: checkpoint illeggibile ({e}), lo ricalcolo.")

        print(f"  blocco {ci + 1}/{n_chunks} (istanze {lo}-{hi - 1})")
        part = _part_from_result(
            run_model_on_dataset(model, model_name, dataset_name, examples[lo:hi], cfg,
                                 chunk_args, paper_label_by_str, **kwargs),
            hi - lo)
        skipped_here = []
        if part is None:
            print(f"!!! blocco {ci + 1}/{n_chunks} di {model_name}/{dataset_name} fallito: "
                  f"lo rieseguo un'istanza alla volta.")
            sub_parts = []
            for k in range(lo, hi):
                sub = _part_from_result(
                    run_model_on_dataset(model, model_name, dataset_name, examples[k:k + 1],
                                         cfg, chunk_args, paper_label_by_str, **kwargs),
                    1)
                if sub is None:
                    skipped_here.append(k)
                    print(f"!!! istanza {k} di {model_name}/{dataset_name} fallita anche da sola: saltata.")
                    if skipped_total + len(skipped_here) > max_skipped:
                        print(f"!!! {model_name}/{dataset_name}: piu' di {max_skipped} istanze fallite. "
                              f"Non e' un problema di singole istanze ma sistematico: "
                              f"abbandono la cella (i blocchi completati restano su disco).")
                        return None, None, None, None
                    continue
                sub_parts.append(sub)
            if not sub_parts:
                print(f"!!! blocco {ci + 1}/{n_chunks}: nessuna istanza riuscita, abbandono la cella.")
                return None, None, None, None
            part = _concat_parts(sub_parts)
            pd.DataFrame({"instance_index": skipped_here}).to_csv(skipped_path, index=False)
            skipped_total += len(skipped_here)

        for k, pth in paths.items():
            part[k].to_csv(pth, index=False)
        parts.append(part)

    # --- Riunione dei blocchi ---
    per_inst = pd.concat([p["per_instance"] for p in parts], ignore_index=True)
    common = set.intersection(*[set(p["per_instance"].columns) for p in parts])
    est_cols = [c for c in per_inst.columns
                if c not in ("model", "dataset", "quality") and c in common]
    dropped = [c for c in per_inst.columns if c not in ("model", "dataset", "quality") and c not in common]
    if dropped:
        print(f"ATTENZIONE {model_name}/{dataset_name}: stimatori assenti in qualche blocco, "
              f"esclusi dalla cella: {dropped}")
    per_inst = per_inst[["model", "dataset", "quality"] + est_cols]
    quality = per_inst["quality"].to_numpy(dtype=float)
    scores = {c: per_inst[c].to_numpy(dtype=float) for c in est_cols}
    meta = pd.concat([p["meta"] for p in parts], ignore_index=True)
    quality_name = meta["quality_metric"].iloc[0]

    stats_df, per_instance_df = stats_from_arrays(
        quality, quality_name, scores, model_name, dataset_name, paper_label_by_str,
        args.max_rejection, args.n_bootstrap, SEED)
    if stats_df is None:
        return None, None, None, None
    n_done = meta["n_instances"].sum()
    pf = meta["parse_failure_sum"]
    stats_df["answer_parse_failure_rate"] = (pf.sum() / n_done) if pf.notna().any() else np.nan
    stats_df["n_skipped_instances"] = skipped_total

    prr_df_template = parts[0]["prr"].copy()
    # Stessi valori che UEManager.eval_ue() scriverebbe in man.metrics su
    # tutta la cella: il PRR e la sua versione normalizzata fra punteggio
    # casuale e oracolo ("prr_0.5_normalized").
    ue_metric = PredictionRejectionArea(max_rejection=args.max_rejection)
    q_valid = quality[~np.isnan(quality)]
    if len(q_valid) > 0:
        oracle_score = ue_metric(-q_valid, q_valid)
        random_score = get_random_scores(ue_metric, q_valid)
    else:
        oracle_score = random_score = np.nan
    prr_by_est = {e: _prr_from_arrays(ue_metric, v, quality) for e, v in scores.items()}
    values = []
    for e, um in zip(prr_df_template["estimator"], prr_df_template["ue_metric"]):
        if e not in prr_by_est:
            values.append(np.nan)
        elif um == str(ue_metric):
            values.append(prr_by_est[e])
        elif um == str(ue_metric) + "_normalized":
            values.append(normalize_metric(prr_by_est[e], oracle_score, random_score))
        else:
            values.append(np.nan)
    prr_df = prr_df_template
    prr_df["model"] = model_name
    prr_df["dataset"] = dataset_name
    prr_df["value"] = values
    timing_df = _merge_timing(pd.concat([p["timing"] for p in parts], ignore_index=True),
                              model_name, dataset_name)

    acc = stats_df["mean_quality"].iloc[0]
    print(f"=== {model_name}/{dataset_name} riunita: {len(quality)} istanze, "
          f"{quality_name} medio = {acc:.3f}, istanze saltate = {skipped_total} ===")
    return prr_df, timing_df, stats_df, per_instance_df


def run_dataset_section(section_name, datasets_cfg, args, hf_token, paper_label_by_str,
                        results_basename, models=None, estimators_factory=None,
                        content_transform=None, max_new_tokens_override=None,
                        min_datasets=1, stop_strings_key="stop_strings"):
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
    # --no_resume deve valere QUI come nella pipeline principale. Quando non lo
    # faceva, un run lanciato con --no_resume ricalcolava i dataset principali e
    # saltava in silenzio tutte le sezioni extra, che trovavano i propri
    # checkpoint pieni: il risultato erano figure che mescolavano celle nuove e
    # vecchie senza alcun avviso (osservato nella run 15293305, dove la griglia
    # severita' non ha eseguito un solo modello).
    if getattr(args, "no_resume", False):
        print(f"--no_resume: {results_basename} ricalcolato da zero.")
    elif os.path.exists(final_path):
        try:
            existing = pd.read_csv(final_path)
            already_done = set(zip(existing["dataset"], existing["model"]))
            metrics_dfs.append(existing)
            print(f"Combinazioni gia' completate in {results_basename}:", already_done)
        except Exception as e:
            print(f"{results_basename}.csv illeggibile ({e}) -- riparto senza skip.")
    if not getattr(args, "no_resume", False) and os.path.exists(stats_path):
        try:
            stats_dfs.append(pd.read_csv(stats_path))
        except Exception as e:
            print(f"{results_basename}_instance_stats.csv illeggibile ({e}).")
    # Anche i punteggi per-istanza vanno ricaricati in ripresa: prima venivano
    # scritti solo a fine sezione e solo con le celle di QUESTO job, quindi un
    # run spezzato in piu' job (es. --models diversi) sovrascriveva le celle
    # dei job precedenti e rendeva impossibile ricalcolare offline bootstrap,
    # test appaiati e silent failure rate per quei modelli.
    if not getattr(args, "no_resume", False) and os.path.exists(per_inst_path):
        try:
            per_inst_dfs.append(pd.read_csv(per_inst_path))
        except Exception as e:
            print(f"{results_basename}_per_instance.csv illeggibile ({e}).")

    for model_name, model_id in models.items():
        pending = [d for d in examples_by_ds if (d, model_name) not in already_done]
        if not pending:
            continue
        try:
            model = load_whitebox_model(model_id, args.cache_dir, hf_token=hf_token,
                                        use_quantization=uses_quantization(model_name),
                                        attn_implementation=attn_implementation_for(model_name))
        except Exception:
            print(f"!!! Caricamento di {model_name} fallito, salto i suoi dataset.")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        for dataset_name in pending:
            cfg = datasets_cfg[dataset_name]
            df, _, stats_df, per_inst_df = run_cell_chunked(
                results_basename, model, model_name, dataset_name, examples_by_ds[dataset_name],
                cfg, args, paper_label_by_str,
                use_chat_template=(model_name in CHAT_TEMPLATE_MODELS),
                estimators_factory=estimators_factory,
                content_transform=content_transform,
                max_new_tokens_override=max_new_tokens_override,
                stop_strings_key=stop_strings_key,
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
                pd.concat(per_inst_dfs, ignore_index=True).to_csv(per_inst_path, index=False)

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


# ---------------------------------------------------------------------------
# Ripresa da checkpoint.
#
# main.py salta le combinazioni modello x dataset gia' presenti in
# results_final.csv, cosi' che un run interrotto (job SLURM scaduto, OOM, nodo
# riavviato) possa riprendere senza rifare ore di GPU. Questo pero' e' sicuro
# solo se i checkpoint sono stati prodotti dalla STESSA versione del codice.
#
# Nella run del 2026-09-01 non lo erano: la results_dir conteneva ancora i file
# di agosto, precedenti sia all'aggiunta di Mistral-7B-it sia alle colonne di
# costo per-istanza. Conseguenza: i quattro modelli piccoli sono stati saltati
# in blocco (risultavano "gia' fatti"), e' stato eseguito il solo Mistral --
# che e' andato OOM su tutti e 4 i dataset -- e le figure finali sono state
# rigenerate da dati di agosto con un timestamp fresco, mentre la tabella dei
# costi moriva con KeyError sulle colonne che il vecchio CSV non aveva.
# Silenziosamente: nessuna di queste tre cose ferma il job, che e' uscito con
# COMPLETED / ExitCode 0:0 dopo 14 ore.
#
# Le due difese qui sotto: verificare lo schema prima di fidarsi di un
# checkpoint, e dire a voce alta cosa viene saltato e da dove viene.
# ---------------------------------------------------------------------------
RESULTS_REQUIRED_COLUMNS = {"model", "dataset", "estimator", "ue_metric", "value"}
TIMINGS_REQUIRED_COLUMNS = {
    "model", "dataset", "estimator", "paper_label", "seconds",
    "seconds_marginal_per_instance", "seconds_full_per_instance",
    "needs_sampling", "needs_nli", "peak_memory_gb",
}
# Le statistiche per-istanza vanno ricaricate insieme ai risultati: senza,
# una ripresa produce figure PRR complete ma accuracy table, intervalli di
# confidenza e Kendall tau calcolati sui soli modelli rieseguiti.
STATS_REQUIRED_COLUMNS = {
    "model", "dataset", "estimator", "paper_label", "prr",
    "prr_ci_low", "prr_ci_high", "quality_metric", "mean_quality",
    "n_instances", "nan_rate", "silent_failure_rate",
}
PER_INSTANCE_REQUIRED_COLUMNS = {"model", "dataset", "quality"}


def load_checkpoint_csv(path, required_columns, label):
    """Ricarica un checkpoint solo se ha lo schema che il resto della pipeline
    si aspetta. Uno schema incompleto significa "prodotto da una versione
    precedente del codice": in quel caso il file viene ignorato (meglio
    rieseguire che mescolare due versioni in una figura sola)."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"!!! {label} presente ma illeggibile ({e}) -- ignorato, si riparte da zero.")
        return None
    missing = required_columns - set(df.columns)
    if missing:
        print(f"!!! {label} e' stato prodotto da una versione PRECEDENTE del codice "
              f"(colonne mancanti: {sorted(missing)}). Lo ignoro e riparto da zero su "
              f"questa parte: riprendere da li' mescolerebbe risultati di due versioni "
              f"diverse nelle stesse figure. Sposta o cancella la vecchia results_dir "
              f"per non rivedere questo avviso.")
        return None
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Selective QA UQ benchmark (lm-polygraph) -- replica Sezione 5.1 Vashurin et al."
    )
    parser.add_argument("--n_test_samples", type=int, default=None,
                         help="Se specificato, sovrascrive n_test per TUTTI i dataset (utile per smoke test rapidi).")
    # Default 1: e' la configurazione del protocollo ufficiale di lm-polygraph
    # su CoQA/GSM8k, usa la memoria minima e non introduce padding nei batch.
    # Il tempo in piu' riguarda solo i modelli piccoli, che sono i piu' veloci.
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=100,
                         help="Istanze per blocco con checkpoint dentro ogni cella modello x dataset "
                              "(vedi run_cell_chunked). 0 = nessun blocco.")
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
    parser.add_argument("--datasets", nargs="+", default=None, choices=list(DATASETS.keys()),
                         help="Esegue solo questi dataset della pipeline principale (e delle sezioni "
                              "verbalized e quantizzazione, che usano gli stessi). Serve a dividere il "
                              "lavoro in piu' run nella STESSA --results_dir: le celle gia' calcolate "
                              "restano nei checkpoint e le figure finali le includono tutte. Es.: prima "
                              "--datasets CoQA TriviaQA MMLU, poi --datasets GSM8k. La griglia di "
                              "severita' non e' influenzata. Default: tutti.")
    parser.add_argument("--models", nargs="+", default=None, choices=list(MODELS.keys()),
                         help="Esegue solo i modelli indicati invece di tutti. Serve per i rerun "
                              "mirati (un modello fallito, un test su piu' istanze) e per isolare i "
                              "problemi di memoria: caricato da solo, un modello trova la GPU pulita "
                              "invece che frammentata dai modelli eseguiti prima di lui. "
                              "Default: tutti i modelli di MODELS.")
    parser.add_argument("--no_resume", action="store_true",
                         help="Ignora i checkpoint presenti in --results_dir e riesegue TUTTE le "
                              "combinazioni modello x dataset. Da usare ogni volta che il codice e' "
                              "cambiato dall'ultimo run: senza questo, i modelli gia' presenti in "
                              "results_final.csv vengono saltati e le figure finiscono per mescolare "
                              "risultati vecchi e nuovi (vedi il commento sopra load_checkpoint_csv).")
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

    # Controlli che devono fallire SUBITO se qualcosa non va, invece di
    # produrre celle vuote dopo ore di GPU.
    n_casi = self_test_mcq_metric()
    print(f"Self-test estrazione risposte MCQ: {n_casi} casi, tutti corretti.")

    # Sottoinsieme di modelli su cui lavorare. Vale per la pipeline principale
    # e per TUTTE le sezioni extra, cosi' che un rerun mirato non riesegua di
    # nascosto gli altri modelli in una delle sezioni.
    selected_models = MODELS if not args.models else {
        name: MODELS[name] for name in MODELS if name in set(args.models)
    }
    if args.models:
        print(f"--models: eseguo solo {list(selected_models)} "
              f"(gli altri restano ai valori gia' presenti nei checkpoint).")

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
        if args.datasets and ds_name not in args.datasets:
            print(f"--datasets: salto {ds_name} (resta ai valori gia' presenti nei checkpoint).")
            continue
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

    timing_final_path = os.path.join(args.results_dir, "estimator_timings.csv")
    all_timing_dfs = []

    if args.no_resume:
        print("--no_resume: ignoro qualunque checkpoint, eseguo tutte le combinazioni da zero.")
    else:
        results_df_existing = load_checkpoint_csv(
            final_path, RESULTS_REQUIRED_COLUMNS, "results_final.csv")
        if results_df_existing is not None:
            already_done_pairs = set(zip(results_df_existing["dataset"], results_df_existing["model"]))
            skipped_models = sorted({m for _, m in already_done_pairs})
            print(f"ATTENZIONE: riprendo da results_final.csv. I risultati di questi modelli "
                  f"NON vengono ricalcolati e provengono da un run precedente: {skipped_models}. "
                  f"Se il codice e' cambiato da allora, rilancia con --no_resume.")
            print("Combinazioni dataset x modello gia' completate:", already_done_pairs)
        else:
            print("Nessun risultato precedente utilizzabile -- eseguo tutte le combinazioni.")

        timing_existing = load_checkpoint_csv(
            timing_final_path, TIMINGS_REQUIRED_COLUMNS, "estimator_timings.csv")
        if timing_existing is not None:
            all_timing_dfs = [timing_existing]

    all_metrics_dfs = [results_df_existing] if results_df_existing is not None else []

    # Statistiche a livello di istanza (accuracy, intervalli bootstrap, silent
    # failure rate) e valori grezzi per-istanza. Questi ultimi vengono salvati
    # su disco per poter rifare bootstrap, test appaiati e analisi di soglia
    # senza rieseguire nulla sulla GPU.
    all_stats_dfs = []
    all_per_instance_dfs = []

    # Ricaricate insieme ai risultati (vedi STATS_REQUIRED_COLUMNS): sono i
    # dati da cui derivano accuracy table, barre d'errore e Kendall tau.
    if not args.no_resume and results_df_existing is not None:
        stats_existing = load_checkpoint_csv(
            os.path.join(args.results_dir, "instance_level_stats.csv"),
            STATS_REQUIRED_COLUMNS, "instance_level_stats.csv")
        if stats_existing is not None:
            all_stats_dfs.append(stats_existing)
        else:
            print("ATTENZIONE: risultati ripresi da checkpoint ma le statistiche per-istanza "
                  "non sono ricaricabili. Accuracy table, intervalli di confidenza e Kendall tau "
                  "copriranno solo i modelli rieseguiti in questo run.")
        per_inst_existing = load_checkpoint_csv(
            os.path.join(args.results_dir, "per_instance_scores.csv"),
            PER_INSTANCE_REQUIRED_COLUMNS, "per_instance_scores.csv")
        if per_inst_existing is not None:
            all_per_instance_dfs.append(per_inst_existing)

    # Loop esterno sui MODELLI (non sui dataset): caricare un modello da 4B
    # e' l'operazione piu' costosa, quindi lo facciamo una volta sola e gli
    # facciamo girare tutti e 4 i dataset prima di scaricarlo, invece di
    # ricaricarlo per ogni dataset.
    for model_name, model_id in selected_models.items():
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
                                         use_quantization=uses_quantization(model_name),
                                         attn_implementation=attn_implementation_for(model_name))
            log_gpu_mem(f"{model_name} loaded")
        except Exception:
            print(f"!!! Caricamento di {model_name} fallito, salto tutti i suoi dataset.")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        for dataset_name in pending_datasets:
            cfg = DATASETS[dataset_name]
            df, timing_df, stats_df, per_inst_df = run_cell_chunked(
                "main", model, model_name, dataset_name, dataset_examples[dataset_name], cfg, args,
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
            # Checkpoint anche dei punteggi per-istanza: prima erano scritti
            # solo a fine run, quindi un job interrotto (OOM, limite di tempo
            # SLURM) perdeva quelli di tutte le celle gia' completate.
            if all_per_instance_dfs:
                pd.concat(all_per_instance_dfs, ignore_index=True).to_csv(
                    os.path.join(args.results_dir, "per_instance_scores.csv"), index=False
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

    # keep="last": se una combinazione e' stata rieseguita in questo run, il
    # suo valore nuovo prevale su quello ricaricato dal checkpoint.
    stats_all = (
        pd.concat(all_stats_dfs, ignore_index=True).drop_duplicates(
            subset=["model", "dataset", "estimator"], keep="last")
        if all_stats_dfs else None
    )
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

    # Le statistiche per-istanza (accuracy, intervalli bootstrap, silent
    # failure rate) esistono solo per le combinazioni eseguite IN QUESTO run:
    # non vengono ricaricate dai checkpoint. Quindi, in una ripresa, accuracy
    # table e Kendall tau coprono meno modelli delle figure PRR, e le barre
    # d'errore dei modelli saltati risultano assenti. Meglio dirlo che lasciarlo
    # scoprire da una figura con meta' delle barre senza intervallo.
    if already_done_pairs:
        covered = sorted(stats_all["model"].unique()) if stats_all is not None else []
        print(f"ATTENZIONE: statistiche per-istanza disponibili solo per {covered} "
              f"(i modelli ripresi da checkpoint non le hanno). Accuracy table, intervalli "
              f"di confidenza e Kendall tau saranno limitati a questi modelli.")

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

    # Le figure (Figure A/B, griglia, costi, Pareto, ...) le disegna
    # make_figures.py, richiamato alla fine di questo script: qui si producono
    # solo i dati.
    figure_a_labels = [m["paper_label"] for m in PAPER_METHODS if m["figure"] in ("A", "AB") and m["factory"] is not None]

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
                **{c: (c, "max") for c in ("needs_cross_encoder", "needs_extra_forward")
                   if c in timing_all.columns},
                # Memoria di picco dell'intera cella modello x dataset, NON del
                # singolo metodo: tutti i metodi della stessa cella hanno lo
                # stesso valore. Utile solo per confrontare i modelli.
                peak_memory_gb_cell=("peak_memory_gb", "max"),
            ).sort_values("sec_full_per_instance", ascending=False)
            cost_path = os.path.join(args.results_dir, "estimator_cost_table.csv")
            cost_table.to_csv(cost_path, index=False)
            print(f"Salvato: {cost_path}")

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
                paper_label_by_str, "results_severity_grid", models=selected_models,
                min_datasets=2,
            )
            if severity_raw is not None:
                build_accuracy_table(
                    severity_stats, args.results_dir, "accuracy_table_severity_grid.csv"
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
    verbalized_datasets = {k: v for k, v in VERBALIZED_DATASETS.items()
                           if not args.datasets or k in args.datasets}
    if args.run_verbalized and not verbalized_datasets:
        print("--run_verbalized: nessun dataset selezionato da --datasets e' previsto per i "
              "metodi verbalized, salto la sezione.")
    if args.run_verbalized and verbalized_datasets:
        for style in ("numeric", "linguistic"):
            try:
                verb_raw, verb_stats = run_dataset_section(
                    f"Metodi verbalized ({style})", verbalized_datasets, args, hf_token,
                    paper_label_by_str, f"results_verbalized_{style}",
                    models=selected_models,
                    estimators_factory=lambda s=style: build_verbalized_estimators(s),
                    content_transform=lambda c, s=style: build_verbalized_content(c, s),
                    max_new_tokens_override=args.verbalized_max_new_tokens,
                    # La confidenza va scritta su una riga dopo la risposta:
                    # fermarsi al primo a capo la taglierebbe sempre. Qui ci si
                    # ferma solo quando il modello inizia un esempio inventato.
                    stop_strings_key="continuation_stop_strings",
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
                quant_stats_path = os.path.join(args.results_dir, "results_quant_comparison_instance_stats.csv")
                quant_per_inst_path = os.path.join(args.results_dir, "results_quant_comparison_per_instance.csv")
                quant_metrics_dfs = []
                quant_stats_dfs = []
                quant_per_inst_dfs = []
                quant_already_done = set()
                # Anche qui --no_resume deve valere: vedi il commento in
                # run_dataset_section.
                if args.no_resume:
                    print("--no_resume: confronto quantizzazione ricalcolato da zero.")
                elif os.path.exists(quant_final_path):
                    try:
                        existing = pd.read_csv(quant_final_path)
                        quant_already_done = set(zip(existing["dataset"], existing["model"]))
                        quant_metrics_dfs.append(existing)
                        print("Combinazioni quant gia' completate:", quant_already_done)
                    except Exception as e:
                        print(f"results_quant_comparison.csv illeggibile ({e}) -- riparto senza skip.")
                    # Come in run_dataset_section: in ripresa vanno ricaricate
                    # anche statistiche e punteggi per-istanza, altrimenti le
                    # celle dei job precedenti spariscono dai file finali.
                    for _path, _dfs in ((quant_stats_path, quant_stats_dfs),
                                        (quant_per_inst_path, quant_per_inst_dfs)):
                        if os.path.exists(_path):
                            try:
                                _dfs.append(pd.read_csv(_path))
                            except Exception as e:
                                print(f"{os.path.basename(_path)} illeggibile ({e}).")

                variants = [("quantized (4-bit nf4)", True), ("full precision (bf16)", False)]
                for variant_label, use_quant in variants:
                    pending = [d for d in dataset_examples if (d, variant_label) not in quant_already_done]
                    if not pending:
                        continue
                    try:
                        model = load_whitebox_model(model_id, args.cache_dir, hf_token=hf_token,
                                                    use_quantization=use_quant,
                                                    attn_implementation=attn_implementation_for(quant_model_name))
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
                        df, _, quant_stats_df, quant_per_inst_df = run_cell_chunked(
                            "quant", model, variant_label, dataset_name, dataset_examples[dataset_name],
                            cfg, args, paper_label_by_str,
                            use_chat_template=(quant_model_name in CHAT_TEMPLATE_MODELS),
                            # La variante bf16 ha pesi 4 volte piu' grandi di
                            # quella 4-bit a parita' di modello: il batch va
                            # deciso sul modello vero e sulla precisione vera,
                            # non sull'etichetta della serie.
                            quantized=use_quant, weights_model_name=quant_model_name,
                        )
                        if df is not None:
                            quant_metrics_dfs.append(df)
                            pd.concat(quant_metrics_dfs, ignore_index=True).to_csv(quant_final_path, index=False)
                            print(f"Checkpoint quant salvato dopo {variant_label}/{dataset_name}.")
                        if quant_stats_df is not None:
                            quant_stats_dfs.append(quant_stats_df)
                            pd.concat(quant_stats_dfs, ignore_index=True).to_csv(quant_stats_path, index=False)
                        if quant_per_inst_df is not None:
                            quant_per_inst_dfs.append(quant_per_inst_df)
                            pd.concat(quant_per_inst_dfs, ignore_index=True).to_csv(quant_per_inst_path, index=False)

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
                        quant_stats_all.to_csv(quant_stats_path, index=False)
                        quant_raw = quant_raw.merge(
                            quant_stats_all[["model", "dataset", "estimator",
                                             "prr_ci_low", "prr_ci_high", "mean_quality"]],
                            on=["model", "dataset", "estimator"], how="left",
                        )
                        build_accuracy_table(quant_stats_all, args.results_dir,
                                             "accuracy_table_quant_comparison.csv")
                    quant_raw.to_csv(os.path.join(args.results_dir, "results_quant_comparison_mapped.csv"), index=False)

                    # La figura del confronto la disegna make_figures.py
                    # (fig_quant_comparison.png), con gli intervalli di
                    # confidenza: qui ne veniva prodotta una seconda versione
                    # senza intervalli, duplicata e fuorviante.
                    print("Confronto quantizzazione completato.")
                else:
                    print("!!! Nessuna combinazione quant completata con successo.")
            except Exception:
                print("!!! Confronto quantizzazione fallito:")
                traceback.print_exc()

    # Analisi offline e figure, sui CSV appena scritti. Girano come processi
    # separati (sono script autonomi, rilanciabili a mano con gli stessi
    # comandi): un loro errore non tocca i risultati gia' salvati.
    here = os.path.dirname(os.path.abspath(__file__))
    for script, extra in (("paired_comparisons.py", []),
                          ("paired_comparisons.py",
                           ["--per_instance_file", "results_severity_grid_per_instance.csv"]),
                          ("make_figures.py", [])):
        if "severity" in " ".join(extra) and not os.path.exists(
                os.path.join(args.results_dir, "results_severity_grid_per_instance.csv")):
            continue
        cmd = [sys.executable, os.path.join(here, script), args.results_dir] + extra
        print(f"\n--- {' '.join(cmd[1:])} ---")
        try:
            esito = subprocess.run(cmd, check=False)
            if esito.returncode != 0:
                print(f"!!! {script} terminato con codice {esito.returncode}: i CSV sono salvi, "
                      f"si puo' rilanciare a mano.")
        except Exception:
            print(f"!!! {script} non eseguito:")
            traceback.print_exc()

    print("\nBENCHMARK COMPLETATO.")


if __name__ == "__main__":
    main()
