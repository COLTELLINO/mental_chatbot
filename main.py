import argparse
import copy
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.dataset import Dataset as PolygraphDataset
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.processor import Logger
from lm_polygraph.utils.builder_enviroment_stat_calculator import BuilderEnvironmentStatCalculator
from lm_polygraph.defaults.register_default_stat_calculators import register_default_stat_calculators
from lm_polygraph.ue_metrics import PredictionRejectionArea
from lm_polygraph.estimators import *

import analysis_lib as AL
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
# di turno tipo <start_of_turn>/<end_of_turn>, [INST] o <|im_start|>). I loro
# prompt vengono costruiti con tokenizer.apply_chat_template().
#
# Fino al 29/09 LFM2 non era in questo insieme: era trattato come un modello
# "base" e riceveva un prompt a completamento di testo. Ma LFM2-350M e
# LFM2-1.2B sono modelli chat (SFT + allineamento sulle preferenze, template
# ChatML con <|im_start|>, dalla model card di LiquidAI). Tra modelli piccoli e
# grandi cambiavano quindi due cose insieme, la dimensione e il formato del
# prompt, e non si poteva dire quale delle due spiegasse le differenze; un
# modello chat interrogato senza il suo formato tende anche a "continuare da
# solo" inventando nuove domande, che e' proprio cio' che si vedeva nei log.
# Ora TUTTI i modelli ricevono il proprio chat template. Se un tokenizer non ne
# ha uno, run_model_on_dataset lo segnala e usa il prompt semplice.
CHAT_TEMPLATE_MODELS = {"LFM2-350M", "LFM2-1.2B", "MedGemma-4B-it", "Gemma3-4B-it", "Mistral-7B-it"}

# Modelli che NON possono essere quantizzati a 4-bit: la famiglia Gemma3
# produce logit NaN sotto bitsandbytes nf4 (verificato escludendo prima
# backend di attenzione, formato del prompt e quantizzazione del solo lm_head),
# quindi vanno caricati in bf16.
CANNOT_QUANTIZE_MODELS = {"MedGemma-4B-it", "Gemma3-4B-it"}

# Precisione dell'ancora Mistral-7B (--anchor_precision, default bf16). Fino al
# 29/09 era sempre 4-bit, con la motivazione che in bf16 (~15 GB di pesi) la
# memoria non sarebbe bastata. Ma la memoria non la mangiavano i pesi: la
# mangiava il campionamento dei K campioni, che in lm-polygraph conserva per
# ogni campione gli stati di tutti gli strati e il log-softmax sull'intero
# vocabolario a ogni token. Con il campionatore in batch (batched_sampling.py)
# quella memoria non c'e' piu', e l'ancora puo' girare nella precisione del
# paper. Il valore effettivo viene impostato in main() e scritto nella scheda
# delle condizioni di esecuzione (run_conditions.csv).
ANCHOR_PRECISION = "bf16"

# Insieme dei modelli caricati in bf16 in questa esecuzione.
NO_QUANT_MODELS = set(CANNOT_QUANTIZE_MODELS)


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

# Batch del modello NLI (DeBERTa-large) e del cross-encoder. Per ogni domanda
# le coppie di campioni da confrontare sono K*K = 100: con batch 2 erano 50
# chiamate per domanda, ed era il motivo dei tempi NLI fino a 48 secondi per
# domanda sui modelli grandi. Il batch 2 per i modelli "pesanti" era una difesa
# contro gli OOM, che pero' venivano dal campionamento (vedi
# batched_sampling.py), non dall'NLI: DeBERTa-large su coppie di frasi brevi
# occupa poche centinaia di MB anche a batch 50. Se un OOM capita comunque,
# run_model_on_dataset riprova con batch piu' piccoli.
DEBERTA_BATCH_SIZE_DEFAULT = 50
DEBERTA_BATCH_SIZE_HEAVY = 20

# Override espliciti, per i casi che la regola sopra non prende. Ha la
# precedenza su tutto.
MODEL_BATCH_SIZE = {}


def uses_quantization(model_name):
    """4-bit nf4 per default; bf16 per i modelli che sotto quantizzazione
    producono NaN (CANNOT_QUANTIZE_MODELS) e per l'ancora se
    --anchor_precision bf16 (vedi NO_QUANT_MODELS)."""
    return model_name not in NO_QUANT_MODELS


def set_anchor_precision(precision):
    """Aggiorna NO_QUANT_MODELS secondo --anchor_precision."""
    global ANCHOR_PRECISION
    ANCHOR_PRECISION = precision
    NO_QUANT_MODELS.clear()
    NO_QUANT_MODELS.update(CANNOT_QUANTIZE_MODELS)
    if precision == "bf16":
        NO_QUANT_MODELS.add(REPLICATION_ANCHOR_MODEL)


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
    tokenizer_adds_bos,
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
    "PRR del paper (0 = casuale, 1 = oracolo). Da leggere insieme a accuracy_table.csv: con "
    "un'accuracy vicina al caso il modello tira a indovinare, nessun segnale interno puo' "
    "predire l'esito e il PRR e' vicino a 0 per tutti i metodi."
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

# Fase di costo di ogni calcolatore di statistiche, per NOME DI CLASSE ESATTO.
#
# Fino al 29/09 la fase veniva indovinata da sottostringhe del nome ("semantic"
# -> NLI, "greedy" -> generazione greedy, ...), e il nome inganna: il
# confronto NLI sulle alternative dei token usato da CCP
# (GreedyAlternativesNLICalculator) finiva in "generazione greedy", i due
# cross-encoder di SAR/SentenceSAR/TokenSAR e la chiamata extra al modello di
# P(True) finivano in "altro". Qui ogni calcolatore ha la sua fase dichiarata;
# un calcolatore che non compare nella tabella viene segnalato a stampa
# invece di finire in silenzio in una voce generica.
PHASE_OF_CALCULATOR = {
    "GreedyProbsCalculator": "generazione_greedy",
    "BatchedSamplingCalculator": "campionamento_K",
    "SamplingGenerationCalculator": "campionamento_K",
    "SemanticMatrixCalculator": "nli_campioni",
    "SemanticClassesCalculator": "nli_campioni",          # solo raggruppamento, sulla matrice NLI
    "GreedySemanticMatrixCalculator": "nli_campioni",
    "ConcatGreedySemanticMatrixCalculator": "nli_campioni",
    "GreedyAlternativesNLICalculator": "nli_alternative",   # CCP
    # SAR, SentenceSAR e anche TokenSAR: in lm-polygraph 0.7.0 la statistica
    # "token_similarity" di TokenSAR e' prodotta da questo stesso calcolatore,
    # che dipende dai K campioni. Il costo pieno di TokenSAR include quindi il
    # campionamento: e' come la libreria lo calcola, anche se un'implementazione
    # minima di TokenSAR potrebbe farne a meno (va detto in tesi).
    "CrossEncoderSimilarityMatrixCalculator": "cross_encoder_campioni",
    "GreedyCrossEncoderSimilarityMatrixCalculator": "cross_encoder_greedy",  # non usato dai 26 metodi
    "GreedyLMProbsCalculator": "forward_senza_contesto",    # PMI, CPMI
    "PromptCalculator": "forward_ptrue",                    # P(True)
    "EntropyCalculator": "aritmetica",
    "RawInputCalculator": "aritmetica",
    "InitialStateCalculator": "aritmetica",
}

# Calcolo della metrica di qualita' (AlignScore, un modello da 355M
# parametri, o l'estrazione della lettera): non e' un costo dei metodi UQ ma
# pesa sul tempo totale, quindi viene cronometrato come fase a parte.
QUALITY_PHASE = "metrica_qualita"

# Stimatori che fanno lavoro costoso DENTRO la propria chiamata, senza
# dichiararlo come statistica: BB P(True) (PTrueEmpirical) genera da se' le
# proprie risposte. Il suo tempo di stima e' quindi generazione, non
# aritmetica.
ESTIMATOR_OWN_PHASE = {"PTrueEmpirical": "generazione_interna_stimatore"}


def classify_stat_calculator(name):
    """Fase di costo di un calcolatore (vedi PHASE_OF_CALCULATOR)."""
    phase = PHASE_OF_CALCULATOR.get(name)
    if phase is None:
        print(f"  ATTENZIONE: calcolatore {name} senza fase dichiarata in PHASE_OF_CALCULATOR, "
              f"conteggiato come 'non_classificato'.")
        return "non_classificato"
    return phase


# Calcolatori che caricano un modello ausiliario o fanno un forward in piu',
# per le colonne della tabella dei costi.
_SAMPLING_CALCULATORS = {c for c, f in PHASE_OF_CALCULATOR.items() if f == "campionamento_K"}
_NLI_CALCULATORS = {c for c, f in PHASE_OF_CALCULATOR.items() if f.startswith("nli_")}
_CROSS_ENCODER_CALCULATORS = {c for c, f in PHASE_OF_CALCULATOR.items() if f.startswith("cross_encoder")}
_EXTRA_FORWARD_CALCULATORS = {c for c, f in PHASE_OF_CALCULATOR.items() if f.startswith("forward_")}

# Campionatore dei K campioni: "batched" (batched_sampling.py, una chiamata a
# generate per tutti i campioni, senza stati nascosti) oppure "library"
# (SamplingGenerationCalculator di lm-polygraph, per confronto). Impostato da
# --sampler in main().
SAMPLER = "batched"


def default_stat_calculators(cache_dir, deberta_batch_size=None):
    """Descrizioni dei calcolatori di statistiche usate da UEManager (vedi
    build_manager). Servono anche a ricostruire le dipendenze di ogni metodo."""
    containers = register_default_stat_calculators(
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
    if SAMPLER == "batched":
        from omegaconf import OmegaConf
        from lm_polygraph.utils.factory_stat_calculator import StatCalculatorContainer
        from batched_sampling import BatchedSamplingCalculator
        replaced = []
        for c in containers:
            if c.name == "SamplingGenerationCalculator":
                c = StatCalculatorContainer(
                    name="BatchedSamplingCalculator",
                    obj=BatchedSamplingCalculator,
                    builder="batched_sampling",
                    cfg=OmegaConf.create({"obj": "BatchedSamplingCalculator", "samples_n": 10}),
                    dependencies=BatchedSamplingCalculator.meta_info()[1],
                    stats=BatchedSamplingCalculator.meta_info()[0],
                )
            replaced.append(c)
        containers = replaced
    return containers


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
# Eccezione storica per la famiglia Gemma: Gemma 2 usa il soft-capping dei
# logit di attenzione (config.attn_logit_softcapping), che i kernel fusi di
# SDPA non applicano, e per quel modello HuggingFace raccomanda eager. Fino al
# 29/09 eager era imposto a tutti i Gemma "per prudenza" (commento: "DA
# VERIFICARE"). Gemma 3 pero' ha tolto il soft-capping dell'attenzione (lo ha
# sostituito con la normalizzazione QK), quindi per Gemma 3 e MedGemma eager
# era costo puro, quadratico nella lunghezza del contesto (GSM8k con 8 esempi
# few-shot: il blocco da 3 ore della run di settembre). Ora la scelta e'
# automatica: si legge la configurazione del modello e si usa eager SOLO se
# attn_logit_softcapping e' impostato. Le forzature esplicite restano
# possibili in MODEL_ATTN_IMPLEMENTATION.
ATTN_IMPLEMENTATION_DEFAULT = "sdpa"
MODEL_ATTN_IMPLEMENTATION = {}


def attn_logit_softcapping(model_id, cache_dir=None, hf_token=None):
    """Valore di attn_logit_softcapping nella configurazione del modello (anche
    dentro text_config per i modelli multimodali come Gemma 3), o None."""
    try:
        cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, token=hf_token)
    except Exception as e:
        print(f"  (configurazione di {model_id} non leggibile: {e}; assumo nessun soft-capping)")
        return None
    for c in (cfg, getattr(cfg, "text_config", None)):
        v = getattr(c, "attn_logit_softcapping", None) if c is not None else None
        if v is not None:
            return v
    return None


def attn_implementation_for(model_name, model_id=None, cache_dir=None, hf_token=None):
    if model_name in MODEL_ATTN_IMPLEMENTATION:
        return MODEL_ATTN_IMPLEMENTATION[model_name]
    if model_id is not None and attn_logit_softcapping(model_id, cache_dir, hf_token) is not None:
        print(f"  {model_name}: attn_logit_softcapping impostato -> attenzione eager.")
        return "eager"
    return ATTN_IMPLEMENTATION_DEFAULT


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


class TimedGenerationMetric:
    """Involucro trasparente attorno alla metrica di qualita' che ne cronometra
    il calcolo (fase QUALITY_PHASE). Prima non era misurato e finiva dentro la
    voce "altro" (~20% del tempo totale). Gli attributi non ridefiniti qui
    (n_parse_failures, esempi, ...) vengono letti dalla metrica originale."""

    def __init__(self, metric, timing_dict):
        self._metric = metric
        self._timing = timing_dict
        self.level = metric.level
        self.stats_dependencies = metric.stats_dependencies

    def __str__(self):
        return str(self._metric)

    def __getattr__(self, name):
        return getattr(self._metric, name)

    def __call__(self, stats, target_texts):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = self._metric(stats, target_texts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._timing[QUALITY_PHASE] = self._timing.get(QUALITY_PHASE, 0.0) + (time.perf_counter() - start)
        return out


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
        generation_metrics=[TimedGenerationMetric(generation_metric, stat_timing_dict)
                            if stat_timing_dict is not None else generation_metric],
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


def prr_metric_name(max_rejection):
    """Nome della colonna ue_metric con il PRR del paper (normalizzato fra
    caso e oracolo), come la scrive lm-polygraph: "prr_0.5_normalized"."""
    return f"{PredictionRejectionArea(max_rejection=max_rejection)}_normalized"


def _prr_from_arrays(ue, quality, max_rejection):
    """PRR del paper, (AUC_unc - AUC_rnd) / (AUC_oracle - AUC_rnd), dagli array
    per-istanza, con i pareggi risolti in valore atteso: vedi
    analysis_lib.prr_normalized e il commento che la precede.

    Fino al 29/09 le figure usavano solo AUC_unc (la colonna "prr_0.5"): un
    numero che parte dall'accuracy del modello e sale di poco se il metodo e'
    utile. Resta calcolato come "prr_raw" per confronto.

    ATTENZIONE, un dettaglio con conseguenze reali sui metodi verbalized: un
    punteggio NaN significa "confidenza non estraibile dal testo", ma
    lm-polygraph lo sostituisce con -1e7, il valore piu' BASSO possibile di
    incertezza, quindi quelle istanze vengono trattate come le piu' confidenti
    in assoluto e non vengono mai scartate. Un modello che non produce il
    formato richiesto viene cosi' premiato invece che penalizzato: e' il motivo
    per cui il parse-failure rate va sempre riportato accanto al PRR dei metodi
    verbalized. La convenzione e' mantenuta per restare confrontabili con la
    libreria e con il paper."""
    return AL.prr_normalized(ue, quality, max_rejection)


def silent_failure_rate(ue, quality, quantile=0.10):
    """Vecchia metrica (vedi analysis_lib.silent_failure_rate), conservata
    solo per confronto con le run precedenti."""
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

    mean_quality = float(np.nanmean(quality))
    n_instances = int(len(quality))
    err_overall = AL.error_rate_overall(quality, CORRECTNESS_THRESHOLD)

    rows = []
    per_instance = {"quality": quality}
    for est_name, ue in scores.items():
        if len(ue) != len(quality):
            print(f"ATTENZIONE: {est_name} ha {len(ue)} valori ma la qualita' ne ha "
                  f"{len(quality)}; salto le statistiche per-istanza di questo stimatore.")
            continue
        per_instance[est_name] = ue
        ci_low, ci_high = AL.bootstrap_prr_ci(ue, quality, max_rejection, n_bootstrap, seed)
        rows.append({
            "model": model_name,
            "dataset": dataset_name,
            "estimator": est_name,
            "paper_label": paper_label_by_str.get(est_name, est_name),
            # PRR del paper (normalizzato: 0 = caso, 1 = oracolo), con
            # intervallo bootstrap al 95%.
            "prr": _prr_from_arrays(ue, quality, max_rejection),
            "prr_ci_low": ci_low,
            "prr_ci_high": ci_high,
            # Solo l'area, la colonna riportata fino al 29/09: per confronto.
            "prr_raw": AL.prr_raw(ue, quality, max_rejection),
            "quality_metric": quality_name,
            "mean_quality": mean_quality,
            "n_instances": n_instances,
            # Frazione di istanze in cui lo stimatore non ha prodotto un
            # numero. Per i metodi verbalized coincide col parse-failure rate
            # (confidenza non estraibile dal testo generato); per gli altri
            # metodi dovrebbe essere zero.
            "nan_rate": float(np.isnan(ue).mean()),
            # Frazione di punteggi distinti: vicina a 0 = il metodo non ordina
            # quasi nulla (NumSet, o campioni tutti uguali).
            "distinct_fraction": AL.distinct_fraction(ue),
            # Errori fra le risposte date con piu' fiducia (10% piu'
            # confidente), da leggere accanto all'error rate complessivo.
            "error_rate_top10": AL.error_rate_most_confident(
                ue, quality, CORRECTNESS_THRESHOLD, 0.10),
            "error_rate_overall": err_overall,
            # Vecchia metrica, con un tetto quando gli errori sono tanti.
            "silent_failure_rate": silent_failure_rate(ue, quality),
        })

    stats_df = pd.DataFrame(rows)
    per_instance_df = pd.DataFrame(per_instance)
    per_instance_df.insert(0, "dataset", dataset_name)
    per_instance_df.insert(0, "model", model_name)
    return stats_df, per_instance_df


# Colonne dei file per-istanza che non sono punteggi di uno stimatore.
PER_INSTANCE_META_COLUMNS = ("model", "dataset", "instance_index", "quality", "greedy_text")


def _library_versions():
    import importlib.metadata as md
    out = {}
    for pkg in ("torch", "transformers", "lm-polygraph", "bitsandbytes", "accelerate"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = "?"
    return out


def record_run_conditions(results_dir, row):
    """Scheda delle condizioni di esecuzione: una riga per cella (modello x
    dataset x sezione) con precisione, attenzione, formato del prompt, budget di
    token, arresti, batch, campionatore e versioni delle librerie. Serve a
    poter scrivere in tesi, per ogni numero, in che condizioni e' stato
    ottenuto, senza ricostruirlo dai log."""
    path = os.path.join(results_dir, "run_conditions.csv")
    try:
        df = pd.DataFrame([row])
        if os.path.exists(path):
            old = pd.read_csv(path)
            df = pd.concat([old, df], ignore_index=True).drop_duplicates(
                subset=["section", "model", "dataset", "instance_offset"], keep="last")
        df.to_csv(path, index=False)
    except Exception:
        print("  (scrittura di run_conditions.csv fallita, ignoro)")


def run_model_on_dataset(model, model_name, dataset_name, examples, cfg, args, paper_label_by_str,
                         use_chat_template, estimators_factory=None, content_transform=None,
                         max_new_tokens_override=None, quantized=None, weights_model_name=None,
                         stop_strings_key="stop_strings", section_tag="main", instance_offset=0,
                         log_label=None):
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
    - `max_new_tokens_override`: sostituisce il max_new_tokens del dataset.
    `instance_offset` e' la posizione della prima istanza nella cella (per i
    blocchi), `log_label` l'etichetta messa davanti alle righe di log."""
    label = log_label or f"{model_name}/{dataset_name}"
    print(f"\n--- {label} ---")
    ds_start = time.time()
    df, timing_df, stats_df, per_instance_df = None, None, None, None
    try:
        contents = [ex["content"] for ex in examples]
        if content_transform is not None:
            contents = [content_transform(c) for c in contents]

        if use_chat_template and getattr(model.tokenizer, "chat_template", None) is None:
            print(f"  ATTENZIONE {label}: il tokenizer non ha un chat template, uso il prompt "
                  f"semplice (risultato NON confrontabile con gli altri modelli).")
            use_chat_template = False
        if use_chat_template:
            prompts = [format_chat_prompt(model.tokenizer, c) for c in contents]
        else:
            prompts = [format_prompt(c, cfg["plain_suffix"]) for c in contents]
        references = [ex["reference"] for ex in examples]
        # Controllo del doppio BOS sul tokenizer VERO (vedi format_chat_prompt):
        # la sequenza che il modello riceve deve iniziare con al massimo un BOS.
        bos_id = getattr(model.tokenizer, "bos_token_id", None)
        if bos_id is not None and prompts:
            first_ids = model.tokenizer(prompts[0])["input_ids"]
            n_bos = 0
            while n_bos < len(first_ids) and first_ids[n_bos] == bos_id:
                n_bos += 1
            if n_bos > 1:
                raise RuntimeError(f"{label}: il prompt tokenizzato inizia con {n_bos} token BOS.")

        # `model_name` puo' essere un'etichetta di variante ("full precision
        # (bf16)") invece di un nome in MODELS: in quel caso il conteggio dei
        # parametri va cercato sotto il nome del modello vero, passato dal
        # chiamante in weights_model_name.
        weights_name = weights_model_name or model_name
        is_quantized = uses_quantization(weights_name) if quantized is None else quantized
        batch_size = batch_size_for(weights_name, args.batch_size, is_quantized)
        deberta_batch_size = deberta_batch_size_for(weights_name, is_quantized)
        if batch_size != args.batch_size or deberta_batch_size != DEBERTA_BATCH_SIZE_DEFAULT:
            print(f"  [{label}] batch ridotto "
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
        print(f"  [{label}] stringhe di arresto ({stop_strings_key}): {stop_strings!r}")

        # Un OOM non deve far perdere la cella: si riprova con batch via via
        # piu' piccoli (prima si passava subito a 1/1, che con l'NLI a batch 1
        # significa 100 chiamate a DeBERTa per domanda). Se fallisce anche
        # l'ultimo tentativo, l'errore risale e lo gestisce l'esecuzione a
        # blocchi.
        attempts = []
        for att in ((batch_size, deberta_batch_size),
                    (1, max(1, deberta_batch_size // 4)),
                    (1, 1)):
            if att not in attempts:
                attempts.append(att)
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
            own_phase = {str(e): ESTIMATOR_OWN_PHASE.get(type(e).__name__) for e in base_estimators}
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
                nxt = attempts[attempt_idx + 1]
                print(f"  OOM su {label} con batch generazione={gen_bs}, NLI={nli_bs}: "
                      f"riprovo con {nxt[0]}/{nxt[1]}.")
                del man
                gc.collect()
                torch.cuda.empty_cache()

        greedy_texts = [str(t) for t in man.stats.get("greedy_texts", [])]
        # Diagnostica: generazioni vuote. Con le stringhe di arresto un modello
        # che iniziasse la risposta andando a capo produrrebbe una risposta
        # vuota; va visto subito, non scoperto nelle figure.
        n_empty = sum(1 for t in greedy_texts if not t.strip())
        if greedy_texts and n_empty / len(greedy_texts) > 0.05:
            print(f"  ATTENZIONE {label}: {n_empty}/{len(greedy_texts)} generazioni vuote.")
        log_gpu_mem(f"{label} done")
        peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else np.nan

        stats_df, per_instance_df = compute_instance_level_stats(
            man, model_name, dataset_name, paper_label_by_str,
            args.max_rejection, args.n_bootstrap, SEED,
        )
        if per_instance_df is not None:
            # Indice dell'istanza nella cella (serve ad appaiare modelli diversi
            # sulle stesse domande) e testo generato, per poter rileggere le
            # risposte e ricalcolare la metrica di qualita' senza GPU.
            per_instance_df.insert(2, "instance_index",
                                   np.arange(instance_offset, instance_offset + len(per_instance_df)))
            if len(greedy_texts) == len(per_instance_df):
                per_instance_df.insert(4, "greedy_text", greedy_texts)
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
            messaggio = (f"{label}: {qname} medio = {acc:.3f} "
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
        # I valori di man.metrics vengono sostituiti con quelli ricalcolati
        # dagli array per-istanza (pareggi in valore atteso, AUC casuale
        # esatta): cosi' una cella eseguita in un colpo e una eseguita a blocchi
        # danno esattamente gli stessi numeri. Il valore della libreria resta
        # in value_lmpolygraph per tracciabilita'.
        df["value_lmpolygraph"] = df["value"]
        q_arr, _, s_arr = extract_instance_level(man)
        raw_name = str(PredictionRejectionArea(max_rejection=args.max_rejection))
        new_vals = []
        for e, um, v in zip(df["estimator"], df["ue_metric"], df["value"]):
            if e in s_arr and um == raw_name:
                new_vals.append(AL.prr_raw(s_arr[e], q_arr, args.max_rejection))
            elif e in s_arr and um == raw_name + "_normalized":
                new_vals.append(AL.prr_normalized(s_arr[e], q_arr, args.max_rejection))
            else:
                new_vals.append(v)
        df["value"] = new_vals
        n_ok = df["value"].notna().sum() if "value" in df.columns else 0
        print(f"{label}: {n_ok}/{len(df)} righe metrica con valore.")

        # Tempo per fase (vedi PHASE_OF_CALCULATOR).
        phase_seconds = {}
        for calc_name, seconds in stat_timing_dict.items():
            phase = QUALITY_PHASE if calc_name == QUALITY_PHASE else classify_stat_calculator(calc_name)
            phase_seconds[phase] = phase_seconds.get(phase, 0.0) + seconds
        for est_str, seconds in timing_dict.items():
            phase = own_phase.get(est_str) or "aritmetica_stimatori"
            phase_seconds[phase] = phase_seconds.get(phase, 0.0) + seconds
        n_inst = max(len(examples), 1)

        timing_rows = []
        for est_str, seconds in timing_dict.items():
            closure = closures.get(est_str, {"GreedyProbsCalculator"})
            # Costo pieno standalone: se questa fosse l'unica tecnica in
            # esecuzione, dovrebbe pagarsi da sola ogni calcolatore della sua
            # catena di dipendenze (generazione greedy, campioni, modelli
            # ausiliari, forward extra), oltre alla propria chiamata. I tempi
            # sono quelli misurati calcolatore per calcolatore. La metrica di
            # qualita' NON entra: sul telefono non c'e' una risposta di
            # riferimento con cui confrontarsi.
            full = seconds + sum(stat_timing_dict.get(c, 0.0) for c in closure)
            phases = sorted({PHASE_OF_CALCULATOR.get(c, "non_classificato") for c in closure}
                            | ({own_phase[est_str]} if own_phase.get(est_str) else set()))
            timing_rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "estimator": est_str,
                "paper_label": paper_label_by_str.get(est_str, est_str),
                # Costo marginale: solo la chiamata dello stimatore su
                # statistiche gia' pronte (utile a chi ne calcola molti insieme).
                "seconds": seconds,
                "seconds_marginal_per_instance": seconds / n_inst,
                # Costo pieno: quello che conta per la scelta on-device.
                "seconds_full_standalone": full,
                "seconds_full_per_instance": full / n_inst,
                "needs_sampling": bool(closure & _SAMPLING_CALCULATORS) or bool(own_phase.get(est_str)),
                "needs_nli": bool(closure & _NLI_CALCULATORS),
                "needs_cross_encoder": bool(closure & _CROSS_ENCODER_CALCULATORS),
                "needs_extra_forward": bool(closure & _EXTRA_FORWARD_CALCULATORS),
                "calculators": ";".join(sorted(closure)),
                "phases": ";".join(phases),
                "peak_memory_gb": peak_mem_gb,
                "n_instances": n_inst,
            })
        timing_df = pd.DataFrame(timing_rows)
        # Tempi per fase della cella (somma su tutti i calcolatori), anche
        # nel CSV dei tempi: una riga per fase con estimator = "__fase__:<nome>".
        for phase, seconds in phase_seconds.items():
            timing_df = pd.concat([timing_df, pd.DataFrame([{
                "model": model_name, "dataset": dataset_name,
                "estimator": f"__fase__:{phase}", "paper_label": f"__fase__:{phase}",
                "seconds": seconds, "seconds_marginal_per_instance": seconds / n_inst,
                "seconds_full_standalone": seconds, "seconds_full_per_instance": seconds / n_inst,
                "peak_memory_gb": peak_mem_gb, "n_instances": n_inst,
            }])], ignore_index=True)

        for phase, seconds in sorted(phase_seconds.items(), key=lambda kv: -kv[1]):
            print(f"  [fase {label}] {phase}: {seconds:.1f}s totali "
                  f"({seconds / n_inst:.3f}s per istanza)")

        record_run_conditions(args.results_dir, {
            "section": section_tag, "model": model_name, "dataset": dataset_name,
            "instance_offset": instance_offset, "n_instances": len(examples),
            "weights_model": weights_name,
            "precision": "4-bit nf4" if is_quantized else "bf16",
            "attention": getattr(model.model.config, "_attn_implementation", "?"),
            "prompt_format": "chat_template" if use_chat_template else "plain",
            "bos_in_prompt_stripped": bool(use_chat_template and tokenizer_adds_bos(model.tokenizer)),
            "max_new_tokens": max_new_tokens, "stop_strings": json.dumps(stop_strings),
            "generation_batch": gen_bs, "nli_batch": nli_bs, "sampler": SAMPLER,
            "quality_metric": str(generation_metric),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            **{f"version_{k}": v for k, v in _library_versions().items()},
        })
        del man
    except Exception:
        print(f"!!! {label} fallito, salto alla prossima combinazione.")
        traceback.print_exc()
        df, timing_df, stats_df, per_instance_df = None, None, None, None
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Tempo {label}: {time.time() - ds_start:.1f}s")
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
        **{c: (c, "first") for c in ("needs_cross_encoder", "needs_extra_forward", "calculators",
                                     "phases")
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
            "needs_cross_encoder", "needs_extra_forward", "calculators", "phases",
            "peak_memory_gb", "n_instances"]
    return agg[[c for c in cols if c in agg.columns]]


def _move_aside(results_dir, path):
    """Sposta un file o una cartella di risultati superati in
    results_dir/_superati/<data-ora>/, mantenendo il percorso relativo. Non
    cancella niente: i risultati vecchi restano consultabili."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rel = os.path.relpath(path, results_dir)
    dest = os.path.join(results_dir, "_superati", stamp, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    k = 1
    while os.path.exists(dest):
        dest = f"{os.path.join(results_dir, '_superati', stamp, rel)}.{k}"
        k += 1
    shutil.move(path, dest)
    return os.path.relpath(dest, results_dir)


def prepare_chunk_dir(args, chunk_dir):
    """Crea la cartella dei blocchi di una cella e la marca con l'impronta del
    run. I blocchi gia' presenti vengono ripresi solo se l'impronta coincide:
    senza questo controllo, i blocchi lasciati da una versione precedente del
    codice (con gli stessi nomi di cartella) venivano ripresi come validi. Con
    --force_resume l'utente dichiara che la modifica non tocca generazioni e
    metriche, e i blocchi vengono tenuti."""
    stamp_path = os.path.join(chunk_dir, "run_fingerprint.txt")
    current_fp = getattr(args, "run_fingerprint", None)
    if current_fp and os.path.isdir(chunk_dir) and os.listdir(chunk_dir):
        saved_fp = None
        if os.path.exists(stamp_path):
            with open(stamp_path) as f:
                saved_fp = f.read().strip()
        if saved_fp != current_fp and not getattr(args, "force_resume", False):
            dest = _move_aside(args.results_dir, chunk_dir)
            print(f"  blocchi in {os.path.basename(chunk_dir)} prodotti da un'altra versione del codice "
                  f"(impronta {saved_fp or 'assente'}), spostati in {dest}: la cella riparte da zero.")
    os.makedirs(chunk_dir, exist_ok=True)
    if current_fp:
        with open(stamp_path, "w") as f:
            f.write(current_fp + "\n")


def quant_variants(model_name):
    """Le due serie del confronto quantizzazione. L'etichetta contiene il nome
    del modello: la ripresa, le cartelle dei blocchi e le figure sono
    indicizzate da essa, e prima (solo "quantized (4-bit nf4)" / "full
    precision (bf16)") un rilancio con un altro --quant_compare_model nella
    stessa cartella riusava in silenzio le celle del modello precedente."""
    return [(f"{model_name} 4-bit nf4", True), (f"{model_name} bf16", False)]


def run_cell_chunked(section_tag, model, model_name, dataset_name, examples, cfg, args,
                     paper_label_by_str, **kwargs):
    """Come run_model_on_dataset (stessi argomenti e stesso valore di ritorno),
    ma a blocchi con checkpoint. Vedi il commento sopra."""
    chunk_size = int(getattr(args, "chunk_size", 0) or 0)
    n = len(examples)
    if chunk_size <= 0 or n <= chunk_size:
        return run_model_on_dataset(model, model_name, dataset_name, examples, cfg, args,
                                    paper_label_by_str, section_tag=section_tag, **kwargs)

    chunk_dir = os.path.join(args.results_dir, "chunks", _safe_name(section_tag),
                             f"{_safe_name(model_name)}__{_safe_name(dataset_name)}__n{n}_c{chunk_size}")
    if getattr(args, "no_resume", False) and os.path.isdir(chunk_dir):
        shutil.rmtree(chunk_dir)
    prepare_chunk_dir(args, chunk_dir)

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
                                 chunk_args, paper_label_by_str, section_tag=section_tag,
                                 instance_offset=lo,
                                 log_label=f"{model_name}/{dataset_name} blocco {ci + 1}/{n_chunks}",
                                 **kwargs),
            hi - lo)
        skipped_here = []
        if part is None:
            print(f"!!! blocco {ci + 1}/{n_chunks} di {model_name}/{dataset_name} fallito: "
                  f"lo rieseguo un'istanza alla volta.")
            sub_parts = []
            for k in range(lo, hi):
                sub = _part_from_result(
                    run_model_on_dataset(model, model_name, dataset_name, examples[k:k + 1],
                                         cfg, chunk_args, paper_label_by_str,
                                         section_tag=section_tag, instance_offset=k,
                                         log_label=f"{model_name}/{dataset_name} istanza {k}",
                                         **kwargs),
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
                if c not in PER_INSTANCE_META_COLUMNS and c in common]
    dropped = [c for c in per_inst.columns if c not in PER_INSTANCE_META_COLUMNS and c not in common]
    if dropped:
        print(f"ATTENZIONE {model_name}/{dataset_name}: stimatori assenti in qualche blocco, "
              f"esclusi dalla cella: {dropped}")
    meta_cols = [c for c in PER_INSTANCE_META_COLUMNS if c in common]
    per_inst = per_inst[meta_cols + est_cols]
    quality = per_inst["quality"].to_numpy(dtype=float)
    scores = {c: per_inst[c].to_numpy(dtype=float) for c in est_cols}
    meta = pd.concat([p["meta"] for p in parts], ignore_index=True)
    quality_name = meta["quality_metric"].iloc[0]

    stats_df, per_instance_df = stats_from_arrays(
        quality, quality_name, scores, model_name, dataset_name, paper_label_by_str,
        args.max_rejection, args.n_bootstrap, SEED)
    if stats_df is None:
        return None, None, None, None
    # Indice dell'istanza e testo generato, dai blocchi.
    for pos, c in ((2, "instance_index"), (4, "greedy_text")):
        if c in per_inst.columns and c not in per_instance_df.columns:
            per_instance_df.insert(min(pos, per_instance_df.shape[1]), c, per_inst[c].to_numpy())
    n_done = meta["n_instances"].sum()
    pf = meta["parse_failure_sum"]
    stats_df["answer_parse_failure_rate"] = (pf.sum() / n_done) if pf.notna().any() else np.nan
    stats_df["n_skipped_instances"] = skipped_total

    prr_df_template = parts[0]["prr"].copy()
    # Gli stessi due valori che UEManager.eval_ue() scriverebbe in man.metrics
    # su tutta la cella -- l'area ("prr_0.5") e il PRR del paper
    # ("prr_0.5_normalized") -- ricalcolati sulle istanze riunite, con i
    # pareggi in valore atteso (vedi analysis_lib).
    raw_name = str(PredictionRejectionArea(max_rejection=args.max_rejection))
    values = []
    for e, um in zip(prr_df_template["estimator"], prr_df_template["ue_metric"]):
        if e not in scores:
            values.append(np.nan)
        elif um == raw_name:
            values.append(AL.prr_raw(scores[e], quality, args.max_rejection))
        elif um == raw_name + "_normalized":
            values.append(AL.prr_normalized(scores[e], quality, args.max_rejection))
        else:
            values.append(np.nan)
    prr_df = prr_df_template
    if "value_lmpolygraph" in prr_df.columns:
        prr_df["value_lmpolygraph"] = np.nan  # valore di un solo blocco: non ha senso sulla cella
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
                                        attn_implementation=attn_implementation_for(
                                            model_name, model_id, args.cache_dir, hf_token))
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
    raw_df = (combined[combined["ue_metric"] == prr_metric_name(args.max_rejection)].copy()
              if "ue_metric" in combined.columns else combined.copy())
    raw_df["paper_label"] = raw_df["estimator"].map(paper_label_by_str).fillna(raw_df["estimator"])

    stats_all = pd.concat(stats_dfs, ignore_index=True) if stats_dfs else None
    if stats_all is not None:
        # Porta gli intervalli di confidenza accanto ai valori PRR, cosi' che
        # le funzioni di plotting possano disegnare le barre d'errore.
        raw_df = raw_df.merge(
            stats_all[[c for c in MAPPED_STATS_COLUMNS if c in stats_all.columns]],
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
    sono pochissimi errori da trovare e la stima e' rumorosa (intervalli di
    confidenza larghi). Il PRR del paper non dipende meccanicamente
    dall'accuracy come l'area grezza usata fino al 29/09, ma va comunque letto
    accanto a questa tabella."""
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


def compute_rank_transfer(stats_df, results_dir, per_instance_df=None,
                          anchor=REPLICATION_ANCHOR_MODEL, max_rejection=0.5, n_resamples=200):
    """Kendall tau fra la classifica dei metodi UQ del modello-ancora a 7B e
    quella di ogni altro modello, DATASET PER DATASET, con intervallo bootstrap
    appaiato sulle domande (vedi analysis_lib.kendall_tau_with_ci per il
    perche' non si usa piu' il p-value di scipy ne' la media fra dataset).

    Risponde alla domanda "le conclusioni del benchmark, costruite su modelli
    da 7-12B, sopravvivono scendendo a 0.35-4B?". Tau vale 1 se i due ranking
    coincidono, 0 se sono scorrelati, -1 se sono invertiti."""
    if stats_df is None or stats_df.empty:
        return None
    if anchor not in stats_df["model"].unique():
        print(f"!!! Modello-ancora {anchor} assente dai risultati, salto Kendall tau.")
        return None

    rows = []
    for dataset_name, st_d in stats_df.groupby("dataset"):
        piv = st_d.pivot_table(index="estimator", columns="model", values="prr", aggfunc="first")
        if anchor not in piv.columns:
            continue
        for model_name in piv.columns:
            if model_name == anchor:
                continue
            pair = piv[[anchor, model_name]].dropna()
            if len(pair) < 3:
                continue
            methods = list(pair.index)
            # Stima puntuale dalle celle intere; se ci sono i punteggi
            # per-istanza viene sostituita da quella sulle sole domande comuni
            # ai due modelli, cioe' le stesse su cui e' calcolato l'intervallo
            # (altrimenti stima e intervallo verrebbero da insiemi diversi).
            tau = AL._kendall(pair[anchor].to_numpy(), pair[model_name].to_numpy())
            lo = hi = np.nan
            n_inst = np.nan
            if per_instance_df is not None:
                ca = per_instance_df[(per_instance_df["model"] == anchor)
                                     & (per_instance_df["dataset"] == dataset_name)]
                cb = per_instance_df[(per_instance_df["model"] == model_name)
                                     & (per_instance_df["dataset"] == dataset_name)]
                if (len(ca) and len(cb) and all(m in ca.columns and m in cb.columns for m in methods)):
                    tau_paired, lo, hi, n_inst = AL.kendall_tau_with_ci(
                        ca, cb, methods, max_rejection, n_resamples, SEED)
                    if np.isfinite(tau_paired):
                        tau = tau_paired
            rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "params_B": MODEL_PARAMS_B.get(model_name, np.nan),
                "kendall_tau_vs_anchor": tau,
                "tau_ci_low": lo,
                "tau_ci_high": hi,
                "n_methods_compared": len(pair),
                "n_instances_paired": n_inst,
            })

    if not rows:
        return None
    tau_df = pd.DataFrame(rows).sort_values(["params_B", "model", "dataset"])
    path = os.path.join(results_dir, "rank_transfer_kendall_tau.csv")
    tau_df.to_csv(path, index=False)
    print(f"Salvato: {path}")
    print(tau_df.round(3).to_string(index=False))
    return tau_df


# Modello scelto per il confronto quantizzato vs non-quantizzato (vedi
# --run_quant_comparison in main()). MedGemma-4B-it e Gemma3-4B-it sono
# esclusi a priori: producono logit NaN sotto bitsandbytes 4-bit (vedi
# CANNOT_QUANTIZE_MODELS), quindi un confronto quantizzato/non-quantizzato su
# di loro non e' fattibile. Tra i modelli rimasti scegliamo LFM2-1.2B: la
# quantizzazione comprime maggiormente un modello con piu' parametri, quindi un
# eventuale effetto sulla qualita' delle stime di incertezza e' piu'
# probabilmente misurabile rispetto al modello da 350M. Con
# --quant_compare_model Mistral-7B-it si confronta invece l'ancora.
# (Questa costante era stata cancellata per errore nel commit f447892 insieme
# al codice delle figure: main.py in quella versione non partiva.)
QUANT_COMPARE_MODEL_DEFAULT = "LFM2-1.2B"


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
# Colonne delle statistiche per-istanza portate accanto ai valori PRR nei file
# *_mapped.csv (che make_figures.py legge).
MAPPED_STATS_COLUMNS = ["model", "dataset", "estimator", "prr_ci_low", "prr_ci_high", "prr_raw",
                        "mean_quality", "quality_metric", "nan_rate", "distinct_fraction",
                        "error_rate_top10", "error_rate_overall", "silent_failure_rate"]
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
    "prr_ci_low", "prr_ci_high", "prr_raw", "quality_metric", "mean_quality",
    "n_instances", "nan_rate", "distinct_fraction", "error_rate_top10",
    "error_rate_overall", "silent_failure_rate",
}
PER_INSTANCE_REQUIRED_COLUMNS = {"model", "dataset", "quality"}


# Impronta del codice e delle impostazioni che determinano generazioni e
# metriche. Viene scritta in results_dir/run_fingerprint.json al primo lancio;
# una ripresa con un'impronta diversa viene RIFIUTATA (salvo --force_resume):
# prima un rilancio senza --no_resume dopo una modifica al codice mescolava in
# silenzio celle vecchie e nuove nello stesso file (il codice avvertiva solo a
# stampa, e i blocchi in chunks/ venivano ripresi senza alcun controllo).
FINGERPRINT_FILES = ("main.py", "dataset_prep.py", "batched_sampling.py", "analysis_lib.py")


def compute_run_fingerprint(args):
    here = os.path.dirname(os.path.abspath(__file__))
    files = {}
    for name in FINGERPRINT_FILES:
        path = os.path.join(here, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                files[name] = hashlib.sha256(f.read()).hexdigest()[:16]
    settings = {
        "max_rejection": args.max_rejection,
        "n_test_samples": args.n_test_samples,
        "sampler": args.sampler,
        "anchor_precision": args.anchor_precision,
        "verbalized_max_new_tokens": args.verbalized_max_new_tokens,
    }
    digest = hashlib.sha256(json.dumps({"files": files, "settings": settings},
                                       sort_keys=True).encode()).hexdigest()[:16]
    return {"fingerprint": digest, "files": files, "settings": settings}


def check_run_fingerprint(args):
    """Confronta l'impronta attuale con quella salvata. Esce con errore se la
    ripresa mescolerebbe versioni diverse."""
    path = os.path.join(args.results_dir, "run_fingerprint.json")
    current = compute_run_fingerprint(args)
    os.makedirs(args.results_dir, exist_ok=True)
    # Qualunque risultato di qualunque sezione (non solo la pipeline principale:
    # prima un cartella con solo la griglia clinica o il verbalized contava come
    # vuota e l'impronta veniva riscritta senza controllo).
    existing = [e for e in os.listdir(args.results_dir)
                if e == "chunks" or (e.endswith(".csv") and not e.startswith("."))]
    has_results = bool(existing)
    if args.no_resume and has_results:
        # anche le figure: altrimenti quelle delle sezioni non rilanciate
        # resterebbero accanto ai CSV nuovi senza piu' i dati da cui vengono
        existing += [e for e in os.listdir(args.results_dir)
                     if e.endswith((".png", ".md")) and not e.startswith(".")]
    if args.no_resume and has_results:
        # --no_resume = si riparte da zero in TUTTE le sezioni. I risultati
        # presenti vengono spostati da parte subito, non quando il run
        # raggiunge la loro cella: prima un run --no_resume interrotto (o
        # limitato con --models) riscriveva l'impronta, e il rilancio
        # successivo riprendeva come validi i checkpoint vecchi delle celle e
        # delle sezioni che quel run non aveva toccato.
        moved = [_move_aside(args.results_dir, os.path.join(args.results_dir, e))
                 for e in sorted(existing)]
        print(f"--no_resume: {len(moved)} file/cartelle di risultati precedenti spostati in "
              f"{os.path.dirname(moved[0])}/ (non cancellati).")
    if args.no_resume or not has_results:
        with open(path, "w") as f:
            json.dump(current, f, indent=2)
        return current
    saved = None
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
    if saved is not None and saved.get("fingerprint") == current["fingerprint"]:
        print(f"Impronta del run invariata ({current['fingerprint']}): ripresa sicura.")
        return current
    if saved is None:
        motivo = ("la cartella contiene risultati di una versione del codice precedente "
                  "all'impronta (prima del 29/09)")
    else:
        diff_files = sorted(k for k in set(saved.get("files", {})) | set(current["files"])
                            if saved.get("files", {}).get(k) != current["files"].get(k))
        diff_set = sorted(k for k in current["settings"]
                          if saved.get("settings", {}).get(k) != current["settings"][k])
        motivo = f"file cambiati: {diff_files or '-'}; impostazioni cambiate: {diff_set or '-'}"
    if args.force_resume:
        print(f"ATTENZIONE --force_resume: riprendo nonostante {motivo}.")
        with open(path, "w") as f:
            json.dump(current, f, indent=2)
        return current
    raise SystemExit(
        f"!!! Ripresa rifiutata in {args.results_dir}: {motivo}. Riprendere mescolerebbe risultati "
        f"di due versioni nelle stesse figure. Usa una --results_dir nuova, oppure --no_resume "
        f"(ricalcola tutto), oppure --force_resume se la modifica non tocca generazioni e metriche.")


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
    parser.add_argument("--anchor_precision", choices=["bf16", "4bit"], default="bf16",
                         help="Precisione dell'ancora Mistral-7B nella pipeline principale e nelle "
                              "sezioni extra (default: %(default)s, come nel paper). '4bit' riproduce "
                              "le run fino al 29/09.")
    parser.add_argument("--sampler", choices=["batched", "library"], default="batched",
                         help="Generatore dei K campioni: 'batched' (batched_sampling.py, una chiamata "
                              "per tutti i campioni, memoria costante) o 'library' "
                              "(SamplingGenerationCalculator di lm-polygraph, per confronto). "
                              "Default: %(default)s.")
    parser.add_argument("--force_resume", action="store_true",
                         help="Riprende dai checkpoint anche se l'impronta del codice e delle "
                              "impostazioni e' cambiata (vedi run_fingerprint.json). Da usare solo "
                              "se la modifica non tocca generazioni ne' metriche (es. un commento).")
    args = parser.parse_args()

    global SAMPLER
    SAMPLER = args.sampler
    set_anchor_precision(args.anchor_precision)

    print_banner()
    print(f"Ancora {REPLICATION_ANCHOR_MODEL}: {ANCHOR_PRECISION}. Campionatore: {SAMPLER}. "
          f"Modelli in bf16: {sorted(NO_QUANT_MODELS)}.")

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
    fp = check_run_fingerprint(args)
    args.run_fingerprint = fp["fingerprint"]
    print(f"Impronta del run: {fp['fingerprint']} (salvata in run_fingerprint.json)")

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
                                         attn_implementation=attn_implementation_for(
                                            model_name, model_id, args.cache_dir, hf_token))
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

    raw_df = (results_df[results_df["ue_metric"] == prr_metric_name(args.max_rejection)].copy()
              if "ue_metric" in results_df.columns else results_df.copy())
    raw_df["paper_label"] = raw_df["estimator"].map(paper_label_by_str).fillna(raw_df["estimator"])

    if stats_all is not None:
        raw_df = raw_df.merge(
            stats_all[[c for c in MAPPED_STATS_COLUMNS if c in stats_all.columns]],
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
            per_inst_all = (pd.concat(all_per_instance_dfs, ignore_index=True)
                            if all_per_instance_dfs else None)
            if per_inst_all is not None and "instance_index" in per_inst_all.columns:
                per_inst_all = per_inst_all.drop_duplicates(
                    subset=["model", "dataset", "instance_index"], keep="last")
            compute_rank_transfer(stats_labeled, args.results_dir, per_inst_all,
                                  max_rejection=args.max_rejection)
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
            # Righe "__fase__:<nome>": tempo totale di ogni fase per cella
            # (generazione greedy, campionamento, NLI, cross-encoder, metrica
            # di qualita', ...). Vanno in un file a parte; le tabelle per
            # metodo le escludono.
            is_phase = timing_all["estimator"].astype(str).str.startswith("__fase__:")
            phases = timing_all[is_phase].copy()
            if not phases.empty:
                phases["phase"] = phases["estimator"].str.replace("__fase__:", "", regex=False)
                phases[["model", "dataset", "phase", "seconds", "seconds_marginal_per_instance",
                        "n_instances"]].rename(
                    columns={"seconds_marginal_per_instance": "seconds_per_instance"}).to_csv(
                    os.path.join(args.results_dir, "phase_timings.csv"), index=False)
            timing_all = timing_all[~is_phase]

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
                # Si salva la metrica nuova (error rate fra le risposte piu'
                # confidenti, accanto a quello complessivo); il vecchio silent
                # failure rate ha un tetto e non distingue nulla quando gli
                # errori sono molti (vedi analysis_lib.error_rate_most_confident).
                if severity_stats is not None and "error_rate_top10" in severity_stats.columns:
                    err = severity_stats.pivot_table(
                        index="paper_label", columns=["dataset", "model"],
                        values="error_rate_top10", aggfunc="first",
                    )
                    overall = (severity_stats.groupby(["dataset", "model"])["error_rate_overall"]
                               .first())
                    err.loc["(error rate complessivo)"] = [overall.get(c, np.nan) for c in err.columns]
                    err_path = os.path.join(args.results_dir, "error_rate_most_confident.csv")
                    err.to_csv(err_path)
                    print(f"Salvato: {err_path}")
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
        # BUG CORRETTO il 29/09: il controllo guardava CHAT_TEMPLATE_MODELS
        # (che include anche Mistral) invece dei modelli che non tollerano la
        # quantizzazione, e cosi' impediva il confronto proprio sull'ancora.
        if quant_model_name in CANNOT_QUANTIZE_MODELS:
            print(f"!!! {quant_model_name} produce NaN sotto quantizzazione 4-bit (vedi "
                  "CANNOT_QUANTIZE_MODELS), confronto non eseguibile su questo modello -- salto.")
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

                variants = quant_variants(quant_model_name)
                for variant_label, use_quant in variants:
                    pending = [d for d in dataset_examples if (d, variant_label) not in quant_already_done]
                    if not pending:
                        continue
                    try:
                        model = load_whitebox_model(model_id, args.cache_dir, hf_token=hf_token,
                                                    use_quantization=use_quant,
                                                    attn_implementation=attn_implementation_for(
                                                        quant_model_name, model_id, args.cache_dir, hf_token))
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
                        quant_df[quant_df["ue_metric"] == prr_metric_name(args.max_rejection)].copy()
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
