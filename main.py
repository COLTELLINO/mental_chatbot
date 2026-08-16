import argparse
import gc
import os
import re
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.dataset import Dataset as PolygraphDataset
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.processor import Logger
from lm_polygraph.utils.builder_enviroment_stat_calculator import BuilderEnvironmentStatCalculator
from lm_polygraph.defaults.register_default_stat_calculators import register_default_stat_calculators
from lm_polygraph.generation_metrics import AccuracyMetric, AlignScore
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from lm_polygraph.ue_metrics import PredictionRejectionArea
from lm_polygraph.estimators import *

SEED = 3407

MODELS = {
    "LFM2-350M":      "LiquidAI/LFM2-350M",
    "LFM2-1.2B":      "LiquidAI/LFM2-1.2B",
    "MedGemma-4B-it": "google/medgemma-4b-it",
    "Gemma3-4B-it":   "google/gemma-3-4b-it",
}

GATED_MODELS = {"MedGemma-4B-it", "Gemma3-4B-it"}

# Instruction-tuned ("-it") models expect an explicit chat template (turn
# markers like <start_of_turn>/<end_of_turn>); given the same plain-text
# completion prompt as the LFM2 base models, they don't generate correctly.
# Their prompts are built with tokenizer.apply_chat_template() instead.
# They also produce NaN logits under bitsandbytes 4-bit quantization
# (Gemma3-family issue) and are loaded in bf16 instead — see
# load_whitebox_model's use_quantization flag.
CHAT_TEMPLATE_MODELS = {"MedGemma-4B-it", "Gemma3-4B-it"}


def print_banner():
    print("=" * 70)
    print("UQ BENCHMARK -- Sezione 5.1 Vashurin et al. (arXiv:2406.15627)")
    print("Selective QA: CoQA, TriviaQA, MMLU, GSM8k")
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


# ---------------------------------------------------------------------------
# Caricamento dataset (Sezione 5.1 del paper). Ogni loader restituisce una
# lista di dict {"content": <corpo del prompt, senza suffisso finale>,
# "reference": <testo/lettera/numero di riferimento>}, cosi' che il resto
# della pipeline (format_prompt/format_chat_prompt, generation_metric) sia
# identico per tutti i dataset.
#
# Dimensioni ridotte rispetto al paper (che usa 2000 istanze/dataset, 100/
# subject per MMLU) per contenere il tempo di calcolo sul cluster: n=100 per
# CoQA/TriviaQA/MMLU (stessa scala usata finora per MedQA), n=50 per GSM8k
# (le sue generazioni sono molto piu' lunghe -- ~128 token medi di
# ragionamento contro i ~4 degli altri tre -- quindi pesano di piu' su tutti
# gli stimatori a campionamento). Deciso esplicitamente da Filo (2026-08-14).
#
# I prompt 5-shot (TriviaQA/MMLU/GSM8k) sono costruiti da noi in uno stile
# coerente col resto della pipeline, NON sono una replica byte-per-byte dei
# template di lm-evaluation-harness citati nel paper -- da tenere come
# caveat in tesi. Per MMLU il paper campiona fino a 100 domande PER SUBJECT
# (57 subject); noi campioniamo 100 domande totali su tutti i subject
# insieme (stesso budget usato per gli altri dataset), quindi non tutti i
# 57 subject saranno necessariamente rappresentati.
# ---------------------------------------------------------------------------

def prepare_coqa(n_test, seed, cache_dir=None):
    """CoQA (stanfordnlp/coqa, split 'validation', 500 righe = 500 istanze
    di test come da Table 2 del paper). Ogni riga e' una conversazione;
    usiamo tutte le domande/risposte tranne l'ultima come storico della
    conversazione (few-shot "naturale") e l'ultima domanda come target,
    esattamente come descritto nel paper."""
    raw = load_dataset("stanfordnlp/coqa", cache_dir=cache_dir)["validation"]
    raw = raw.shuffle(seed=seed).select(range(min(n_test, len(raw))))
    examples = []
    for row in raw:
        questions = row["questions"]
        answers = row["answers"]["input_text"]
        if len(questions) < 1:
            continue
        history = list(zip(questions[:-1], answers[:-1]))
        lines = [
            "Leggi il passaggio seguente e rispondi in modo breve e diretto alla domanda "
            "finale della conversazione, nello stesso stile delle risposte precedenti.",
            "",
            f"Passaggio: {row['story'].strip()}",
            "",
        ]
        for q, a in history:
            lines.append(f"D: {q.strip()}")
            lines.append(f"R: {a.strip()}")
        lines.append(f"D: {questions[-1].strip()}")
        examples.append({"content": "\n".join(lines), "reference": answers[-1]})
    return examples


def prepare_triviaqa(n_test, seed, cache_dir=None, n_fewshot=5):
    """TriviaQA (mandarjoshi/trivia_qa, config 'rc.nocontext' -- "without
    context" come nel paper). Split train/test hanno 138384/17210 righe,
    identici ai numeri di Table 2. 5-shot da esempi del train set."""
    raw = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", cache_dir=cache_dir)
    fewshot = raw["train"].shuffle(seed=seed).select(range(n_fewshot))
    fewshot_block = "\n\n".join(
        f"Domanda: {r['question'].strip()}\nRisposta: {r['answer']['value']}" for r in fewshot
    )
    test = raw["test"].shuffle(seed=seed).select(range(min(n_test, len(raw["test"]))))
    examples = []
    for r in test:
        content = (
            "Rispondi alla domanda in modo breve e diretto (poche parole), senza spiegazioni.\n\n"
            f"{fewshot_block}\n\nDomanda: {r['question'].strip()}"
        )
        examples.append({"content": content, "reference": r["answer"]["value"]})
    return examples


def prepare_mmlu(n_test, seed, cache_dir=None):
    """MMLU (cais/mmlu, config 'all'). 5-shot per-subject usando lo split
    'dev' (5 esempi per subject, protocollo standard MMLU), come nel paper."""
    raw = load_dataset("cais/mmlu", "all", cache_dir=cache_dir)
    letters = ["A", "B", "C", "D"]
    dev_by_subject = {}
    for r in raw["dev"]:
        dev_by_subject.setdefault(r["subject"], []).append(r)
    test = raw["test"].shuffle(seed=seed).select(range(min(n_test, len(raw["test"]))))
    examples = []
    for r in test:
        fewshot_rows = dev_by_subject.get(r["subject"], [])[:5]
        blocks = []
        for f in fewshot_rows:
            opt_lines = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(f["choices"]))
            blocks.append(f"Domanda: {f['question']}\n{opt_lines}\nRisposta: {letters[f['answer']]}")
        fewshot_block = "\n\n".join(blocks)
        opt_lines = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(r["choices"]))
        content = (
            "Rispondi SOLO con la lettera dell'opzione corretta (A, B, C o D), senza altro testo.\n\n"
            f"{fewshot_block}\n\nDomanda: {r['question'].strip()}\n{opt_lines}"
        )
        examples.append({"content": content, "reference": letters[r["answer"]]})
    return examples


def prepare_gsm8k(n_test, seed, cache_dir=None, n_fewshot=5):
    """GSM8k (openai/gsm8k, config 'main'). Split train/test hanno
    7473/1319 righe, identici a Table 2. 5-shot dal train set, con la
    risposta 'reference' ridotta al solo numero finale (dopo '####')."""
    raw = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)
    fewshot = raw["train"].shuffle(seed=seed).select(range(n_fewshot))
    blocks = []
    for r in fewshot:
        ans_clean = r["answer"].replace("####", "Risposta finale:")
        blocks.append(f"Problema: {r['question'].strip()}\nSoluzione: {ans_clean}")
    fewshot_block = "\n\n".join(blocks)
    test = raw["test"].shuffle(seed=seed).select(range(min(n_test, len(raw["test"]))))
    examples = []
    for r in test:
        content = (
            "Risolvi il problema passo per passo, poi scrivi il risultato finale preceduto "
            "esattamente da 'Risposta finale:'.\n\n"
            f"{fewshot_block}\n\nProblema: {r['question'].strip()}"
        )
        final = r["answer"].split("####")[-1].strip().replace(",", "")
        examples.append({"content": content, "reference": final})
    return examples


class GSM8kAccuracyMetric(GenerationMetric):
    """lm-polygraph non ha una metrica pronta per problemi matematici a
    risposta numerica (solo Accuracy per multiple-choice o AlignScore per
    testo libero). Estrae l'ULTIMO numero presente nel testo generato
    (robusto anche se il modello non segue esattamente il formato "Risposta
    finale: X" richiesto nel prompt) e lo confronta con il numero di
    riferimento con una tolleranza numerica minima."""

    def __init__(self):
        super().__init__(["greedy_texts"], "sequence")

    def __str__(self):
        return "GSM8kAccuracy"

    @staticmethod
    def _extract_number(text):
        cleaned = text.replace(",", "")
        matches = re.findall(r"-?\d+\.?\d*", cleaned)
        if not matches:
            return None
        try:
            return float(matches[-1])
        except ValueError:
            return None

    def __call__(self, stats, target_texts):
        preds = stats["greedy_texts"]
        scores = []
        for pred, ref in zip(preds, target_texts):
            pred_num = self._extract_number(pred)
            try:
                ref_num = float(str(ref).replace(",", ""))
            except (TypeError, ValueError):
                ref_num = None
            if pred_num is None or ref_num is None:
                scores.append(0.0)
            else:
                scores.append(1.0 if abs(pred_num - ref_num) < 1e-4 else 0.0)
        return np.array(scores)


# Regex condivisa con MMLU/MedQA-style: tiene solo la lettera A-D, scarta
# qualunque testo dopo (serve perche' AccuracyMetric confronta stringhe
# esatte e i modelli a volte aggiungono testo/punteggiatura extra).
_MCQ_IGNORE_REGEX = r"(?<=[ABCDabcd])[\s\S]*"


# ---------------------------------------------------------------------------
# Confronto extra "safe vs non-safe": stessi 4 modelli, stessa pipeline UQ,
# ma su due dataset medici con conseguenze molto diverse in caso di errore:
#   - MedQA-USMLE: domande da esame di medicina. Un errore costa solo un
#     punteggio piu' basso in un test, nessuna conseguenza reale.
#   - MedicationQA: domande reali di pazienti su farmaci (dosaggi,
#     interazioni, effetti collaterali). Un errore qui (dose sbagliata,
#     interazione non segnalata) puo' avere conseguenze gravi o letali.
# Voluto da Filo (2026-08-16) per vedere se il ranking dei metodi UQ cambia
# tra uno scenario "a basso rischio" e uno "ad alto rischio".
# ---------------------------------------------------------------------------

MEDQA_FEWSHOT_EXAMPLES = [
    {
        "question": "A 3-month-old baby died suddenly at night while asleep. His mother noticed that he had died only after she awoke in the morning. No cause of death was determined based on the autopsy. Which of the following precautions could have prevented the death of the baby?",
        "options": {
            "A": "Placing the infant in a supine position on a firm mattress while sleeping",
            "B": "Keeping the infant covered and maintaining a high room temperature",
            "C": "Application of a device to maintain the sleeping position",
            "D": "Avoiding pacifier use during sleep",
        },
        "answer_idx": "A",
    },
    {
        "question": "A 9-month-old female is brought to the emergency department after experiencing a seizure. She was born at home and was normal at birth according to her parents. Since then, they have noticed that she does not appear to be achieving developmental milestones as quickly as her siblings, and often appears lethargic. Physical exam reveals microcephaly, very light pigmentation (as compared to her family), and a \"musty\" body odor. The varied manifestations of this disease can most likely be attributed to which of the following genetic principles?",
        "options": {
            "A": "Anticipation",
            "B": "Multiple gene mutations",
            "C": "Pleiotropy",
            "D": "Variable expressivity",
        },
        "answer_idx": "C",
    },
    {
        "question": "A 68-year-old man presents to the emergency department with leg pain. He states that the pain started suddenly while he was walking outside. The patient has a past medical history of diabetes, hypertension, obesity, and atrial fibrillation. His temperature is 99.3°F (37.4°C), blood pressure is 152/98 mmHg, pulse is 97/min, respirations are 15/min, and oxygen saturation is 99% on room air. Physical exam is notable for a cold and pale left leg. The patient's sensation is markedly diminished in the left leg when compared to the right, and his muscle strength is 1/5 in his left leg. Which of the following is the best next step in management?",
        "options": {
            "A": "Graded exercise and aspirin",
            "B": "Heparin drip",
            "C": "Surgical thrombectomy",
            "D": "Tissue plasminogen activator",
        },
        "answer_idx": "B",
    },
]


def _build_medqa_fewshot_block():
    blocks = []
    for ex in MEDQA_FEWSHOT_EXAMPLES:
        opts = ex["options"]
        blocks.append(
            f"Domanda: {ex['question']}\n\n"
            f"A) {opts['A']}\nB) {opts['B']}\nC) {opts['C']}\nD) {opts['D']}\n\n"
            f"Risposta: {ex['answer_idx']}"
        )
    return "\n\n".join(blocks)


def prepare_medqa(n_test, seed, cache_dir=None):
    """MedQA-USMLE (GBaker/MedQA-USMLE-4-options, split 'test'). Scenario
    SAFE: multiple-choice da esame di medicina, few-shot (3 esempi) +
    istruzione a rispondere solo con la lettera. Stesso identico prompt gia'
    usato prima che il dataset principale passasse a CoQA/TriviaQA/MMLU/
    GSM8k (vedi commit 'port MedQA UQ benchmark notebook to main.py')."""
    raw = load_dataset("GBaker/MedQA-USMLE-4-options", cache_dir=cache_dir)
    fewshot_block = _build_medqa_fewshot_block()
    test_split = raw["test"].shuffle(seed=seed).select(range(min(n_test, len(raw["test"]))))
    examples = []
    for ex in test_split:
        opts = ex["options"]
        content = (
            "Sei un assistente medico che risponde a domande in stile esame USMLE.\n"
            "Leggi il quesito clinico e rispondi SOLO con la lettera dell'opzione corretta "
            "(A, B, C o D), senza altro testo.\n\n"
            f"{fewshot_block}\n\n"
            f"Domanda: {ex['question'].strip()}\n\n"
            f"A) {opts['A'].strip()}\nB) {opts['B'].strip()}\nC) {opts['C'].strip()}\nD) {opts['D'].strip()}"
        )
        examples.append({"content": content, "reference": ex["answer_idx"]})
    return examples


def prepare_medicationqa(n_test, seed, cache_dir=None):
    """MedicationQA (truehealth/medicationqa, unico split 'train', 690
    righe). Scenario NON-SAFE: domande reali di pazienti su farmaci
    (dosaggi, interazioni, effetti collaterali) con risposte in testo
    libero -- niente few-shot nel dataset originale, prompt a zero-shot
    diretto in stile CoQA/TriviaQA. Alcune righe hanno Question/Answer
    vuoti (dataset raccolto da FAQ reali, non curato come benchmark): le
    scartiamo prima di campionare."""
    raw = load_dataset("truehealth/medicationqa", cache_dir=cache_dir)["train"]
    raw = raw.filter(lambda ex: ex["Question"] and ex["Answer"])
    raw = raw.shuffle(seed=seed).select(range(min(n_test, len(raw))))
    examples = []
    for ex in raw:
        content = (
            "Sei un assistente farmaceutico che risponde a domande reali di pazienti sui farmaci "
            "(dosaggi, interazioni, effetti collaterali). Rispondi in modo chiaro e conciso.\n\n"
            f"Domanda: {ex['Question'].strip()}"
        )
        examples.append({"content": content, "reference": ex["Answer"].strip()})
    return examples


SAFETY_DATASETS = {
    "MedQA": {
        "loader": prepare_medqa,
        "n_test": 100,
        "max_new_tokens": 3,
        "plain_suffix": "\n\nRisposta:",
        "generation_metric_factory": lambda: AccuracyMetric(output_ignore_regex=_MCQ_IGNORE_REGEX),
        "safety_label": "SAFE -- errore = domanda d'esame sbagliata, nessuna conseguenza reale",
    },
    "MedicationQA": {
        "loader": prepare_medicationqa,
        "n_test": 100,
        # Le risposte reali (MedlinePlus/DailyMed) vanno da poche parole a
        # spiegazioni di 60-80 parole (~100-120 token): margine piu' ampio
        # di CoQA/TriviaQA per non troncare le risposte piu' lunghe.
        "max_new_tokens": 100,
        "plain_suffix": "\nRisposta:",
        "generation_metric_factory": lambda: AlignScore(),
        "safety_label": "NON-SAFE -- errore su farmaci = potenzialmente grave o letale",
    },
}

DATASETS = {
    "CoQA": {
        "loader": prepare_coqa,
        "n_test": 100,
        "max_new_tokens": 30,
        "plain_suffix": "\nR:",
        "generation_metric_factory": lambda: AlignScore(),
    },
    "TriviaQA": {
        "loader": prepare_triviaqa,
        "n_test": 100,
        "max_new_tokens": 20,
        "plain_suffix": "\nRisposta:",
        "generation_metric_factory": lambda: AlignScore(),
    },
    "MMLU": {
        "loader": prepare_mmlu,
        "n_test": 100,
        "max_new_tokens": 3,
        "plain_suffix": "\n\nRisposta:",
        "generation_metric_factory": lambda: AccuracyMetric(output_ignore_regex=_MCQ_IGNORE_REGEX),
    },
    "GSM8k": {
        "loader": prepare_gsm8k,
        "n_test": 50,
        # Target medio 128.6 token (Table 2, tokenizer Mistral 7B v0.2):
        # margine ampio per lasciare spazio al ragionamento completo prima
        # del numero finale (non calcoliamo il 99esimo percentile esatto
        # come nel paper, usiamo un valore fisso prudente).
        "max_new_tokens": 200,
        "plain_suffix": "\nSoluzione:",
        "generation_metric_factory": lambda: GSM8kAccuracyMetric(),
    },
}


def format_prompt(content, suffix):
    """Prompt a completamento di testo semplice (modelli base LFM2)."""
    return content + suffix


def format_chat_prompt(tokenizer, content):
    """Prompt con chat template per modelli instruction-tuned (MedGemma/Gemma3 -it)."""
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


FIGURE_A_MODELS_NOTE = (
    "Figura A ~ Vashurin et al. Fig. 2 (white-box, full access: StableLM v2 12b / "
    "Mistral v0.2 7b base), Mean PRR aggregato su CoQA/TriviaQA/MMLU/GSM8k -> sostituita "
    "da LFM2-350M / LFM2-1.2B / MedGemma-4B-it / Gemma3-4B-it."
)
FIGURE_B_MODELS_NOTE = (
    "Figura B ~ Vashurin et al. Fig. 3 (black-box/reflexive: StableLM v2 12b Chat / "
    "Mistral v0.2 7b Instruct / GPT-4o-mini), Mean PRR aggregato su CoQA/TriviaQA/MMLU/GSM8k -> "
    "sostituita dagli stessi 4 modelli nostri (nel nostro caso l'accesso e' sempre white-box, "
    "quindi i metodi 'black-box' producono lo stesso valore che avrebbero in accesso ristretto)."
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        result = self._estimator(stats)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
        key = str(self._estimator)
        self._timing_dict[key] = self._timing_dict.get(key, 0.0) + elapsed
        return result


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


def build_manager(model, dataset, estimators, cache_dir, max_rejection, max_new_tokens, generation_metric):
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

    man = UEManager(
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


def run_model_on_dataset(model, model_name, dataset_name, examples, cfg, args, paper_label_by_str, use_chat_template):
    """Esegue tutti gli estimator UQ di build_estimators() su un singolo
    modello gia' caricato contro un singolo dataset gia' preparato (prompt +
    reference). Ritorna (prr_df, timing_df), oppure (None, None) se la
    combinazione fallisce. Estratta dal loop principale (Sezione 5.1) cosi'
    da poter essere riusata identica dai confronti extra (safety,
    quantizzazione) senza duplicare tre volte la stessa logica."""
    print(f"\n--- {model_name} su {dataset_name} ---")
    ds_start = time.time()
    df, timing_df = None, None
    try:
        if use_chat_template:
            prompts = [format_chat_prompt(model.tokenizer, ex["content"]) for ex in examples]
        else:
            prompts = [format_prompt(ex["content"], cfg["plain_suffix"]) for ex in examples]
        references = [ex["reference"] for ex in examples]

        model_dataset = PolygraphDataset(prompts, references, batch_size=args.batch_size)

        timing_dict = {}
        estimators = [TimedEstimator(e, timing_dict) for e in build_estimators()]
        man = build_manager(
            model, model_dataset, estimators, args.cache_dir, args.max_rejection,
            cfg["max_new_tokens"], cfg["generation_metric_factory"](),
        )
        man()
        log_gpu_mem(f"{model_name}/{dataset_name} done")

        df = extract_prr_table(man, model_name)
        df["dataset"] = dataset_name
        n_ok = df["value"].notna().sum() if "value" in df.columns else 0
        print(f"{model_name}/{dataset_name}: {n_ok}/{len(df)} righe metrica con valore.")

        timing_rows = [
            {
                "model": model_name,
                "dataset": dataset_name,
                "estimator": est_str,
                "paper_label": paper_label_by_str.get(est_str, est_str),
                "seconds": seconds,
            }
            for est_str, seconds in timing_dict.items()
        ]
        timing_df = pd.DataFrame(timing_rows)
        del man
    except Exception:
        print(f"!!! {model_name}/{dataset_name} fallito, salto alla prossima combinazione.")
        traceback.print_exc()
        df, timing_df = None, None
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Tempo {model_name}/{dataset_name}: {time.time() - ds_start:.1f}s")
    return df, timing_df


def plot_prr_bars(df, labels, group_order, title, note, out_path, xlabel, add_mean=True):
    """Bar chart orizzontale generico: una barra per elemento di group_order
    (colonna 'model' del df, che puo' contenere sia nomi di modelli veri
    sia etichette di varianti come "quantized"/"full precision"), una riga
    per etichetta 'paper_label' in labels. Se add_mean=True aggiunge anche
    una barra "Mean". Usata sia per le Figure A/B aggregate sul paper sia
    per i confronti extra (safety, quantizzazione) -- stesso identico stile
    visivo, incluso il margine sinistro dinamico per non troncare le
    etichette piu' lunghe."""
    subset = df[df["paper_label"].isin(labels)]
    if subset.empty:
        print(f"!!! Nessun dato per {out_path}, salto.")
        return None
    pivot = subset.pivot_table(index="paper_label", columns="model", values="value", aggfunc="first")
    pivot = pivot.reindex(columns=group_order)
    row_mean = pivot.mean(axis=1)
    if add_mean:
        pivot["Mean"] = row_mean
    pivot = pivot.loc[row_mean.sort_values(ascending=True).index]  # barh: prima riga in cima = ultima disegnata
    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.4)))
    pivot.plot(kind="barh", ax=ax, width=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8)
    # Margine sinistro proporzionale alla label piu' lunga (in caratteri):
    # con figsize fisso a 10", il margine di default di matplotlib (~0.125)
    # tronca etichette lunghe tipo "Monte Carlo Normalized Sequence
    # Entropy" o "Eccentricity Jaccard Score". ~0.011 di frazione-figura
    # per carattere e' una stima prudente per il font di default a 10".
    max_label_len = max(len(str(lbl)) for lbl in pivot.index)
    left_margin = min(0.55, max(0.22, max_label_len * 0.011))
    fig.subplots_adjust(left=left_margin, bottom=0.16)
    wrapped_note = "\n".join(textwrap.wrap(note, width=115))
    fig.text(0.01, 0.02, wrapped_note, fontsize=7, va="bottom")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato: {out_path}")
    return pivot


def plot_safety_comparison(df, labels, group_order, out_path):
    """Confronto affiancato (2 pannelli, stesso ordine di metodi in
    entrambi) PRR su MedQA (scenario safe) vs MedicationQA (scenario
    non-safe/potenzialmente letale). L'ordine comune e' dato dalla media
    delle due "Mean PRR aggregate sui 4 modelli", cosi' lo stesso metodo
    occupa la stessa riga in entrambi i pannelli e il confronto visivo e'
    immediato."""
    subset = df[df["paper_label"].isin(labels)]
    if subset.empty:
        print(f"!!! Nessun dato per {out_path}, salto.")
        return
    pivots = {}
    for ds_name in SAFETY_DATASETS:
        ds_subset = subset[subset["dataset"] == ds_name]
        if ds_subset.empty:
            continue
        pivot = ds_subset.pivot_table(index="paper_label", columns="model", values="value", aggfunc="first")
        pivot = pivot.reindex(columns=group_order)
        pivot["Mean"] = pivot.mean(axis=1)
        pivots[ds_name] = pivot
    if len(pivots) < 2:
        print(f"!!! Solo {len(pivots)}/2 dataset disponibili per {out_path}, salto il confronto affiancato.")
        return

    ds_names = list(SAFETY_DATASETS.keys())
    combined_mean = sum(pivots[d]["Mean"] for d in ds_names) / len(ds_names)
    common_order = combined_mean.sort_values(ascending=True).index

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(common_order) * 0.4)), sharey=True)
    for i, (ax, ds_name) in enumerate(zip(axes, ds_names)):
        pivot = pivots[ds_name].reindex(common_order)
        pivot.plot(kind="barh", ax=ax, width=0.8, legend=False)
        ax.set_title(f"{ds_name}\n({SAFETY_DATASETS[ds_name]['safety_label']})", fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean PRR (raw, max_rejection=0.5)")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, legend_labels, loc="lower right", fontsize=8)
    fig.suptitle("PRR per metodo UQ e modello: MedQA (safe) vs MedicationQA (non-safe/potenzialmente letale)")
    max_label_len = max(len(str(lbl)) for lbl in common_order)
    left_margin = min(0.45, max(0.20, max_label_len * 0.009))
    fig.subplots_adjust(left=left_margin, bottom=0.1, top=0.90, wspace=0.06)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato: {out_path}")


# Modello scelto per il confronto quantizzato vs non-quantizzato (vedi
# --run_quant_comparison in main()). MedGemma-4B-it e Gemma3-4B-it sono
# esclusi a priori: producono logit NaN sotto bitsandbytes 4-bit (vedi
# CHAT_TEMPLATE_MODELS sopra) e per questo nella pipeline vengono SEMPRE
# caricati in bf16 -- un confronto quantizzato/non-quantizzato su di loro
# non e' fattibile con questo codice. Tra i due modelli rimasti (LFM2-350M,
# LFM2-1.2B) scegliamo il piu' grande: la quantizzazione 4-bit comprime
# maggiormente un modello con piu' parametri, quindi un eventuale effetto
# sulla qualita' delle stime di incertezza ha piu' probabilita' di essere
# misurabile rispetto al modello da 350M. Scelta di Filo (2026-08-16):
# "scegli te un modello".
QUANT_COMPARE_MODEL_DEFAULT = "LFM2-1.2B"


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
    parser.add_argument("--run_safety_comparison", action="store_true",
                         help="In piu' rispetto alla pipeline principale, esegue anche il confronto "
                              "MedQA (safe) vs MedicationQA (non-safe) su tutti e 4 i modelli "
                              "(fig_safety_comparison.png).")
    parser.add_argument("--run_quant_comparison", action="store_true",
                         help="In piu' rispetto alla pipeline principale, esegue anche il confronto "
                              "quantizzato (4-bit) vs non-quantizzato (bf16) su un modello "
                              "(--quant_compare_model) sui 4 dataset principali.")
    parser.add_argument("--quant_compare_model", type=str, default=QUANT_COMPARE_MODEL_DEFAULT,
                         choices=list(MODELS.keys()),
                         help="Modello su cui eseguire --run_quant_comparison (default: %(default)s). "
                              "MedGemma-4B-it/Gemma3-4B-it non sono utilizzabili: producono NaN sotto "
                              "quantizzazione 4-bit (vedi CHAT_TEMPLATE_MODELS).")
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
            use_quant = model_name not in CHAT_TEMPLATE_MODELS
            model = load_whitebox_model(model_id, args.cache_dir, hf_token=hf_token, use_quantization=use_quant)
            log_gpu_mem(f"{model_name} loaded")
        except Exception:
            print(f"!!! Caricamento di {model_name} fallito, salto tutti i suoi dataset.")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            continue

        for dataset_name in pending_datasets:
            cfg = DATASETS[dataset_name]
            df, timing_df = run_model_on_dataset(
                model, model_name, dataset_name, dataset_examples[dataset_name], cfg, args,
                paper_label_by_str, use_chat_template=(model_name in CHAT_TEMPLATE_MODELS),
            )
            if df is not None:
                all_metrics_dfs.append(df)
            if timing_df is not None:
                all_timing_dfs.append(timing_df)

            if all_metrics_dfs:
                pd.concat(all_metrics_dfs, ignore_index=True).to_csv(
                    os.path.join(args.results_dir, "results_partial.csv"), index=False
                )
                print(f"Checkpoint salvato dopo {model_name}/{dataset_name}.")
            if all_timing_dfs:
                pd.concat(all_timing_dfs, ignore_index=True).to_csv(
                    os.path.join(args.results_dir, "estimator_timings_partial.csv"), index=False
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

    raw_df = results_df[results_df["ue_metric"] == "prr_0.5"].copy() if "ue_metric" in results_df.columns else results_df.copy()
    raw_df["paper_label"] = raw_df["estimator"].map(paper_label_by_str).fillna(raw_df["estimator"])

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

    model_order = [m for m in MODELS.keys() if m in raw_df["model"].unique()]
    dataset_order = [d for d in DATASETS.keys() if d in raw_df["dataset"].unique()]

    # Mean PRR aggregato su tutti i task di selective QA, esattamente come
    # da didascalia originale delle Figure 2/3 del paper ("aggregated over
    # all selective QA tasks for each ... LLM separately").
    agg_df = raw_df.groupby(["model", "paper_label"], as_index=False)["value"].mean()

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

    # Tabella + grafico di efficienza: tempo (secondi) speso da ciascun
    # metodo UQ per modello, sommato su tutti i dataset.
    try:
        if all_timing_dfs:
            timing_all = pd.concat(all_timing_dfs, ignore_index=True).drop_duplicates(
                subset=["model", "dataset", "estimator"], keep="last"
            )
            timing_pivot = timing_all.pivot_table(
                index="paper_label", columns="model", values="seconds", aggfunc="sum"
            )
            timing_pivot = timing_pivot.reindex(columns=model_order)
            timing_pivot["Total"] = timing_pivot.sum(axis=1)
            timing_pivot = timing_pivot.sort_values("Total", ascending=False)
            timing_path = os.path.join(args.results_dir, "estimator_timing_table.csv")
            timing_pivot.to_csv(timing_path)
            print(f"Salvato: {timing_path}")

            # Scala log sull'asse x: i tempi coprono diversi ordini di
            # grandezza (es. BB P(True) molto piu' lento di tutto il resto),
            # in scala lineare tutte le barre tranne una sarebbero invisibili.
            plot_pivot = timing_pivot.drop(columns=["Total"]).loc[
                timing_pivot.sort_values("Total", ascending=True).index
            ]
            fig, ax = plt.subplots(figsize=(10, max(8, len(plot_pivot) * 0.4)))
            plot_pivot.plot(kind="barh", ax=ax, width=0.8, logx=True)
            ax.set_xlabel("Tempo totale di calcolo, sommato su tutti i batch e dataset (secondi, scala log)")
            ax.set_title("Tempo di calcolo per metodo UQ e modello — CoQA+TriviaQA+MMLU+GSM8k")
            ax.legend(loc="lower right", fontsize=8)
            fig.subplots_adjust(bottom=0.12)
            plt.tight_layout()
            timing_chart_path = os.path.join(args.results_dir, "estimator_timing_chart.png")
            plt.savefig(timing_chart_path, dpi=150)
            plt.close(fig)
            print(f"Salvato: {timing_chart_path}")
    except Exception:
        print("!!! tabella/grafico tempi falliti:")
        traceback.print_exc()

    # -------------------------------------------------------------------
    # Confronto extra 1: MedQA (safe) vs MedicationQA (non-safe). Stessi 4
    # modelli, stessa pipeline, dataset e figura separati dalla riproduzione
    # del paper qui sopra (non entrano in agg_df/fig_a/fig_b/table6).
    # -------------------------------------------------------------------
    if args.run_safety_comparison:
        print("\n--- Confronto extra: MedQA (safe) vs MedicationQA (non-safe) ---")
        try:
            safety_examples = {}
            for ds_name, cfg in SAFETY_DATASETS.items():
                n_test = args.n_test_samples if args.n_test_samples is not None else cfg["n_test"]
                print(f"Caricamento {ds_name} (n_test={n_test})...")
                try:
                    examples = cfg["loader"](n_test, SEED, cache_dir=args.datasets_cache_dir)
                    print(f"{ds_name}: {len(examples)} esempi caricati.")
                    safety_examples[ds_name] = examples
                except Exception:
                    print(f"!!! Caricamento di {ds_name} fallito, salto.")
                    traceback.print_exc()

            if len(safety_examples) < 2:
                print("!!! Meno di 2 dataset safety caricati con successo, salto il confronto.")
            else:
                safety_final_path = os.path.join(args.results_dir, "results_safety_comparison.csv")
                safety_metrics_dfs = []
                safety_already_done = set()
                if os.path.exists(safety_final_path):
                    try:
                        existing = pd.read_csv(safety_final_path)
                        safety_already_done = set(zip(existing["dataset"], existing["model"]))
                        safety_metrics_dfs.append(existing)
                        print("Combinazioni safety gia' completate:", safety_already_done)
                    except Exception as e:
                        print(f"results_safety_comparison.csv illeggibile ({e}) -- riparto senza skip.")

                for model_name, model_id in MODELS.items():
                    pending = [d for d in safety_examples if (d, model_name) not in safety_already_done]
                    if not pending:
                        continue
                    try:
                        use_quant = model_name not in CHAT_TEMPLATE_MODELS
                        model = load_whitebox_model(model_id, args.cache_dir, hf_token=hf_token, use_quantization=use_quant)
                    except Exception:
                        print(f"!!! Caricamento di {model_name} fallito, salto i suoi dataset safety.")
                        traceback.print_exc()
                        continue

                    for dataset_name in pending:
                        cfg = SAFETY_DATASETS[dataset_name]
                        df, _ = run_model_on_dataset(
                            model, model_name, dataset_name, safety_examples[dataset_name], cfg, args,
                            paper_label_by_str, use_chat_template=(model_name in CHAT_TEMPLATE_MODELS),
                        )
                        if df is not None:
                            safety_metrics_dfs.append(df)
                            pd.concat(safety_metrics_dfs, ignore_index=True).to_csv(safety_final_path, index=False)
                            print(f"Checkpoint safety salvato dopo {model_name}/{dataset_name}.")

                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

                if safety_metrics_dfs:
                    safety_df = pd.concat(safety_metrics_dfs, ignore_index=True)
                    safety_raw = (
                        safety_df[safety_df["ue_metric"] == "prr_0.5"].copy()
                        if "ue_metric" in safety_df.columns else safety_df.copy()
                    )
                    safety_raw["paper_label"] = safety_raw["estimator"].map(paper_label_by_str).fillna(safety_raw["estimator"])
                    safety_raw.to_csv(os.path.join(args.results_dir, "results_safety_comparison_mapped.csv"), index=False)

                    safety_model_order = [m for m in MODELS.keys() if m in safety_raw["model"].unique()]
                    plot_safety_comparison(
                        safety_raw, figure_a_labels, safety_model_order,
                        os.path.join(args.results_dir, "fig_safety_comparison.png"),
                    )
                    print("Confronto safety completato.")
                else:
                    print("!!! Nessuna combinazione safety completata con successo.")
        except Exception:
            print("!!! Confronto safety fallito:")
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
                        df, _ = run_model_on_dataset(
                            model, variant_label, dataset_name, dataset_examples[dataset_name], cfg, args,
                            paper_label_by_str, use_chat_template=False,
                        )
                        if df is not None:
                            quant_metrics_dfs.append(df)
                            pd.concat(quant_metrics_dfs, ignore_index=True).to_csv(quant_final_path, index=False)
                            print(f"Checkpoint quant salvato dopo {variant_label}/{dataset_name}.")

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
