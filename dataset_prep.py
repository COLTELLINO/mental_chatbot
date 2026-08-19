"""Loader di dataset e configurazione DATASETS/SAFETY_DATASETS.

Modulo separato da main.py cosi' da poter essere importato sia da main.py
sia da train_stat_builder.py (il builder custom usato per Mahalanobis
Distance/RDE/Relative Mahalanobis Distance) senza incorrere nel problema
"main" vs "__main__": lm-polygraph importa i builder custom via
import_module(nome_modulo), e se quel modulo facesse `from main import ...`
mentre main.py e' in esecuzione come script (__main__), Python
ri-eseguirebbe l'intero file da capo sotto un nome di modulo diverso invece
di riusare l'istanza gia' caricata. Un modulo dedicato, mai eseguito come
script, evita il problema."""

import re

import numpy as np
from datasets import load_dataset

from lm_polygraph.generation_metrics import AccuracyMetric, AlignScore
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric


# ---------------------------------------------------------------------------
# Caricamento dataset (Sezione 5.1 del paper). Ogni loader restituisce una
# lista di dict {"content": <corpo del prompt, senza suffisso finale>,
# "reference": <testo/lettera/numero di riferimento>}, cosi' che il resto
# della pipeline (format_prompt/format_chat_prompt, generation_metric) sia
# identico per tutti i dataset.
#
# Dimensioni ridotte rispetto al paper (che usa 2000 istanze/dataset, 100/
# subject per MMLU) per contenere il tempo di calcolo sul cluster: n=100 per
# CoQA/TriviaQA/MMLU, n=50 per GSM8k (le sue generazioni sono molto piu'
# lunghe -- ~128 token medi di ragionamento contro i ~4 degli altri tre --
# quindi pesano di piu' su tutti gli stimatori a campionamento).
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
# Obiettivo: vedere se il ranking dei metodi UQ cambia tra uno scenario "a
# basso rischio" e uno "ad alto rischio".
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
    istruzione a rispondere solo con la lettera."""
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


# ---------------------------------------------------------------------------
# Background set generico per Relative Mahalanobis Distance (Ren et al.
# 2023): serve un dataset AMPIO e FUORI DOMINIO rispetto ai nostri 4 task QA,
# usato per calcolare una seconda Mahalanobis Distance "di riferimento" da
# sottrarre a quella in-domain (RMD(x) = MD(x) - MD_0(x)). Riusare uno dei
# nostri 4 dataset come background non avrebbe senso (sono tutti QA dello
# stesso tipo, non genererebbero il contrasto che la metrica richiede):
# usiamo invece paragrafi di Wikipedia (Salesforce/wikitext,
# wikitext-103-raw-v1, split 'train', 1.8M righe -- verificato via HF API,
# stesso tipo di corpus generico usato in letteratura per questo scopo).
# Non serve una "reference"/risposta: il calcolatore usa solo gli embedding
# della generazione, quindi il testo di riferimento e' fittizio (stringa
# vuota) e non entra in nessun calcolo.
# ---------------------------------------------------------------------------

def prepare_wikitext_background(n, seed, cache_dir=None):
    """Campiona n paragrafi non vuoti da WikiText-103 (raw) come testo
    generico fuori dominio. Ogni "prompt" e' semplicemente l'inizio del
    paragrafo stesso (il modello lo completa liberamente): non e' un
    task di QA, serve solo a far emergere embedding rappresentativi di
    testo generico per il centroide di background di RMD."""
    raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)["train"]
    # Le righe di wikitext-103-raw-v1 includono molte righe vuote/titoli di
    # sezione cortissimi (formattazione originale del dump): filtriamo per
    # tenere solo paragrafi di prosa con una lunghezza minima ragionevole.
    raw = raw.filter(lambda ex: len(ex["text"].strip()) > 200)
    raw = raw.shuffle(seed=seed).select(range(min(n, len(raw))))
    examples = []
    for ex in raw:
        text = ex["text"].strip()
        # Tronchiamo a ~800 caratteri: paragrafi wikitext possono essere
        # molto lunghi, e qui serve solo materiale generico per gli
        # embedding, non l'intero articolo. Wrappato con una breve
        # istruzione (invece di testo grezzo) cosi' l'interazione resta
        # coerente con lo stile prompt/chat-template usato ovunque nel
        # resto della pipeline per i modelli instruction-tuned.
        examples.append({
            "content": f"Continua il seguente testo in modo naturale:\n\n{text[:800]}",
            "reference": "",
        })
    return examples


def format_prompt(content, suffix):
    """Prompt a completamento di testo semplice (modelli base LFM2)."""
    return content + suffix


def format_chat_prompt(tokenizer, content):
    """Prompt con chat template per modelli instruction-tuned (MedGemma/Gemma3 -it)."""
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
