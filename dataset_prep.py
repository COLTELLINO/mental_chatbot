"""Loader di dataset, metriche di qualita' e configurazioni DATASETS /
SEVERITY_DATASETS.

Modulo separato da main.py per tenere main.py leggibile: qui vive tutto cio'
che riguarda "da dove vengono i dati e come si costruisce il prompt", mentre
main.py si occupa di eseguire il benchmark e produrre le figure.

Ogni loader restituisce una lista di dict:
    {"content": <corpo del prompt, senza suffisso finale>,
     "reference": <testo/lettera/numero di riferimento>}
cosi' che il resto della pipeline (format_prompt / format_chat_prompt,
generation_metric) sia identico per tutti i dataset.
"""

import ast
import re

import numpy as np
from datasets import load_dataset

from lm_polygraph.generation_metrics import AccuracyMetric, AlignScore
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric


# ---------------------------------------------------------------------------
# SEZIONE 1 -- I quattro dataset di selective QA della Sezione 5.1 del paper
# (Vashurin et al., arXiv:2406.15627).
#
# Dimensioni ridotte rispetto al paper (che usa 2000 istanze/dataset, 100 per
# subject su MMLU) per contenere il tempo di calcolo sul cluster: n=100 per
# CoQA/TriviaQA/MMLU, n=50 per GSM8k (le sue generazioni sono molto piu'
# lunghe -- ~128 token medi di ragionamento contro i ~4 degli altri tre --
# quindi pesano di piu' su tutti gli stimatori a campionamento).
#
# I prompt 5-shot (TriviaQA/MMLU/GSM8k) sono costruiti in uno stile coerente
# col resto della pipeline, NON sono una replica byte-per-byte dei template di
# lm-evaluation-harness citati nel paper -- da tenere come caveat in tesi.
# Per MMLU il paper campiona fino a 100 domande PER SUBJECT (57 subject); qui
# se ne campionano 100 totali su tutti i subject insieme (stesso budget degli
# altri dataset), quindi non tutti i 57 subject saranno rappresentati.
# ---------------------------------------------------------------------------

def prepare_coqa(n_test, seed, cache_dir=None):
    """CoQA (stanfordnlp/coqa, split 'validation', 500 righe = 500 istanze di
    test come da Table 2 del paper). Ogni riga e' una conversazione: usiamo
    tutte le domande/risposte tranne l'ultima come storico (few-shot
    "naturale") e l'ultima domanda come target, come descritto nel paper."""
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
    """GSM8k (openai/gsm8k, config 'main'). Split train/test hanno 7473/1319
    righe, identici a Table 2. 5-shot dal train set, con la 'reference'
    ridotta al solo numero finale (dopo '####')."""
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
    finale: X" richiesto nel prompt) e lo confronta col numero di riferimento
    con una tolleranza numerica minima.

    Range: 0.0 o 1.0 per istanza (binaria), quindi la media su un dataset e'
    l'accuracy nell'intervallo [0, 1]."""

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


# Regex condivisa da tutti i task multiple-choice: tiene solo la lettera
# iniziale (A-D) e scarta qualunque testo dopo. Serve perche' AccuracyMetric
# confronta stringhe esatte e i modelli aggiungono spesso punteggiatura o
# spiegazioni dopo la lettera.
_MCQ_IGNORE_REGEX = r"(?<=[ABCDabcd])[\s\S]*"


# ---------------------------------------------------------------------------
# SEZIONE 2 -- Griglia 2x2 severita' x formato di risposta.
#
# L'obiettivo e' separare due effetti che nei risultati precedenti erano
# confusi tra loro: la SEVERITA' delle conseguenze di un errore e il FORMATO
# della risposta (scelta multipla vs testo libero). Confrontando MedQA (MCQ,
# benigno) con MedicationQA (testo libero, pericoloso) i due effetti si
# muovevano insieme, quindi un eventuale calo del PRR non era attribuibile
# all'uno o all'altro.
#
#                  | Formato MCQ            | Formato a risposta libera
#   Severita' alta | MedQAbstain-LT         | MedicationQA
#   Severita' bassa| MedQAbstain-Safe       | MedQuAD
#
# Vincolo metodologico rispettato: le due celle sulla stessa RIGA (stesso
# formato) usano la stessa identica metrica di qualita', quindi il confronto
# verticale (severita' a parita' di formato) e' pulito. Confronti tra righe
# diverse restano solo qualitativi, perche' AccuracyMetric (binaria) e
# AlignScore (continua) non sono direttamente commensurabili.
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


def _parse_options_dict(raw_value):
    """I campi `options` e `original_options` di MedQAbstain sono stringhe che
    contengono la repr Python di un dict (es. "{'A': 'testo', 'B': ...}"),
    non JSON: literal_eval e' il modo sicuro di riconvertirle."""
    if isinstance(raw_value, dict):
        return raw_value
    return ast.literal_eval(raw_value)


def prepare_medqabstain(n_test, seed, cache_dir=None, split="LT", source="medqa_4opt"):
    """MedQAbstain (disi-unibo-nlp/MedQAbstain), split 'LT' (Life-Threatening)
    o 'Safe', annotati dagli autori in base alla SEVERITA' CLINICA di una
    decisione sbagliata: in LT un'azione errata plausibile puo' causare danno
    grave o morte, in Safe difficilmente causa danno serio.

    IMPORTANTE -- usiamo `original_options` / `original_answer`, non `options`
    / `answer`. Il benchmark originale e' costruito per misurare l'ASTENSIONE:
    rimuove la risposta corretta dalle opzioni e aggiunge "I abstain", che
    diventa l'unica risposta giusta. Su quel task i modelli si astengono
    quasi mai (e' la tesi del paper che lo introduce), quindi l'accuracy
    collasserebbe verso zero e il PRR non avrebbe abbastanza risposte corrette
    da ordinare per essere informativo. Usando i campi `original_*` si
    recupera la domanda MCQ intatta -- accuracy in un regime normale -- e si
    conserva comunque cio' che qui serve davvero: l'annotazione di severita'
    LT/Safe, che e' il contributo del dataset.

    Filtri applicati:
    - `modality == "text-only"`: le istanze multimodali richiederebbero un
      encoder di immagini che la nostra pipeline non ha.
    - `dataset == source` (default "medqa_4opt", cioe' MedQA-USMLE): tiene il
      numero di opzioni fisso a 4 tra LT e Safe. Mescolare fonti con un numero
      diverso di opzioni cambierebbe la baseline casuale (25% con 4 opzioni,
      10% con 10) e renderebbe le due celle non confrontabili. Se il filtro
      lascia meno istanze di n_test, si allarga automaticamente a tutte le
      fonti text-only e lo si segnala.
    """
    raw = load_dataset("disi-unibo-nlp/MedQAbstain", cache_dir=cache_dir)[split]
    raw = raw.filter(lambda ex: ex["modality"] == "text-only")

    filtered = raw.filter(lambda ex: ex["dataset"] == source)
    if len(filtered) >= n_test:
        raw = filtered
    else:
        print(f"ATTENZIONE MedQAbstain/{split}: solo {len(filtered)} istanze con "
              f"dataset=={source} (ne servono {n_test}); uso tutte le fonti text-only "
              f"({len(raw)} istanze). Il numero di opzioni puo' variare tra istanze.")

    raw = raw.shuffle(seed=seed).select(range(min(n_test, len(raw))))
    fewshot_block = _build_medqa_fewshot_block()

    examples = []
    for ex in raw:
        opts = _parse_options_dict(ex["original_options"])
        gold = ex["original_answer"].strip()
        if gold not in opts:
            continue
        opt_lines = "\n".join(f"{letter}) {text}" for letter, text in sorted(opts.items()))
        letters_str = ", ".join(sorted(opts.keys()))
        content = (
            "Sei un assistente medico che risponde a domande in stile esame USMLE.\n"
            "Leggi il quesito clinico e rispondi SOLO con la lettera dell'opzione corretta "
            f"({letters_str}), senza altro testo.\n\n"
            f"{fewshot_block}\n\n"
            f"Domanda: {ex['question'].strip()}\n\n{opt_lines}"
        )
        examples.append({"content": content, "reference": gold})
    return examples


def prepare_medqabstain_lt(n_test, seed, cache_dir=None):
    """Cella MCQ / severita' ALTA della griglia 2x2."""
    return prepare_medqabstain(n_test, seed, cache_dir=cache_dir, split="LT")


def prepare_medqabstain_safe(n_test, seed, cache_dir=None):
    """Cella MCQ / severita' BASSA della griglia 2x2."""
    return prepare_medqabstain(n_test, seed, cache_dir=cache_dir, split="Safe")


# Finestra di lunghezza (caratteri) applicata alle risposte di riferimento di
# ENTRAMBE le celle a risposta libera. AlignScore confronta la generazione col
# testo di riferimento: se una cella avesse riferimenti da poche righe e
# l'altra da interi articoli, il punteggio differirebbe per la lunghezza dei
# riferimenti e non per la difficolta' del task, contaminando il confronto di
# severita'. Il limite superiore e' scelto per stare vicino al max_new_tokens
# di 100 usato per generare (~400-600 caratteri di testo inglese).
_FREE_TEXT_MIN_CHARS = 80
_FREE_TEXT_MAX_CHARS = 600


def prepare_medicationqa(n_test, seed, cache_dir=None):
    """MedicationQA (truehealth/medicationqa, unico split 'train', 690 righe).
    Cella a risposta libera / severita' ALTA: domande reali di pazienti su
    farmaci (dosaggi, interazioni, effetti collaterali), dove un'informazione
    sbagliata puo' avere conseguenze gravi. Risposte in testo libero, nessun
    few-shot nel dataset originale, quindi prompt zero-shot diretto.

    Alcune righe hanno Question/Answer vuoti (raccolte da FAQ reali, non
    curate come benchmark): vengono scartate prima del campionamento. La
    finestra di lunghezza sulle risposte e' la stessa usata per MedQuAD, per
    non introdurre una differenza sistematica tra le due celle a risposta
    libera (vedi commento in prepare_medquad)."""
    raw = load_dataset("truehealth/medicationqa", cache_dir=cache_dir)["train"]
    raw = raw.filter(
        lambda ex: ex["Question"] and ex["Answer"]
        and _FREE_TEXT_MIN_CHARS <= len(ex["Answer"].strip()) <= _FREE_TEXT_MAX_CHARS
    )
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


# Tipi di domanda MedQuAD considerati puramente informativi: chiedono cosa sia
# una malattia, quanto sia diffusa, quali sintomi dia, come si erediti. Un
# errore su questi e' un'informazione sbagliata, non un'indicazione ad agire.
# Sono esclusi di proposito i tipi che porterebbero a una decisione clinica
# (treatment, prevention, exams and tests, ...), che appartengono al lato ad
# alta severita' della griglia e confonderebbero la cella benigna.
_MEDQUAD_BENIGN_TYPES = {
    "information",
    "frequency",
    "symptoms",
    "causes",
    "inheritance",
    "susceptibility",
    "genetic changes",
}


def prepare_medquad(n_test, seed, cache_dir=None):
    """MedQuAD (lavita/MedQuAD, 47441 coppie domanda-risposta da siti NIH).
    Cella a risposta libera / severita' BASSA della griglia 2x2: stesso
    dominio clinico e stesso formato di MedicationQA, ma domande informative
    (vedi _MEDQUAD_BENIGN_TYPES) dove un errore non si traduce in un'azione
    pericolosa sul paziente.

    Usa la stessa identica metrica (AlignScore), lo stesso max_new_tokens e la
    stessa finestra di lunghezza dei riferimenti di MedicationQA: e' il
    vincolo che rende interpretabile il confronto di severita' a parita' di
    formato. Nota: in questa versione del dataset le risposte provenienti da
    MedlinePlus sono gia' state rimosse per motivi di copyright, quindi non
    c'e' sovrapposizione di fonte con MedicationQA."""
    raw = load_dataset("lavita/MedQuAD", cache_dir=cache_dir)["train"]
    raw = raw.filter(
        lambda ex: ex["question"] and ex["answer"]
        and ex["question_type"] in _MEDQUAD_BENIGN_TYPES
        and _FREE_TEXT_MIN_CHARS <= len(ex["answer"].strip()) <= _FREE_TEXT_MAX_CHARS
    )
    raw = raw.shuffle(seed=seed).select(range(min(n_test, len(raw))))
    examples = []
    for ex in raw:
        content = (
            "Sei un assistente medico che risponde a domande informative di pazienti "
            "su malattie e condizioni cliniche. Rispondi in modo chiaro e conciso.\n\n"
            f"Domanda: {ex['question'].strip()}"
        )
        examples.append({"content": content, "reference": ex["answer"].strip()})
    return examples


# ---------------------------------------------------------------------------
# SEZIONE 3 -- Prompt per i metodi verbalized.
#
# Verbalized1S e Linguistic1S leggono la confidenza dichiarata dal modello
# dentro `greedy_texts`, cioe' la STESSA generazione greedy condivisa da tutti
# gli altri stimatori. Per usarli serve quindi un prompt che chieda
# esplicitamente la confidenza e un max_new_tokens abbastanza grande da
# contenerla: entrambe le cose cambiano la generazione condivisa e quindi i
# valori di tutti gli altri metodi. Per questo i verbalized girano in una
# sezione separata del benchmark, con la propria configurazione, e i loro
# numeri non vanno mescolati con quelli della pipeline principale.
# ---------------------------------------------------------------------------

# Mappa espressione -> confidenza per Linguistic1S. L'ORDINE CONTA: l'estimator
# fa `if expression in answer` e si ferma al primo match, quindi le espressioni
# composte vanno prima delle loro sottostringhe ("Very Low" prima di "Low",
# "Very High" prima di "High"), altrimenti "Very High" verrebbe classificata
# come "High".
LINGUISTIC_EXPRESSIONS = {
    "Very Low": 0.1,
    "Very High": 0.9,
    "Low": 0.3,
    "Moderate": 0.5,
    "High": 0.7,
}

# Regex per Verbalized1S: cattura il numero dopo "Confidence:". L'estimator fa
# `1 - float(match.group(1))`, quindi il numero deve essere una probabilita' in
# [0, 1]; se la regex non matcha restituisce NaN, ed e' esattamente il segnale
# che usiamo per misurare il parse-failure rate.
VERBALIZED_CONFIDENCE_REGEX = r"Confidence:\s*([01]?\.?\d+)"

_VERBALIZED_NUMERIC_INSTRUCTION = (
    "\n\nDopo la risposta, su una nuova riga, dichiara quanto sei sicuro della tua "
    "risposta nel formato esatto:\nConfidence: <numero tra 0.00 e 1.00>"
)

_VERBALIZED_LINGUISTIC_INSTRUCTION = (
    "\n\nDopo la risposta, su una nuova riga, dichiara quanto sei sicuro della tua "
    "risposta nel formato esatto:\nConfidence: <una tra Very Low, Low, Moderate, High, Very High>"
)


def build_verbalized_content(content, style):
    """Aggiunge in coda al prompt la richiesta di dichiarare la confidenza.
    `style` e' "numeric" (per Verbalized1S) o "linguistic" (per Linguistic1S)."""
    if style == "numeric":
        return content + _VERBALIZED_NUMERIC_INSTRUCTION
    if style == "linguistic":
        return content + _VERBALIZED_LINGUISTIC_INSTRUCTION
    raise ValueError(f"Stile verbalized sconosciuto: {style}")


# ---------------------------------------------------------------------------
# SEZIONE 4 -- Costruzione del prompt finale e registri di configurazione.
# ---------------------------------------------------------------------------

def format_prompt(content, suffix):
    """Prompt a completamento di testo semplice (modelli base, es. LFM2)."""
    return content + suffix


def format_chat_prompt(tokenizer, content):
    """Prompt con chat template per modelli instruction-tuned (MedGemma,
    Gemma3, Mistral-Instruct)."""
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# I quattro dataset della replica del paper (Sezione 5.1).
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
        # margine ampio per lasciare spazio al ragionamento completo prima del
        # numero finale.
        "max_new_tokens": 200,
        "plain_suffix": "\nSoluzione:",
        "generation_metric_factory": lambda: GSM8kAccuracyMetric(),
    },
}


# Griglia 2x2 severita' x formato. I campi `severity` e `answer_format`
# servono a main.py per disporre i pannelli della figura e per verificare che
# le celle confrontate direttamente condividano la metrica.
SEVERITY_DATASETS = {
    "MedQAbstain-LT": {
        "loader": prepare_medqabstain_lt,
        "n_test": 100,
        "max_new_tokens": 3,
        "plain_suffix": "\n\nRisposta:",
        "generation_metric_factory": lambda: AccuracyMetric(output_ignore_regex=_MCQ_IGNORE_REGEX),
        "severity": "alta",
        "answer_format": "MCQ",
        "severity_label": "SEVERITA' ALTA -- un'azione errata puo' causare danno grave o morte",
    },
    "MedQAbstain-Safe": {
        "loader": prepare_medqabstain_safe,
        "n_test": 100,
        "max_new_tokens": 3,
        "plain_suffix": "\n\nRisposta:",
        "generation_metric_factory": lambda: AccuracyMetric(output_ignore_regex=_MCQ_IGNORE_REGEX),
        "severity": "bassa",
        "answer_format": "MCQ",
        "severity_label": "SEVERITA' BASSA -- un'azione errata difficilmente causa danno serio",
    },
    "MedicationQA": {
        "loader": prepare_medicationqa,
        "n_test": 100,
        "max_new_tokens": 100,
        "plain_suffix": "\nRisposta:",
        "generation_metric_factory": lambda: AlignScore(),
        "severity": "alta",
        "answer_format": "libera",
        "severity_label": "SEVERITA' ALTA -- informazione errata su farmaci, conseguenze potenzialmente gravi",
    },
    "MedQuAD": {
        "loader": prepare_medquad,
        "n_test": 100,
        "max_new_tokens": 100,
        "plain_suffix": "\nRisposta:",
        "generation_metric_factory": lambda: AlignScore(),
        "severity": "bassa",
        "answer_format": "libera",
        "severity_label": "SEVERITA' BASSA -- domande informative su malattie, nessuna indicazione ad agire",
    },
}
