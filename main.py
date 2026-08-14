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
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.dataset import Dataset as PolygraphDataset
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.processor import Logger
from lm_polygraph.utils.builder_enviroment_stat_calculator import BuilderEnvironmentStatCalculator
from lm_polygraph.defaults.register_default_stat_calculators import register_default_stat_calculators
from lm_polygraph.generation_metrics import AccuracyMetric
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

FEWSHOT_EXAMPLES = [
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


def print_banner():
    print("=" * 70)
    print("MEDQA UQ BENCHMARK")
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


def build_fewshot_block():
    blocks = []
    for ex in FEWSHOT_EXAMPLES:
        opts = ex["options"]
        blocks.append(
            f"Domanda: {ex['question']}\n\n"
            f"A) {opts['A']}\n"
            f"B) {opts['B']}\n"
            f"C) {opts['C']}\n"
            f"D) {opts['D']}\n\n"
            f"Risposta: {ex['answer_idx']}"
        )
    return "\n\n".join(blocks)


def build_user_content(example, fewshot_block):
    opts = example["options"]
    return (
        "Sei un assistente medico che risponde a domande in stile esame USMLE.\n"
        "Leggi il quesito clinico e rispondi SOLO con la lettera dell'opzione corretta (A, B, C o D), senza altro testo.\n\n"
        f"{fewshot_block}\n\n"
        f"Domanda: {example['question'].strip()}\n\n"
        f"A) {opts['A'].strip()}\n"
        f"B) {opts['B'].strip()}\n"
        f"C) {opts['C'].strip()}\n"
        f"D) {opts['D'].strip()}"
    )


def format_prompt(example, fewshot_block):
    """Plain-text completion prompt (LFM2 base models)."""
    return build_user_content(example, fewshot_block) + "\n\nRisposta:"


def format_chat_prompt(tokenizer, example, fewshot_block):
    """Chat-template prompt for instruction-tuned models (MedGemma/Gemma3 -it)."""
    content = build_user_content(example, fewshot_block)
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


FIGURE_A_MODELS_NOTE = (
    "Figura A ~ Vashurin et al. Fig. 2 (white-box, full access: StableLM v2 12b / "
    "Mistral v0.2 7b base) -> sostituita da LFM2-350M / LFM2-1.2B / MedGemma-4B-it / Gemma3-4B-it."
)
FIGURE_B_MODELS_NOTE = (
    "Figura B ~ Vashurin et al. Fig. 3 (black-box/reflexive: StableLM v2 12b Chat / "
    "Mistral v0.2 7b Instruct / GPT-4o-mini) -> sostituita dagli stessi 4 modelli nostri "
    "(nel nostro caso l'accesso e' sempre white-box, quindi i metodi 'black-box' producono "
    "lo stesso valore che avrebbero in accesso ristretto)."
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
     "reason": "Richiede fit di un modello di densita' su embeddings del train set MedQA "
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
               "che oggi contiene solo la lettera A/B/C/D (max_new_tokens=3, 'senza altro testo'). Per "
               "renderlo utile dovremmo cambiare prompt/lunghezza di generazione per tutti gli stimatori."},
    {"paper_label": "Verbalized 1S top-1", "figure": "B", "factory": None,
     "reason": "Stesso motivo di Verbalized 1S top-k (legge la confidenza dalla generazione condivisa)."},
    {"paper_label": "Verbalized 2S top-k", "figure": "B", "factory": None,
     "reason": "Richiede come 'prima risposta' un campionamento top-k; la nostra risposta condivisa e' "
               "sempre greedy/deterministica, non produciamo varianti top-k della risposta principale."},
    {"paper_label": "Verbalized 2S CoT", "figure": "B", "factory": None,
     "reason": "Richiede una risposta con ragionamento chain-of-thought come primo turno; incompatibile "
               "col prompt attuale, che vieta esplicitamente testo oltre alla lettera."},
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
    n_alias = sum(1 for m in PAPER_METHODS if m["factory"] == "alias:Semantic Entropy" or (isinstance(m["factory"], str) and m["factory"].startswith("alias:")))
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


def build_manager(model, dataset, estimators, cache_dir, max_rejection, max_new_tokens):
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
        generation_metrics=[AccuracyMetric(output_ignore_regex=r"(?<=[ABCDabcd])[\s\S]*")],
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


def run_smoke_test(x_prompts, y_references, cache_dir, hf_token, max_rejection, max_new_tokens):
    print("\n--- SMOKE TEST: 25 esempi, un solo modello (LFM2-350M) ---")
    smoke_dataset = PolygraphDataset(x_prompts[:25], y_references[:25], batch_size=4)
    smoke_model = load_whitebox_model(MODELS["LFM2-350M"], cache_dir, hf_token=hf_token)
    smoke_estimators = build_estimators()
    smoke_man = build_manager(smoke_model, smoke_dataset, smoke_estimators, cache_dir, max_rejection, max_new_tokens)
    smoke_man()
    print(type(smoke_man.metrics))
    print(smoke_man.metrics)
    for i in range(8):
        inputs = smoke_model.tokenizer(x_prompts[i], return_tensors="pt").to(smoke_model.model.device)
        out = smoke_model.model.generate(**inputs, max_new_tokens=5, do_sample=False)
        generated = smoke_model.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Generato: {generated!r} | Riferimento: {y_references[i]!r}")
    del smoke_model, smoke_man
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="MedQA UQ benchmark (lm-polygraph)")
    parser.add_argument("--n_test_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    parser.add_argument("--max_rejection", type=float, default=0.5)
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--results_dir", type=str, default=os.environ.get("RESULTS_DIR", "/workspace/results"))
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HOME", "/llms"))
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

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("\n--- Caricamento dataset: GBaker/MedQA-USMLE-4-options ---")
    raw = load_dataset("GBaker/MedQA-USMLE-4-options")
    print(raw)
    print(raw["test"][0])

    fewshot_block = build_fewshot_block()
    test_split = raw["test"].shuffle(seed=SEED).select(range(args.n_test_samples))
    x_prompts = [format_prompt(ex, fewshot_block) for ex in test_split]
    y_references = [ex["answer_idx"] for ex in test_split]
    print(x_prompts[0])
    print("Riferimento:", y_references[0])

    if args.smoke_test:
        run_smoke_test(x_prompts, y_references, args.cache_dir, hf_token, args.max_rejection, args.max_new_tokens)

    final_path = os.path.join(args.results_dir, "results_final.csv")
    results_df_existing = None
    already_done = set()

    if os.path.exists(final_path):
        try:
            results_df_existing = pd.read_csv(final_path)
            already_done = set(results_df_existing["model"].unique())
            print("Modelli gia' completati (da results_final.csv):", already_done)
        except Exception as e:
            print(f"results_final.csv presente ma illeggibile ({e}) -- riparto senza skip.")
            results_df_existing = None
    else:
        print("Nessun risultato precedente trovato -- primo avvio, eseguo tutti i modelli.")

    models_to_run = {k: v for k, v in MODELS.items() if k not in already_done}
    print("Modelli da eseguire in questo run:", list(models_to_run.keys()))

    write_excluded_methods_report(args.results_dir)

    # Nome-interno (str(estimator)) -> etichetta del paper, per rimappare i
    # risultati alle Figure 2/3 / Tabella 6 senza toccare extract_prr_table.
    paper_label_by_str = {
        str(m["factory"]()): m["paper_label"] for m in PAPER_METHODS if callable(m["factory"])
    }

    all_metrics_dfs = [results_df_existing] if results_df_existing is not None else []

    timing_final_path = os.path.join(args.results_dir, "estimator_timings.csv")
    all_timing_dfs = []
    if os.path.exists(timing_final_path):
        try:
            all_timing_dfs = [pd.read_csv(timing_final_path)]
        except Exception as e:
            print(f"estimator_timings.csv presente ma illeggibile ({e}) -- riparto senza skip.")

    for model_name, model_id in models_to_run.items():
        print(f"\n=== Modello: {model_name} ({model_id}) ===")
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            use_quant = model_name not in CHAT_TEMPLATE_MODELS
            model = load_whitebox_model(
                model_id, args.cache_dir, hf_token=hf_token, use_quantization=use_quant,
            )
            log_gpu_mem(f"{model_name} loaded")

            if model_name in CHAT_TEMPLATE_MODELS:
                model_x_prompts = [format_chat_prompt(model.tokenizer, ex, fewshot_block) for ex in test_split]
            else:
                model_x_prompts = x_prompts
            model_dataset = PolygraphDataset(model_x_prompts, y_references, batch_size=args.batch_size)

            timing_dict = {}
            estimators = [TimedEstimator(e, timing_dict) for e in build_estimators()]
            man = build_manager(model, model_dataset, estimators, args.cache_dir, args.max_rejection, args.max_new_tokens)
            man()
            log_gpu_mem(f"{model_name} done")

            df = extract_prr_table(man, model_name)
            n_ok = df["value"].notna().sum() if "value" in df.columns else 0
            print(f"{model_name}: {n_ok}/{len(df)} righe metrica con valore.")
            all_metrics_dfs.append(df)

            timing_rows = [
                {
                    "model": model_name,
                    "estimator": est_str,
                    "paper_label": paper_label_by_str.get(est_str, est_str),
                    "seconds": seconds,
                }
                for est_str, seconds in timing_dict.items()
            ]
            all_timing_dfs.append(pd.DataFrame(timing_rows))

            del model, man
        except Exception:
            print(f"!!! {model_name} fallito, salto al prossimo modello.")
            traceback.print_exc()
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            elapsed = time.time() - start
            print(f"Tempo {model_name}: {elapsed:.1f}s")

        if all_metrics_dfs:
            pd.concat(all_metrics_dfs, ignore_index=True).to_csv(
                os.path.join(args.results_dir, "results_partial.csv"), index=False
            )
            print(f"Checkpoint salvato dopo {model_name}.")
        if all_timing_dfs:
            pd.concat(all_timing_dfs, ignore_index=True).to_csv(
                os.path.join(args.results_dir, "estimator_timings_partial.csv"), index=False
            )

    if not all_metrics_dfs:
        print("Nessun modello completato con successo -- niente da salvare/plottare.")
        return

    results_df = pd.concat(all_metrics_dfs, ignore_index=True)
    results_df.to_csv(final_path, index=False)
    print(results_df.head(20))

    if all_timing_dfs:
        timing_df = pd.concat(all_timing_dfs, ignore_index=True)
        timing_df.to_csv(timing_final_path, index=False)

    raw_df = results_df[results_df["ue_metric"] == "prr_0.5"] if "ue_metric" in results_df.columns else results_df
    raw_df = raw_df.copy()
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

    figure_a_labels = [m["paper_label"] for m in PAPER_METHODS if m["figure"] in ("A", "AB") and m["factory"] is not None]
    figure_b_labels = [m["paper_label"] for m in PAPER_METHODS if m["figure"] in ("B", "AB") and m["factory"] is not None]
    model_order = [m for m in MODELS.keys() if m in raw_df["model"].unique()]

    def plot_paper_figure(labels, title, note, out_name):
        subset = raw_df[raw_df["paper_label"].isin(labels)]
        if subset.empty:
            print(f"!!! Nessun dato per {out_name}, salto.")
            return
        pivot = subset.pivot_table(index="paper_label", columns="model", values="value", aggfunc="first")
        pivot = pivot.reindex(columns=model_order)
        pivot["Mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("Mean", ascending=True)  # barh: prima riga in cima = ultima disegnata
        # Altezza minima piu' alta (8" invece di 6") cosi' anche i grafici con
        # poche righe (es. Figura B, 12 metodi) hanno spazio a sufficienza in
        # basso: prima la nota a piè di figura si sovrapponeva all'xlabel sui
        # grafici bassi (bottom margin insufficiente per pochi metodi).
        fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.4)))
        pivot.plot(kind="barh", ax=ax, width=0.8)
        ax.set_xlabel("Mean PRR (raw, max_rejection=0.5)")
        ax.set_title(title)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.legend(loc="lower right", fontsize=8)
        # Margine inferiore fisso (non dipendente da tight_layout) per dare
        # alla nota il suo spazio dedicato sotto l'xlabel, senza sovrapporsi.
        fig.subplots_adjust(bottom=0.16)
        wrapped_note = "\n".join(textwrap.wrap(note, width=115))
        fig.text(0.01, 0.02, wrapped_note, fontsize=7, va="bottom")
        out_path = os.path.join(args.results_dir, out_name)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Salvato: {out_path}")

    try:
        plot_paper_figure(
            figure_a_labels,
            "PRR per metodo UQ e modello (~ Fig. 2 Vashurin et al., white-box full-access) — MedQA-USMLE",
            FIGURE_A_MODELS_NOTE,
            "fig_a_white_box.png",
        )
    except Exception:
        print("!!! fig_a_white_box.png fallito:")
        traceback.print_exc()

    try:
        plot_paper_figure(
            figure_b_labels,
            "PRR per metodo UQ e modello (~ Fig. 3 Vashurin et al., reflexive/black-box) — MedQA-USMLE",
            FIGURE_B_MODELS_NOTE,
            "fig_b_reflexive.png",
        )
    except Exception:
        print("!!! fig_b_reflexive.png fallito:")
        traceback.print_exc()

    # Tabella stile Tabella 6 del paper: stesso set di metodi della Figura A,
    # ma con i nostri 4 modelli al posto delle 4 dataset (CoQA/TriviaQA/
    # MMLU/GSM8k) -- loro tenevano il modello fisso e variavano il dataset,
    # noi teniamo il dataset fisso (MedQA) e variamo il modello.
    try:
        subset = raw_df[raw_df["paper_label"].isin(figure_a_labels)]
        table6 = subset.pivot_table(index="paper_label", columns="model", values="value", aggfunc="first")
        table6 = table6.reindex(columns=model_order)
        ranks = table6.rank(axis=0, ascending=False)
        table6["Mean Rank"] = ranks.mean(axis=1)
        table6["Mean PRR"] = table6[model_order].mean(axis=1)
        table6 = table6.sort_values("Mean PRR", ascending=False)
        table6_path = os.path.join(args.results_dir, "table6_style.csv")
        table6.to_csv(table6_path)
        print(f"Salvato: {table6_path}")
    except Exception:
        print("!!! table6_style.csv fallito:")
        traceback.print_exc()

    # Tabella di efficienza: tempo (secondi) speso da ciascun metodo UQ per
    # modello, per verificare quali stimatori sono piu' costosi.
    try:
        if all_timing_dfs:
            timing_all = pd.concat(all_timing_dfs, ignore_index=True).drop_duplicates(
                subset=["model", "estimator"], keep="last"
            )
            timing_pivot = timing_all.pivot_table(
                index="paper_label", columns="model", values="seconds", aggfunc="first"
            )
            timing_pivot = timing_pivot.reindex(columns=model_order)
            timing_pivot["Total"] = timing_pivot.sum(axis=1)
            timing_pivot = timing_pivot.sort_values("Total", ascending=False)
            timing_path = os.path.join(args.results_dir, "estimator_timing_table.csv")
            timing_pivot.to_csv(timing_path)
            print(f"Salvato: {timing_path}")

            # Stesso identico grafico ma in forma leggibile a colpo d'occhio.
            # Scala log sull'asse x: i tempi coprono ~5-6 ordini di
            # grandezza (es. BB P(True) ~270s contro P(True) ~0.001s), in
            # scala lineare tutte le barre tranne una sarebbero invisibili.
            plot_pivot = timing_pivot.drop(columns=["Total"]).loc[
                timing_pivot.sort_values("Total", ascending=True).index
            ]
            fig, ax = plt.subplots(figsize=(10, max(8, len(plot_pivot) * 0.4)))
            plot_pivot.plot(kind="barh", ax=ax, width=0.8, logx=True)
            ax.set_xlabel("Tempo totale di calcolo, sommato su tutti i batch (secondi, scala log)")
            ax.set_title("Tempo di calcolo per metodo UQ e modello — MedQA-USMLE")
            ax.legend(loc="lower right", fontsize=8)
            fig.subplots_adjust(bottom=0.12)
            plt.tight_layout()
            timing_chart_path = os.path.join(args.results_dir, "estimator_timing_chart.png")
            plt.savefig(timing_chart_path, dpi=150)
            plt.close(fig)
            print(f"Salvato: {timing_chart_path}")
    except Exception:
        print("!!! estimator_timing_table.csv fallito:")
        traceback.print_exc()

    print("\nBENCHMARK COMPLETATO.")


if __name__ == "__main__":
    main()
