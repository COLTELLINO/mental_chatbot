"""
MedQA UQ Benchmark — ported from MedQA_UQ_Benchmark.ipynb.

Benchmarks lm-polygraph's uncertainty-quantification estimators against
several small/medium LLMs on MedQA-USMLE-4-options. On Colab (T4, 15GB) many
of the multi-sample/similarity-based estimators (Monte Carlo sequence entropy,
lexical similarity, eigenvalue/degree-matrix graph methods, semantic entropy,
SAR variants, LUQ, kernel language entropy, eigenscore, semantic density...)
ran out of VRAM. This runs on a 24GB RTX 3090, hoping those now fit.

Ported 1:1 from the notebook — same models, same estimator list, same
experimental loop and resume/checkpoint logic. Only the environment glue
changed (no Colab, no Drive, no notebook magics):
  - HF_TOKEN comes from the environment instead of google.colab.userdata
  - RESULTS_DIR/CACHE_DIR point into the bind-mounted /workspace and the
    cluster's shared /llms cache instead of Google Drive / /content
  - matplotlib uses the Agg backend (no display) and only saves figures
  - added: an environment/GPU banner and per-model timing + VRAM stats,
    since this is the first real run on this infra and OOM boundaries are
    exactly what we're trying to find

lm-polygraph's UEManager(ignore_exceptions=True) already catches per-estimator
failures internally (that's how the notebook silently "skipped" the ~29
failing methods on Colab instead of crashing) — that resilience is unchanged
here. The outer per-model try/except is an extra safety net for anything
that escapes the manager (e.g. OOM during model loading itself).
"""
import argparse
import gc
import os
import sys
import time
import traceback

# Must be set before `import torch` to take effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")  # headless container, no display
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
from lm_polygraph.estimators import *  # noqa: F401,F403 (same as notebook)

SEED = 3407

MODELS = {
    "LFM2-350M":      "LiquidAI/LFM2-350M",
    "LFM2-1.2B":      "LiquidAI/LFM2-1.2B",
    "MedGemma-4B-it": "google/medgemma-4b-it",
    "Gemma3-4B-it":   "google/gemma-3-4b-it",
}

GATED_MODELS = {"MedGemma-4B-it", "Gemma3-4B-it"}

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


def format_prompt(example, fewshot_block):
    opts = example["options"]
    return (
        "Sei un assistente medico che risponde a domande in stile esame USMLE.\n"
        "Leggi il quesito clinico e rispondi SOLO con la lettera dell'opzione corretta (A, B, C o D), senza altro testo.\n\n"
        f"{fewshot_block}\n\n"
        f"Domanda: {example['question'].strip()}\n\n"
        f"A) {opts['A'].strip()}\n"
        f"B) {opts['B'].strip()}\n"
        f"C) {opts['C'].strip()}\n"
        f"D) {opts['D'].strip()}\n\n"
        "Risposta:"
    )


def build_estimators():
    estimators = [
        MaximumSequenceProbability(),
        Perplexity(),
        MeanTokenEntropy(),
        MeanPointwiseMutualInformation(),
        SelfCertainty(),
        MeanConditionalPointwiseMutualInformation(),
        ClaimConditionedProbability(),
        PTrue(),
        PTrueSampling(),
        MonteCarloSequenceEntropy(),
        MonteCarloNormalizedSequenceEntropy(),
        LexicalSimilarity(metric="rouge1"),
        LexicalSimilarity(metric="rouge2"),
        LexicalSimilarity(metric="rougeL"),
        LexicalSimilarity(metric="BLEU"),
        NumSemSets(),
        EigValLaplacian(similarity_score="NLI_score", affinity="entail"),
        EigValLaplacian(similarity_score="NLI_score", affinity="contra"),
        EigValLaplacian(similarity_score="Jaccard_score"),
        DegMat(similarity_score="NLI_score", affinity="entail"),
        DegMat(similarity_score="NLI_score", affinity="contra"),
        DegMat(similarity_score="Jaccard_score"),
        Eccentricity(similarity_score="NLI_score", affinity="entail"),
        Eccentricity(similarity_score="NLI_score", affinity="contra"),
        Eccentricity(similarity_score="Jaccard_score"),
        SemanticEntropy(),
        SemanticEntropy(entropy_estimation="direct"),
        SAR(),
        TokenSAR(),
        SentenceSAR(),
        LUQ(),
        KernelLanguageEntropy(),
        EigenScore(),
        RenyiNeg(),
        FisherRao(),
        CSL(),
        CocoaMSP(),
        CocoaPPL(),
        CocoaMTE(),
        SemanticDensity(),
        BoostedProbSequence(),
    ]
    print(f"Totale stimatori: {len(estimators)}")
    return estimators


def load_whitebox_model(model_id, cache_dir, hf_token=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        cache_dir=cache_dir,
        attn_implementation="eager",
    )
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
    """Converte man.metrics in un DataFrame lungo: model, method, generation_metric, PRR.
    man.metrics e' tipicamente un dict con chiavi (livello, estimator_name, generation_metric_name,
    ue_metric_name) -> valore. Verificare/adattare in base a quanto osservato nello smoke test
    se la struttura reale differisce."""
    rows = []
    for key, value in man.metrics.items():
        try:
            *rest, ue_metric_name = key
            rows.append({
                "model": model_name,
                "key": key,
                "ue_metric": ue_metric_name,
                "value": value,
            })
        except TypeError:
            rows.append({"model": model_name, "key": str(key), "ue_metric": None, "value": value})
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
        print("!! ATTENZIONE: HF_TOKEN non impostato ma MODELS include modelli gated "
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

    all_metrics_dfs = [results_df_existing] if results_df_existing is not None else []
    full_dataset = PolygraphDataset(x_prompts, y_references, batch_size=args.batch_size)

    for model_name, model_id in models_to_run.items():
        print(f"\n=== Modello: {model_name} ({model_id}) ===")
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            model = load_whitebox_model(model_id, args.cache_dir, hf_token=hf_token)
            log_gpu_mem(f"{model_name} loaded")
            estimators = build_estimators()
            man = build_manager(model, full_dataset, estimators, args.cache_dir, args.max_rejection, args.max_new_tokens)
            man()
            log_gpu_mem(f"{model_name} done")

            print(f"--- Debug generazioni: {model_name} ---")
            try:
                print(f"  man.stats keys: {list(man.stats.keys())}")
                preds = man.stats.get("greedy_texts")
                n_show = min(10, len(preds) if preds is not None else 0, len(y_references))
                for i in range(n_show):
                    print(f"  [{i}] pred={preds[i]!r} true={y_references[i]!r}")
            except Exception:
                print("  (impossibile leggere man.stats per debug)")
                traceback.print_exc()

            df = extract_prr_table(man, model_name)
            n_ok = df["value"].notna().sum() if "value" in df.columns else 0
            print(f"{model_name}: {n_ok}/{len(df)} righe metrica con valore.")
            all_metrics_dfs.append(df)

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

    if not all_metrics_dfs:
        print("Nessun modello completato con successo -- niente da salvare/plottare.")
        return

    results_df = pd.concat(all_metrics_dfs, ignore_index=True)
    results_df.to_csv(final_path, index=False)
    print(results_df.head(20))

    try:
        pivot = results_df.pivot_table(index="key", columns="model", values="value", aggfunc="first")
        plt.figure(figsize=(10, 12))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0, cbar_kws={"label": "PRR"})
        plt.title("Prediction-Rejection Ratio per metodo UQ e modello — MedQA-USMLE")
        plt.tight_layout()
        heatmap_path = os.path.join(args.results_dir, "prr_heatmap.png")
        plt.savefig(heatmap_path, dpi=150)
        print(f"Heatmap salvata: {heatmap_path}")
    except Exception:
        print("!!! Heatmap fallita:")
        traceback.print_exc()

    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        compare_models = ["MedGemma-4B-it", "Gemma3-4B-it"]
        subset = results_df[results_df["model"].isin(compare_models)]
        subset_pivot = subset.pivot_table(index="key", columns="model", values="value", aggfunc="first")
        subset_pivot.plot(kind="barh", ax=axes[0])
        axes[0].set_title("Medico (MedGemma) vs generico (Gemma3), stessa taglia")

        compare_sizes = ["LFM2-350M", "LFM2-1.2B"]
        subset2 = results_df[results_df["model"].isin(compare_sizes)]
        subset2_pivot = subset2.pivot_table(index="key", columns="model", values="value", aggfunc="first")
        subset2_pivot.plot(kind="barh", ax=axes[1])
        axes[1].set_title("Effetto scala: LFM2-350M vs LFM2-1.2B")

        plt.tight_layout()
        comparisons_path = os.path.join(args.results_dir, "comparisons.png")
        plt.savefig(comparisons_path, dpi=150)
        print(f"Comparazioni salvate: {comparisons_path}")
    except Exception:
        print("!!! Grafico di comparazione fallito:")
        traceback.print_exc()

    print("\nBENCHMARK COMPLETATO.")


if __name__ == "__main__":
    main()
