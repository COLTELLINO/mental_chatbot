"""Replica di Vashurin et al. (TACL 2025), selective QA: Tabelle 7 e 10.

Serve a verificare la pipeline sul setting esatto del paper, prima di usarla
sui modelli piccoli. Il paper usa due configurazioni diverse (Sezione 5.1):

- WHITE-BOX (Tabella 7): Mistral 7B v0.2 *base*, senza instruction tuning,
  prompt "a completamento" (5-shot per TriviaQA, MMLU e GSM8k; per CoQA la
  storia e tutte le domande precedenti della conversazione), tutti i metodi
  white-box. Qui: `--parts whitebox`.
- BLACK-BOX (Tabella 10): Mistral 7B v0.2 *Instruct*, solo CoQA, TriviaQA e
  MMLU. I metodi basati sui campioni usano i prompt "empirical baselines" di
  Tian et al. (2023) (`--parts blackbox`); ognuno dei 6 metodi verbalized ha
  il proprio prompt (`--parts verbalized`, una cella per variante).

Tutto cio' che definisce l'esperimento viene dal protocollo ufficiale, non da
una nostra riscrittura:
- i prompt sono quelli dei dataset pubblicati da lm-polygraph su Hugging Face
  (LM-Polygraph/{coqa,triviaqa,mmlu,gsm8k}, sottoinsiemi "continuation",
  "empirical_baselines", "verb_1s_top1", ...), costruiti dai builder della
  libreria;
- max_new_tokens, stringhe di arresto, stimatori e loro parametri, funzioni di
  post-elaborazione delle risposte vengono dai file di configurazione della
  libreria (examples/configs/polygraph_eval_*.yaml e instruct/*.yaml al commit
  32cdf4a), risolti una volta in paper_replica_configs.json;
- numerosita' come nel paper: "limitiamo il set di valutazione a 2.000 istanze,
  tranne MMLU, dove teniamo 100 domande per materia" (5.700, gia' cosi' nel
  dataset pubblicato). Le 2.000 sono scelte come fa la libreria
  (Dataset.subsample: np.random.seed(1) + np.random.choice senza reimmissione),
  quindi con ogni probabilita' sono le stesse domande del paper;
- qualita': AlignScore per CoQA e TriviaQA (massimo sugli alias per TriviaQA),
  accuracy per MMLU e GSM8k, dopo la stessa post-elaborazione del protocollo.

Differenze note, dichiarate:
- precisione bf16 (il paper non la dichiara; e' quella nativa del modello);
- campionatore dei K campioni in batch (batched_sampling.py): stesse
  statistiche della libreria, verificato da tests/test_batched_sampling.py;
- i metodi density-based (Mahalanobis, RDE, RMD, HUQ-MD) sono esclusi per
  design in tutto il lavoro (vedi PAPER_METHODS in main.py).

Uso (nel container, dalla cartella del repo):
    python3.11 paper_replica.py --parts whitebox
    python3.11 paper_replica.py --parts blackbox verbalized
    python3.11 paper_replica.py --parts whitebox --datasets GSM8k
Con sbatch:  bash sbatch_script.sh paper_replica.py --parts whitebox
I risultati vanno in --results_dir (default /workspace/results_paper_replica),
separati da quelli dei modelli piccoli. Alla fine viene scritto
paper_replica_comparison.csv (confronto metodo per metodo con le Tabelle 7 e
10) e make_figures.py disegna fig_paper_replica_*.png.
"""
import argparse
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

import main as M
import analysis_lib as AL
import paper_replica_processing as PP
from dataset_prep import ALL_AVAILABLE, MaxOverReferences
from lm_polygraph import estimators as LPE
from lm_polygraph.generation_metrics import AlignScore, AccuracyMetric
from lm_polygraph.generation_metrics.preprocess_output_target import PreprocessOutputTarget

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS_FILE = os.path.join(HERE, "paper_replica_configs.json")
REFERENCE_FILE = os.path.join(HERE, "paper_reference_prr.csv")

# Numerosita' e selezione delle domande (vedi il docstring).
PAPER_MAX_INSTANCES = 2000
PAPER_NO_SUBSAMPLE = {"MMLU"}          # 100 per materia, gia' nel dataset pubblicato
PAPER_SUBSAMPLE_SEED = 1               # seed dei file di configurazione ufficiali

# Il paper usa AlignScore per CoQA e TriviaQA, accuracy per MMLU e GSM8k.
PAPER_QUALITY = {"CoQA": "AlignScore", "TriviaQA": "AlignScore", "MMLU": "Accuracy",
                 "GSM8k": "Accuracy"}

# Modelli. La v0.2 base non e' pubblicata da mistralai su Hugging Face: l'unica
# copia e' la conversione dei pesi originali in mistral-community (non gated).
WHITEBOX_MODEL = ("Mistral-7B-v0.2-base", "mistral-community/Mistral-7B-v0.2")
BLACKBOX_MODEL = ("Mistral-7B-v0.2-it", "mistralai/Mistral-7B-Instruct-v0.2")

BLACKBOX_SUBSETS = ["empirical_baselines"]
VERBALIZED_SUBSETS = ["ling_1s", "verb_1s_top1", "verb_1s_topk", "verb_2s_cot",
                      "verb_2s_top1", "verb_2s_topk"]

# Nome interno dello stimatore -> etichetta delle Tabelle 7 e 10, per gli
# stimatori che non compaiono in PAPER_METHODS (main.py).
EXTRA_PAPER_LABELS = {
    "SemanticEntropyEmpirical": "BB Semantic Entropy",
    "PTrueEmpirical": "BB P(True)",
    "LabelProb": "Label Prob.",
    "Linguistic1S": "Linguistic 1S",
    "Verbalized1S_top1": "Verbalized 1S top-1",
    "Verbalized1S_topk": "Verbalized 1S top-k",
    "Verbalized2S_top1": "Verbalized 2S top-1",
    "Verbalized2S_topk": "Verbalized 2S top-k",
    "Verbalized2S_cot": "Verbalized 2S CoT",
}

# Nomi del paper che in PAPER_METHODS hanno una grafia diversa.
PAPER_LABEL_ALIASES = {"Fisher-Rao Distance": "Fisher-Rao", "Rényi Divergence": "Renyi Divergence"}


def load_configs():
    with open(CONFIGS_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dati
# ---------------------------------------------------------------------------

def paper_indices(n_rows, dataset_name):
    """Indici delle domande valutate, scelti come Dataset.subsample di
    lm-polygraph (stesso seed, stessa funzione, stesso ordine)."""
    if dataset_name in PAPER_NO_SUBSAMPLE or n_rows <= PAPER_MAX_INSTANCES:
        return list(range(n_rows))
    np.random.seed(PAPER_SUBSAMPLE_SEED)
    return [int(i) for i in np.random.choice(n_rows, PAPER_MAX_INSTANCES, replace=False)]


def make_loader(entry, dataset_name):
    """Loader nel formato di main.py: loader(n_test, seed, cache_dir) ->
    [{"content", "reference", "paper_index"}]. `n_test` minore della
    numerosita' del paper prende le prime n_test domande della selezione (per
    le prove brevi); il seed passato da main.py viene ignorato, perche' la
    selezione e' quella del paper."""
    def load(n_test, seed=None, cache_dir=None):
        repo = entry["hf_dataset"].replace("LM-polygraph/", "LM-Polygraph/")
        d = load_dataset(repo, entry["subset"], split=entry["split"], cache_dir=cache_dir)
        # Il campo `size` dei file di configurazione NON viene passato da
        # scripts/polygraph_eval a Dataset.load per il set di valutazione: si
        # parte sempre dall'intero split di test (per TriviaQA 17.944 domande,
        # non le prime 10.000), e poi si estraggono le 2.000.
        idx = paper_indices(len(d), dataset_name)
        if n_test is not None and n_test < len(idx):
            idx = idx[:n_test]
        inputs, outputs = d["input"], d["output"]
        return [{"content": inputs[i], "reference": outputs[i], "paper_index": i} for i in idx]
    return load


def make_metric_factory(entry, dataset_name):
    """Metrica di qualita' del paper per il dataset, con la stessa
    post-elaborazione del protocollo (PreprocessOutputTarget) e, per TriviaQA,
    il massimo sugli alias (multiref)."""
    def factory():
        if PAPER_QUALITY[dataset_name] == "AlignScore":
            metric = AlignScore(target_is_claims=True)
        else:
            metric = AccuracyMetric(target_ignore_regex=entry.get("target_ignore_regex"),
                                    output_ignore_regex=entry.get("output_ignore_regex"),
                                    normalize=bool(entry.get("normalize")))
        fo, ft = entry.get("process_output_fn"), entry.get("process_target_fn")
        if fo or ft:
            metric = PreprocessOutputTarget(metric,
                                            getattr(PP, fo) if fo else (lambda x: x),
                                            getattr(PP, ft) if ft else (lambda x: x))
        if entry.get("multiref"):
            metric = MaxOverReferences(metric)
        return metric
    return factory


def datasets_cfg_for(entries, wanted=None):
    cfg = {}
    for name, entry in entries.items():
        if wanted and name not in wanted:
            continue
        cfg[name] = {
            "loader": make_loader(entry, name),
            "n_test": ALL_AVAILABLE,
            "max_new_tokens": entry["max_new_tokens"],
            "plain_suffix": "",
            "stop_strings": entry.get("stop_strings"),
            "generation_metric_factory": make_metric_factory(entry, name),
        }
    return cfg


# ---------------------------------------------------------------------------
# Stimatori
# ---------------------------------------------------------------------------

def whitebox_estimators():
    """I metodi della Tabella 7 (white-box) disponibili in PAPER_METHODS."""
    return [m["factory"]() for m in M.PAPER_METHODS
            if callable(m["factory"]) and m["figure"] in ("A", "AB")]


def estimators_from_spec(spec):
    """Istanzia gli stimatori come fa lm-polygraph dai file di
    configurazione: classe per nome, parametri dal campo cfg."""
    out = []
    for e in spec:
        cls = getattr(LPE, e["name"])
        out.append(cls(**(e.get("cfg") or {})))
    return out


def subset_estimators_factory(configs, subset):
    entries = configs["blackbox"][subset]
    specs = [configs["default_blackbox_estimators"] if v["estimators"] == "default_blackbox_estimators"
             else v["estimators"] for v in entries.values()]
    # Gli stimatori (e i loro parametri) sono gli stessi per tutti i dataset
    # di una variante: cambia solo max_new_tokens, che sta nel dataset.
    if any(json.dumps(s, sort_keys=True) != json.dumps(specs[0], sort_keys=True) for s in specs):
        raise ValueError(f"{subset}: stimatori diversi fra i dataset, serve una sezione per dataset.")
    return lambda: estimators_from_spec(specs[0])


def paper_label_map():
    labels = {str(m["factory"]()): m["paper_label"] for m in M.PAPER_METHODS if callable(m["factory"])}
    labels.update(EXTRA_PAPER_LABELS)
    return labels


# ---------------------------------------------------------------------------
# Confronto con il paper
# ---------------------------------------------------------------------------

def load_reference():
    ref = pd.read_csv(REFERENCE_FILE)
    ref["paper_label"] = ref["paper_label"].replace(PAPER_LABEL_ALIASES)
    return ref


def compare_with_paper(results_dir):
    """paper_replica_comparison.csv: per ogni (setting, dataset, metodo) il
    PRR ottenuto con intervallo al 95% e quello del paper; e un riepilogo per
    (setting, dataset) con differenza media assoluta e tau di Kendall fra le
    due classifiche dei metodi."""
    ref = load_reference()
    rows = []
    for setting, model_name, pattern in (("whitebox", WHITEBOX_MODEL[0], "results_paper_whitebox"),
                                         ("blackbox", BLACKBOX_MODEL[0], "results_paper_blackbox_")):
        files = sorted(f for f in os.listdir(results_dir)
                       if f.startswith(pattern) and f.endswith("_instance_stats.csv"))
        for f in files:
            st = pd.read_csv(os.path.join(results_dir, f))
            st = st[st["model"] == model_name]
            if st.empty:
                continue
            subset = f[len("results_paper_"):-len("_instance_stats.csv")]
            for _, r in st.iterrows():
                rows.append({"setting": setting, "subset": subset, "dataset": r["dataset"],
                             "estimator": r["estimator"], "paper_label": r.get("paper_label"),
                             "prr": r["prr"], "prr_ci_low": r.get("prr_ci_low"),
                             "prr_ci_high": r.get("prr_ci_high"),
                             "mean_quality": r.get("mean_quality"), "n_instances": r.get("n_instances")})
    if not rows:
        print("Nessun risultato della replica da confrontare.")
        return None
    ours = pd.DataFrame(rows)
    ours = ours[ours["paper_label"].notna()]
    merged = ours.merge(ref[["setting", "dataset", "paper_label", "prr", "prr_std"]]
                        .rename(columns={"prr": "prr_paper", "prr_std": "prr_paper_std"}),
                        on=["setting", "dataset", "paper_label"], how="left")
    merged["diff"] = merged["prr"] - merged["prr_paper"]
    merged["paper_in_ci"] = ((merged["prr_paper"] >= merged["prr_ci_low"])
                             & (merged["prr_paper"] <= merged["prr_ci_high"]))
    merged.to_csv(os.path.join(results_dir, "paper_replica_comparison.csv"), index=False)

    summary = []
    for (setting, ds), g in merged.dropna(subset=["prr_paper"]).groupby(["setting", "dataset"]):
        tau = AL._kendall(g["prr"].to_numpy(), g["prr_paper"].to_numpy()) if len(g) >= 3 else np.nan
        summary.append({"setting": setting, "dataset": ds, "n_methods": len(g),
                        "mean_abs_diff": float(g["diff"].abs().mean()),
                        "max_abs_diff": float(g["diff"].abs().max()),
                        "share_paper_in_ci": float(g["paper_in_ci"].mean()),
                        "kendall_tau_vs_paper": tau,
                        "mean_quality": float(g["mean_quality"].dropna().iloc[0])
                        if g["mean_quality"].notna().any() else np.nan})
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(results_dir, "paper_replica_summary.csv"), index=False)
    print("\n--- Replica contro il paper ---")
    print(summary.round(3).to_string(index=False))
    return merged


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="+", default=["whitebox", "blackbox", "verbalized"],
                        choices=["whitebox", "blackbox", "verbalized"],
                        help="Parti della replica da eseguire (default: tutte).")
    parser.add_argument("--datasets", nargs="+", default=None, choices=list(PAPER_QUALITY),
                        help="Solo questi dataset (GSM8k esiste solo nella parte white-box).")
    parser.add_argument("--verbalized_variants", nargs="+", default=VERBALIZED_SUBSETS,
                        choices=VERBALIZED_SUBSETS)
    parser.add_argument("--n_test_samples", type=int, default=None,
                        help="Prime N domande della selezione del paper, per le prove brevi.")
    parser.add_argument("--precision", choices=["bf16", "4bit"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--max_rejection", type=float, default=0.5)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--results_dir", type=str,
                        default=os.environ.get("PAPER_RESULTS_DIR", "/workspace/results_paper_replica"))
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HOME", "/llms"))
    parser.add_argument("--datasets_cache_dir", type=str,
                        default=os.environ.get("HF_DATASETS_CACHE", "/workspace/hf_datasets_cache"))
    parser.add_argument("--sampler", choices=["batched", "library"], default="batched")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--force_resume", action="store_true")
    parser.add_argument("--only_compare", action="store_true",
                        help="Non esegue nulla: rifa solo il confronto con il paper e le figure.")
    args = parser.parse_args()

    if args.only_compare:
        compare_with_paper(args.results_dir)
        return

    # Impostazioni condivise con main.py.
    M.SAMPLER = args.sampler
    args.anchor_precision = args.precision
    args.verbalized_max_new_tokens = None
    M.FINGERPRINT_FILES = tuple(M.FINGERPRINT_FILES) + (
        "paper_replica.py", "paper_replica_configs.json", "paper_replica_processing.py")
    for name, _ in (WHITEBOX_MODEL, BLACKBOX_MODEL):
        M.MODEL_PARAMS_B[name] = 7.0
        if args.precision == "bf16":
            M.NO_QUANT_MODELS.add(name)
    M.LIBRARY_CHAT_TEMPLATE_MODELS.add(BLACKBOX_MODEL[0])

    for d in (args.results_dir, args.cache_dir, args.datasets_cache_dir):
        os.makedirs(d, exist_ok=True)
    fp = M.check_run_fingerprint(args)
    args.run_fingerprint = fp["fingerprint"]
    print(f"Replica di Vashurin et al. -- parti {args.parts}, precisione {args.precision}, "
          f"impronta {fp['fingerprint']}")
    np.random.seed(M.SEED)
    torch.manual_seed(M.SEED)

    configs = load_configs()
    labels = paper_label_map()
    hf_token = os.environ.get("HF_TOKEN")

    if "whitebox" in args.parts:
        cfg = datasets_cfg_for(configs["whitebox"], args.datasets)
        if cfg:
            M.run_dataset_section(
                "Replica white-box (Tabella 7): Mistral 7B v0.2 base, prompt a completamento",
                cfg, args, hf_token, labels, "results_paper_whitebox",
                models={WHITEBOX_MODEL[0]: WHITEBOX_MODEL[1]},
                estimators_factory=whitebox_estimators)

    subsets = ((BLACKBOX_SUBSETS if "blackbox" in args.parts else [])
               + (list(args.verbalized_variants) if "verbalized" in args.parts else []))
    for subset in subsets:
        cfg = datasets_cfg_for(configs["blackbox"][subset], args.datasets)
        if not cfg:
            continue
        try:
            M.run_dataset_section(
                f"Replica black-box (Tabella 10): Mistral 7B v0.2 Instruct, prompt {subset}",
                cfg, args, hf_token, labels, f"results_paper_blackbox_{subset}",
                models={BLACKBOX_MODEL[0]: BLACKBOX_MODEL[1]},
                estimators_factory=subset_estimators_factory(configs, subset))
        except Exception:
            print(f"!!! Variante {subset} fallita:")
            traceback.print_exc()

    try:
        compare_with_paper(args.results_dir)
    except Exception:
        print("!!! Confronto con il paper fallito:")
        traceback.print_exc()
    cmd = [sys.executable, os.path.join(HERE, "make_figures.py"), args.results_dir]
    print(f"\n--- {' '.join(cmd[1:])} ---")
    subprocess.run(cmd, check=False)
    print("\nREPLICA COMPLETATA.")


if __name__ == "__main__":
    main()
