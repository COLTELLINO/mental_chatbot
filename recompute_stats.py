"""Ricalcola le statistiche di una cartella di risultati GIA' ESISTENTE con le
metriche corrette il 29/09, senza GPU, dai punteggi per-istanza salvati.

Cosa cambia rispetto ai file prodotti prima del 29/09:
- "prr" e i suoi intervalli diventano il PRR del paper (normalizzato fra caso e
  oracolo), con i pareggi in valore atteso; l'area grezza resta in "prr_raw";
- nuove colonne distinct_fraction, error_rate_top10, error_rate_overall;
- i file *_mapped.csv (letti dalle figure) prendono il PRR normalizzato;
- rank_transfer_kendall_tau.csv diventa per dataset, con intervallo bootstrap.

Cosa NON puo' cambiare: tutto cio' che dipende dalle generazioni (prompt in
italiano, LFM2 senza chat template, doppio BOS, bug della lettera "a" nelle
MCQ, un solo riferimento per TriviaQA). Per quelli serve una run nuova: questo
script serve a rileggere le run vecchie con la metrica giusta, non a renderle
equivalenti a una run nuova.

I file originali vengono copiati una volta sola in <results_dir>/_pre_ricalcolo/.

Uso:
    python3.11 recompute_stats.py <results_dir> [--n_bootstrap 1000]
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

import analysis_lib as AL

SEED = 3407
CORRECTNESS_THRESHOLD = 0.5
META = {"model", "dataset", "instance_index", "quality", "greedy_text"}
ANCHOR = "Mistral-7B-it"

# (file per-istanza, file statistiche, file mapped)
SECTIONS = [
    ("per_instance_scores.csv", "instance_level_stats.csv", "results_paper_mapped.csv"),
    ("results_severity_grid_per_instance.csv", "results_severity_grid_instance_stats.csv",
     "results_severity_grid_mapped.csv"),
    ("results_verbalized_numeric_per_instance.csv", "results_verbalized_numeric_instance_stats.csv",
     "results_verbalized_numeric_mapped.csv"),
    ("results_verbalized_linguistic_per_instance.csv",
     "results_verbalized_linguistic_instance_stats.csv", "results_verbalized_linguistic_mapped.csv"),
    ("results_quant_comparison_per_instance.csv", "results_quant_comparison_instance_stats.csv",
     "results_quant_comparison_mapped.csv"),
]


def backup(results_dir, name):
    src = os.path.join(results_dir, name)
    dst_dir = os.path.join(results_dir, "_pre_ricalcolo")
    dst = os.path.join(dst_dir, name)
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, dst)


def recompute_section(results_dir, per_file, stats_file, mapped_file, n_bootstrap, max_rejection,
                      alias_rows):
    per_path = os.path.join(results_dir, per_file)
    if not os.path.exists(per_path):
        return None, None
    per = pd.read_csv(per_path)
    old_stats_path = os.path.join(results_dir, "_pre_ricalcolo", stats_file)
    if not os.path.exists(old_stats_path):
        old_stats_path = os.path.join(results_dir, stats_file)
    old = pd.read_csv(old_stats_path) if os.path.exists(old_stats_path) else pd.DataFrame()
    carry = [c for c in ("paper_label", "quality_metric", "answer_parse_failure_rate",
                         "n_skipped_instances") if c in old.columns]

    rows = []
    for (model, dataset), cell in per.groupby(["model", "dataset"], sort=False):
        cell = cell.dropna(axis=1, how="all")
        q = cell["quality"].to_numpy(float)
        info = old[(old.get("model") == model) & (old.get("dataset") == dataset)] if len(old) else old
        for est in [c for c in cell.columns if c not in META]:
            ue = cell[est].to_numpy(float)
            lo, hi = AL.bootstrap_prr_ci(ue, q, max_rejection, n_bootstrap, SEED)
            r = {"model": model, "dataset": dataset, "estimator": est,
                 "prr": AL.prr_normalized(ue, q, max_rejection), "prr_ci_low": lo, "prr_ci_high": hi,
                 "prr_raw": AL.prr_raw(ue, q, max_rejection),
                 "mean_quality": float(np.nanmean(q)), "n_instances": len(q),
                 "nan_rate": float(np.isnan(ue).mean()),
                 "distinct_fraction": AL.distinct_fraction(ue),
                 "error_rate_top10": AL.error_rate_most_confident(ue, q, CORRECTNESS_THRESHOLD, 0.10),
                 "error_rate_overall": AL.error_rate_overall(q, CORRECTNESS_THRESHOLD),
                 "silent_failure_rate": AL.silent_failure_rate(ue, q, CORRECTNESS_THRESHOLD, 0.10)}
            src = info[info["estimator"] == est] if len(info) else info
            for c in carry:
                r[c] = src[c].iloc[0] if len(src) else np.nan
            if "paper_label" not in r or pd.isna(r.get("paper_label")):
                r["paper_label"] = est
            rows.append(r)
        print(f"  {model}/{dataset}: {len(q)} istanze ricalcolate")
    stats = pd.DataFrame(rows)
    backup(results_dir, stats_file)
    stats.to_csv(os.path.join(results_dir, stats_file), index=False)

    mapped = stats.rename(columns={"prr": "value"}).copy()
    mapped["ue_metric"] = f"prr_{max_rejection}_normalized"
    if alias_rows is not None and mapped_file == "results_paper_mapped.csv":
        extra = []
        for label, target in alias_rows:
            src = mapped[mapped["paper_label"] == target]
            for _, row in src.iterrows():
                extra.append({**row.to_dict(), "paper_label": label})
        if extra:
            mapped = pd.concat([mapped, pd.DataFrame(extra)], ignore_index=True)
    backup(results_dir, mapped_file)
    mapped.to_csv(os.path.join(results_dir, mapped_file), index=False)
    print(f"Salvati: {stats_file}, {mapped_file}")
    return stats, per


def recompute_kendall(results_dir, stats, per, max_rejection, n_resamples):
    if stats is None or ANCHOR not in set(stats["model"]):
        print("Kendall tau: ancora assente, salto.")
        return
    rows = []
    for dataset, st_d in stats.groupby("dataset"):
        piv = st_d.pivot_table(index="estimator", columns="model", values="prr", aggfunc="first")
        if ANCHOR not in piv.columns:
            continue
        for model in piv.columns:
            if model == ANCHOR:
                continue
            pair = piv[[ANCHOR, model]].dropna()
            if len(pair) < 3:
                continue
            methods = list(pair.index)
            ca = per[(per["model"] == ANCHOR) & (per["dataset"] == dataset)]
            cb = per[(per["model"] == model) & (per["dataset"] == dataset)]
            tau, lo, hi, n = AL.kendall_tau_with_ci(ca, cb, methods, max_rejection, n_resamples, SEED)
            rows.append({"model": model, "dataset": dataset,
                         "kendall_tau_vs_anchor": AL._kendall(pair[ANCHOR].to_numpy(),
                                                              pair[model].to_numpy()),
                         "tau_ci_low": lo, "tau_ci_high": hi,
                         "n_methods_compared": len(pair), "n_instances_paired": n})
    if rows:
        backup(results_dir, "rank_transfer_kendall_tau.csv")
        pd.DataFrame(rows).to_csv(os.path.join(results_dir, "rank_transfer_kendall_tau.csv"), index=False)
        print("Salvato: rank_transfer_kendall_tau.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--n_resamples_kendall", type=int, default=200)
    ap.add_argument("--max_rejection", type=float, default=0.5)
    args = ap.parse_args()

    AL.check_prr_implementation()
    alias_rows = None
    main_py = AL.find_main_py(args.results_dir)
    if main_py:
        pm = AL.load_paper_methods(main_py)
        alias_rows = [(r.paper_label, r.alias_of) for r in pm.itertuples() if r.status == "alias"]

    main_stats = main_per = None
    for per_file, stats_file, mapped_file in SECTIONS:
        if not os.path.exists(os.path.join(args.results_dir, per_file)):
            continue
        print(f"\n--- {per_file} ---")
        st, per = recompute_section(args.results_dir, per_file, stats_file, mapped_file,
                                    args.n_bootstrap, args.max_rejection, alias_rows)
        if per_file == "per_instance_scores.csv":
            main_stats, main_per = st, per
    recompute_kendall(args.results_dir, main_stats, main_per, args.max_rejection,
                      args.n_resamples_kendall)
    print("\nFatto. Ora: paired_comparisons.py e make_figures.py sulla stessa cartella.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
