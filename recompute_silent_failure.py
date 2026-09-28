"""Ricalcola il silent failure rate di una cartella di risultati esistente,
senza GPU, a partire dai punteggi per-istanza gia' salvati.

Serve per le cartelle prodotte prima della correzione della gestione dei
pareggi (vedi analysis_lib.silent_failure_rate): la vecchia versione metteva
nel "decile piu' confidente" tutte le istanze con incertezza <= al 10mo
percentile, quindi un metodo che assegna lo stesso punteggio a molte istanze
finiva con quasi tutte le istanze nel decile e un silent failure rate
gonfiato verso 1.

Uso:
    python recompute_silent_failure.py <results_dir>

Aggiorna in place, dopo averne fatto una copia *.pre_sfr_fix.bak:
  - i file di statistiche per istanza (instance_level_stats.csv e
    results_*_instance_stats.csv),
  - i file *_mapped.csv che riportano la colonna silent_failure_rate,
  - silent_failure_rate.csv (pivot della griglia di severita').
Le celle per cui non esistono punteggi per-istanza (es. il confronto sulla
quantizzazione, che non li salva) vengono messe a NaN: il valore vecchio non
e' affidabile e non puo' essere ricalcolato.
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

from analysis_lib import silent_failure_rate

CORRECTNESS_THRESHOLD = 0.5   # stesso valore di main.py

# (file di statistiche, file per-istanza da cui ricalcolarle)
PAIRS = [
    ("instance_level_stats.csv", "per_instance_scores.csv"),
    ("results_severity_grid_instance_stats.csv", "results_severity_grid_per_instance.csv"),
    ("results_verbalized_numeric_instance_stats.csv", "results_verbalized_numeric_per_instance.csv"),
    ("results_verbalized_linguistic_instance_stats.csv", "results_verbalized_linguistic_per_instance.csv"),
    ("results_quant_comparison_instance_stats.csv", "results_quant_comparison_per_instance.csv"),
]
MAPPED = {
    "instance_level_stats.csv": "results_paper_mapped.csv",
    "results_severity_grid_instance_stats.csv": "results_severity_grid_mapped.csv",
    "results_verbalized_numeric_instance_stats.csv": "results_verbalized_numeric_mapped.csv",
    "results_verbalized_linguistic_instance_stats.csv": "results_verbalized_linguistic_mapped.csv",
}
KEY = ["model", "dataset", "estimator"]


def backup(path):
    bak = path + ".pre_sfr_fix.bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)


def recompute(stats, per_instance):
    new_vals = {}
    if per_instance is not None:
        for (model, dataset), cell in per_instance.groupby(["model", "dataset"]):
            q = cell["quality"].to_numpy(dtype=float)
            for est in cell.columns.difference(["model", "dataset", "quality"]):
                new_vals[(model, dataset, est)] = silent_failure_rate(
                    cell[est].to_numpy(dtype=float), q,
                    correctness_threshold=CORRECTNESS_THRESHOLD)
    out = stats.copy()
    out["silent_failure_rate"] = [
        new_vals.get((m, d, e), np.nan)
        for m, d, e in zip(out["model"], out["dataset"], out["estimator"])]
    return out


def main(results_dir):
    for stats_name, per_inst_name in PAIRS:
        stats_path = os.path.join(results_dir, stats_name)
        if not os.path.exists(stats_path):
            continue
        stats = pd.read_csv(stats_path)
        if "silent_failure_rate" not in stats.columns:
            continue
        per_instance = None
        if per_inst_name and os.path.exists(os.path.join(results_dir, per_inst_name)):
            per_instance = pd.read_csv(os.path.join(results_dir, per_inst_name))
        new = recompute(stats, per_instance)
        changed = (~np.isclose(stats["silent_failure_rate"].fillna(-1),
                               new["silent_failure_rate"].fillna(-1))).sum()
        backup(stats_path)
        new.to_csv(stats_path, index=False)
        src = per_inst_name if per_instance is not None else "nessun file per-istanza: messo a NaN"
        print(f"{stats_name}: {changed}/{len(new)} valori cambiati ({src})")

        mapped_name = MAPPED.get(stats_name)
        mapped_path = os.path.join(results_dir, mapped_name) if mapped_name else None
        if mapped_path and os.path.exists(mapped_path):
            mapped = pd.read_csv(mapped_path)
            if "silent_failure_rate" in mapped.columns:
                lookup = new.set_index(KEY)["silent_failure_rate"].to_dict()
                mapped["silent_failure_rate"] = [
                    lookup.get((m, d, e), np.nan)
                    for m, d, e in zip(mapped["model"], mapped["dataset"], mapped["estimator"])]
                backup(mapped_path)
                mapped.to_csv(mapped_path, index=False)
                print(f"  aggiornato anche {mapped_name}")

        if stats_name == "results_severity_grid_instance_stats.csv":
            sfr = new.pivot_table(index="paper_label", columns=["dataset", "model"],
                                  values="silent_failure_rate", aggfunc="first")
            sfr_path = os.path.join(results_dir, "silent_failure_rate.csv")
            if os.path.exists(sfr_path):
                backup(sfr_path)
            sfr.to_csv(sfr_path)
            print("  riscritto silent_failure_rate.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
