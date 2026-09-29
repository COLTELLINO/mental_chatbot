"""Quali metodi UQ sono statisticamente indistinguibili dal migliore, per cella.

Perche' serve. Le barre d'errore nelle figure dicono quanto balla il PRR di UN
metodo, ma non bastano per scrivere "A batte B": A e B sono valutati sulle
STESSE domande, quindi i loro punteggi sono correlati, e confrontare due
intervalli separati e' troppo conservativo. Il modo corretto e' ricampionare le
domande e ricalcolare i PRR di tutti i metodi sullo stesso campione.

Cosa fa (versione del 29/09). Per ogni cella (modello, dataset):

1. PRR del paper (normalizzato fra caso e oracolo) con i pareggi in valore
   atteso, dagli stessi array per-istanza salvati da main.py (vedi
   analysis_lib.prr_normalized). Prima si usava l'area grezza.
2. Si ricampionano le domande B volte (default 1000) e su ogni ricampionamento
   si ricalcolano i PRR di TUTTI i metodi e si guarda CHI E' IL MIGLIORE IN
   QUEL RICAMPIONAMENTO. Prima il migliore veniva scelto una volta sui dati
   completi e poi testato sugli stessi dati: chi vinceva "per fortuna"
   sembrava sistematicamente migliore di quanto fosse (winner's curse).
3. Un metodo e' dichiarato PEGGIORE del migliore solo se risulta il migliore in
   meno di alpha/(m-1) dei ricampionamenti (m = metodi confrontati): con 26
   metodi e alpha = 0.05, meno del 0.2%. E' la correzione di Bonferroni per i
   m-1 confronti (prima: 25 confronti al 5% ciascuno, nessuna correzione, e
   qualche differenza "significativa" saltava fuori per caso). Non e' un test
   d'ipotesi formale ma una regola di selezione conservativa, nello spirito
   del Model Confidence Set (Hansen, Lunde e Nason, 2011): un metodo resta nel
   gruppo di testa finche' i dati non lo escludono chiaramente. Nella tesi va
   descritta cosi'.
4. Se il PRR non e' calcolabile in almeno meta' dei ricampionamenti, il
   verdetto e' NON TESTABILE invece di "distinguibile" (prima veniva contato
   come se fosse peggiore). Succede quando la qualita' e' costante (accuracy 0
   o 1 nella cella o nella maggior parte dei ricampionamenti), e allora vale
   per tutti i metodi della cella insieme. Un punteggio costante invece NON e'
   non testabile: il suo PRR e' 0 (come scartare a caso) e viene confrontato
   normalmente.

Restano nel CSV, per confronto, la differenza dal migliore sui dati completi e
il suo intervallo appaiato al 95% (non corretto) e con correzione di
Bonferroni.

Non tocca la GPU: rilegge i punteggi per-istanza gia' salvati da main.py.

Uso:
    python3.11 paired_comparisons.py /workspace/results
    python3.11 paired_comparisons.py /workspace/results --per_instance_file results_severity_grid_per_instance.csv
"""
import argparse
import os
import sys
import zlib

import numpy as np
import pandas as pd

import analysis_lib as AL

SEED = 3407
CORRECTNESS_COLUMN = "quality"
NON_ESTIMATOR_COLUMNS = {"model", "dataset", "instance_index", CORRECTNESS_COLUMN, "greedy_text"}


def check_equivalence():
    """Controlli di sanita' sul PRR prima di usarlo (casuale -> 0, oracolo -> 1,
    area identica a lm-polygraph senza pareggi, pareggi = media sugli
    spareggi). Se non passano, i risultati non vanno usati."""
    worst = AL.check_prr_implementation()
    print(f"Controlli PRR superati (scarto massimo dall'area di lm-polygraph: {worst:.1e}).")
    return worst


def compare_cell(df_cell, n_resamples, rng, max_rejection=0.5, alpha=0.05):
    """Confronta ogni metodo contro il migliore, dentro una cella
    (modello, dataset). Ritorna una lista di righe."""
    quality = df_cell[CORRECTNESS_COLUMN].to_numpy(dtype=float)
    estimator_cols = [c for c in df_cell.columns if c not in NON_ESTIMATOR_COLUMNS]
    ue = {c: df_cell[c].to_numpy(dtype=float) for c in estimator_cols}
    point = {c: AL.prr_normalized(ue[c], quality, max_rejection) for c in estimator_cols}
    methods = [c for c in estimator_cols if np.isfinite(point[c])]
    not_testable = [c for c in estimator_cols if c not in methods]
    if len(methods) < 2:
        return []

    n = len(quality)
    boot = np.full((n_resamples, len(methods)), np.nan)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        q = quality[idx]
        for j, m in enumerate(methods):
            boot[b, j] = AL.prr_normalized(ue[m][idx], q, max_rejection)

    best_name = max(methods, key=lambda m: point[m])
    jb = methods.index(best_name)
    m_compared = len(methods) - 1
    alpha_bonf = alpha / max(m_compared, 1)

    # Migliore dentro ogni ricampionamento (fra i metodi calcolabili li').
    valid_rows = np.isfinite(boot).any(axis=1)
    boot_max = np.where(valid_rows, np.nanmax(np.where(np.isfinite(boot), boot, -np.inf), axis=1), np.nan)

    rows = []
    for j, m in enumerate(methods):
        col = boot[:, j]
        ok = np.isfinite(col) & valid_rows
        testable = ok.sum() >= n_resamples / 2
        p_best = float(np.mean(col[ok] >= boot_max[ok] - 1e-12)) if ok.any() else np.nan
        both = ok & np.isfinite(boot[:, jb])
        d = boot[both, jb] - col[both]
        if both.sum() >= n_resamples / 2 and m != best_name:
            ci = (float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2))))
            ci_b = (float(np.percentile(d, 100 * alpha_bonf / 2)),
                    float(np.percentile(d, 100 * (1 - alpha_bonf / 2))))
            mean_d = float(np.mean(d))
        else:
            ci = ci_b = (np.nan, np.nan)
            mean_d = np.nan if m != best_name else 0.0
        if m == best_name:
            verdict = "migliore"
        elif not testable:
            verdict = "non testabile"
        elif p_best < alpha_bonf:
            verdict = "peggiore"
        else:
            verdict = "indistinguibile"
        rows.append({
            "model": df_cell["model"].iloc[0],
            "dataset": df_cell["dataset"].iloc[0],
            "best_method": best_name,
            "best_prr": point[best_name],
            "method": m,
            "prr": point[m],
            "p_best": p_best,
            "alpha_bonferroni": alpha_bonf,
            "verdict": verdict,
            "indistinguishable_from_best": verdict in ("indistinguibile", "migliore"),
            "diff_best_minus_method": mean_d,
            "diff_ci_low": ci[0],
            "diff_ci_high": ci[1],
            "diff_ci_low_bonferroni": ci_b[0],
            "diff_ci_high_bonferroni": ci_b[1],
            "n_instances": int(n),
        })
    for m in not_testable:
        rows.append({
            "model": df_cell["model"].iloc[0], "dataset": df_cell["dataset"].iloc[0],
            "best_method": best_name, "best_prr": point[best_name], "method": m,
            "prr": point[m], "p_best": np.nan, "alpha_bonferroni": alpha_bonf,
            "verdict": "non testabile", "indistinguishable_from_best": False,
            "diff_best_minus_method": np.nan, "diff_ci_low": np.nan, "diff_ci_high": np.nan,
            "diff_ci_low_bonferroni": np.nan, "diff_ci_high_bonferroni": np.nan,
            "n_instances": int(n),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("--per_instance_file", default="per_instance_scores.csv",
                        help="File dei punteggi per-istanza da analizzare. Per la griglia "
                             "severita' usare results_severity_grid_per_instance.csv "
                             "(default: %(default)s).")
    parser.add_argument("--out", default=None,
                        help="Nome del CSV di output (default: derivato dal file di input).")
    parser.add_argument("--n_resamples", type=int, default=1000,
                        help="Ricampionamenti bootstrap per cella (default: %(default)s). "
                             "Abbassarlo accelera in modo proporzionale.")
    parser.add_argument("--max_rejection", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    check_equivalence()

    path = os.path.join(args.results_dir, args.per_instance_file)
    if not os.path.exists(path):
        raise SystemExit(f"!!! Non trovo {path}. Serve un run che abbia salvato i "
                         f"punteggi per-istanza.")
    df = pd.read_csv(path)
    missing = {"model", "dataset", CORRECTNESS_COLUMN} - set(df.columns)
    if missing:
        raise SystemExit(f"!!! {path} non ha le colonne {sorted(missing)}.")

    all_rows = []
    for (model_name, dataset_name), cell in df.groupby(["model", "dataset"], sort=False):
        cell = cell.dropna(axis=1, how="all")
        # Un generatore per cella, con seme ricavato dalla cella: i risultati di
        # una cella non dipendono dall'ordine delle righe nel CSV ne' da quali
        # altre celle ci sono (prima il generatore era condiviso).
        rng = np.random.default_rng([SEED, zlib.crc32(f"{model_name}|{dataset_name}".encode())])
        rows = compare_cell(cell, args.n_resamples, rng, args.max_rejection, args.alpha)
        all_rows.extend(rows)
        if rows:
            r = pd.DataFrame(rows)
            print(f"{model_name}/{dataset_name}: migliore = {rows[0]['best_method']} "
                  f"(PRR {rows[0]['best_prr']:.3f}); indistinguibili "
                  f"{(r['verdict'] == 'indistinguibile').sum()}, peggiori "
                  f"{(r['verdict'] == 'peggiore').sum()}, non testabili "
                  f"{(r['verdict'] == 'non testabile').sum()}.")
        else:
            print(f"{model_name}/{dataset_name}: meno di due metodi con PRR valido, salto.")

    if not all_rows:
        raise SystemExit("!!! Nessun confronto calcolabile.")

    out_name = args.out or (
        os.path.splitext(args.per_instance_file)[0].replace("_per_instance", "")
        + "_paired_comparisons.csv")
    out_path = os.path.join(args.results_dir, out_name)
    result = pd.DataFrame(all_rows)
    result.to_csv(out_path, index=False)
    print(f"\nSalvato: {out_path} ({len(result)} righe)")

    # Riepilogo leggibile: quanti metodi restano in testa per cella. E' il
    # numero da citare quando si scrive una classifica.
    others = result[result["verdict"] != "migliore"]
    summary = (others.groupby(["model", "dataset"])
               .agg(migliore=("best_method", "first"),
                    prr_migliore=("best_prr", "first"),
                    metodi_confrontati=("method", "count"),
                    equivalenti_al_migliore=("verdict", lambda v: int((v == "indistinguibile").sum())),
                    peggiori=("verdict", lambda v: int((v == "peggiore").sum())),
                    non_testabili=("verdict", lambda v: int((v == "non testabile").sum())))
               .reset_index())
    summary_path = os.path.join(args.results_dir,
                                out_name.replace(".csv", "_summary.csv"))
    summary.to_csv(summary_path, index=False)
    print(f"Salvato: {summary_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
