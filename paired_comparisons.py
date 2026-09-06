"""Test appaiati sulle differenze di PRR fra metodi UQ (TODO 5).

Perche' serve. Le barre d'errore nelle figure dicono quanto balla il PRR di UN
metodo, ma non bastano per scrivere "A batte B": A e B sono valutati sulle
STESSE domande, quindi i loro punteggi sono correlati (entrambi faticano sulle
domande difficili) e confrontare due intervalli separati e' troppo
conservativo. Il modo corretto e' ricampionare le domande e calcolare la
differenza PRR(A) - PRR(B) sullo stesso campione, mille volte: se l'intervallo
delle differenze non contiene lo zero, il vantaggio e' reale.

Con ~26 metodi x 5 modelli x 4 dataset la griglia contiene migliaia di
confronti: per puro caso qualche "vincitore" e' garantito, come lanciare 26
monete e stupirsi che una faccia otto teste di fila. Senza questo test, ogni
classifica letta dalle figure e' aneddotica.

Cosa fa. Per ogni coppia (modello, dataset) identifica il metodo con PRR piu'
alto e confronta TUTTI gli altri contro quello. L'output non e' "chi vince" ma
la domanda utile in tesi: **quali metodi sono statisticamente
indistinguibili dal migliore**. Se dieci metodi lo sono, la frase da scrivere
non e' "il metodo X e' il migliore" ma "un gruppo di metodi e' equivalente in
testa, e fra questi X e' il piu' economico" -- che e' anche il modo in cui
questo test si aggancia alla frontiera di Pareto del TODO 6.

Non tocca la GPU: rilegge i punteggi per-istanza gia' salvati da main.py.

Uso (dentro il container, dove ci sono pandas e lm-polygraph):
    python3.11 paired_comparisons.py /workspace/results
    python3.11 paired_comparisons.py /workspace/results --n_resamples 200
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

SEED = 3407
CORRECTNESS_COLUMN = "quality"
NON_ESTIMATOR_COLUMNS = {"model", "dataset", CORRECTNESS_COLUMN}


def prr_fast(ue, quality, max_rejection=0.5):
    """PRR vettorizzato, equivalente a main._prr_from_arrays.

    La versione in main.py ricalcola la media del sottoinsieme trattenuto per
    ogni soglia di rifiuto con un ciclo Python: va benissimo una volta sola per
    cella, ma qui serve dentro un bootstrap appaiato (centinaia di migliaia di
    valutazioni) e diventa il collo di bottiglia. Le stesse medie cumulate si
    ottengono in un colpo con np.cumsum, in O(n log n) per il solo
    ordinamento. L'equivalenza numerica con l'implementazione di riferimento e'
    verificata da check_equivalence() prima di usare questa funzione.

    Replica anche le due convenzioni di lm-polygraph: le istanze con qualita'
    NaN vengono scartate, i NaN nel punteggio di incertezza diventano -1e7
    (cioe' massima confidenza -- vedi il commento in main._prr_from_arrays sul
    perche' questo penalizza la lettura dei metodi verbalized).
    """
    ue = np.nan_to_num(np.asarray(ue, dtype=float), nan=-1e7, neginf=-1e7, posinf=1e7)
    quality = np.asarray(quality, dtype=float)
    keep = ~np.isnan(quality)
    ue, q = ue[keep], quality[keep]
    n = len(q)
    if n == 0:
        return np.nan
    qmin, qmax = np.min(q), np.max(q)
    if qmax == qmin:
        return np.nan  # normalizzazione min-max non definita
    q = (q - qmin) / (qmax - qmin)

    # np.argsort SENZA kind esplicito: deve essere lo stesso algoritmo del
    # riferimento in lm-polygraph. Con "stable" i pareggi vengono ordinati
    # diversamente, e i pareggi qui sono frequenti -- tutti i NaN diventano
    # -1e7 e finiscono appaiati -- il che produce PRR diversi (misurato: fino a
    # 8e-3 di scarto, cioe' abbastanza da cambiare una classifica).
    order = np.argsort(ue)  # incertezza crescente
    q_sorted = q[order]
    csum = np.cumsum(q_sorted)
    n_max = int(n * max_rejection)
    kept_counts = n - np.arange(n_max + 1)
    means = csum[kept_counts - 1] / kept_counts
    return float(np.mean(means))


def check_equivalence(n_checks=25, seed=SEED):
    """Verifica prr_fast contro l'implementazione di main.py su dati casuali.

    Se main.py non e' importabile (fuori dal container, senza lm-polygraph) il
    controllo viene saltato con un avviso esplicito invece di far finta di
    averlo fatto.
    """
    try:
        import main as reference_module
        from lm_polygraph.ue_metrics import PredictionRejectionArea
    except Exception as e:
        print(f"ATTENZIONE: main.py non importabile ({type(e).__name__}), "
              f"equivalenza di prr_fast NON verificata in questa esecuzione.")
        return None

    rng = np.random.default_rng(seed)
    metric = PredictionRejectionArea(max_rejection=0.5)
    worst = 0.0
    for _ in range(n_checks):
        n = int(rng.integers(30, 200))
        quality = (rng.random(n) < rng.uniform(0.2, 0.8)).astype(float)
        ue = rng.normal(0, 1, n)
        if rng.random() < 0.3:
            quality[rng.integers(0, n, size=3)] = np.nan
        mine = prr_fast(ue, quality)
        theirs = reference_module._prr_from_arrays(metric, ue, quality)
        if np.isnan(mine) and np.isnan(theirs):
            continue
        worst = max(worst, abs(mine - theirs))
    print(f"Equivalenza prr_fast vs main._prr_from_arrays: scarto massimo {worst:.2e} "
          f"su {n_checks} casi casuali.")
    if worst > 1e-9:
        raise SystemExit("!!! prr_fast NON e' equivalente all'implementazione di "
                         "riferimento: risultati non affidabili, mi fermo.")
    return worst


def paired_diff_ci(ue_a, ue_b, quality, n_resamples, rng, max_rejection=0.5, alpha=0.05):
    """Intervallo bootstrap appaiato su PRR(A) - PRR(B).

    Il ricampionamento e' sugli INDICI delle domande e viene applicato a
    entrambi i metodi insieme: separarli distruggerebbe l'accoppiamento, che e'
    esattamente cio' che rende il test piu' potente del confronto fra due
    intervalli separati.
    """
    n = len(quality)
    diffs = np.empty(n_resamples)
    diffs[:] = np.nan
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        q = quality[idx]
        va = prr_fast(ue_a[idx], q, max_rejection)
        vb = prr_fast(ue_b[idx], q, max_rejection)
        if np.isfinite(va) and np.isfinite(vb):
            diffs[i] = va - vb
    valid = diffs[np.isfinite(diffs)]
    # Se piu' di meta' dei ricampionamenti e' degenere (tipicamente accuracy
    # costante nel campione), l'intervallo non e' affidabile: meglio NaN che un
    # numero falsamente preciso.
    if len(valid) < n_resamples // 2:
        return np.nan, np.nan, np.nan
    return (float(np.mean(valid)),
            float(np.percentile(valid, 100 * alpha / 2)),
            float(np.percentile(valid, 100 * (1 - alpha / 2))))


def compare_cell(df_cell, n_resamples, rng, max_rejection=0.5):
    """Confronta ogni metodo contro il migliore, dentro una cella
    (modello, dataset). Ritorna una lista di righe."""
    quality = df_cell[CORRECTNESS_COLUMN].to_numpy(dtype=float)
    estimator_cols = [c for c in df_cell.columns if c not in NON_ESTIMATOR_COLUMNS]

    scores = {}
    for col in estimator_cols:
        ue = df_cell[col].to_numpy(dtype=float)
        prr = prr_fast(ue, quality, max_rejection)
        if np.isfinite(prr):
            scores[col] = (ue, prr)
    if len(scores) < 2:
        return []

    best_name = max(scores, key=lambda k: scores[k][1])
    best_ue, best_prr = scores[best_name]

    rows = []
    for name, (ue, prr) in scores.items():
        if name == best_name:
            continue
        mean_d, lo, hi = paired_diff_ci(best_ue, ue, quality, n_resamples, rng, max_rejection)
        # "Indistinguibile dal migliore" = l'intervallo della differenza
        # contiene lo zero.
        indistinguishable = bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi)
        rows.append({
            "model": df_cell["model"].iloc[0],
            "dataset": df_cell["dataset"].iloc[0],
            "best_method": best_name,
            "best_prr": best_prr,
            "method": name,
            "prr": prr,
            "diff_best_minus_method": mean_d,
            "diff_ci_low": lo,
            "diff_ci_high": hi,
            "indistinguishable_from_best": indistinguishable,
            "n_instances": int(len(quality)),
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
                        help="Ricampionamenti bootstrap per confronto (default: %(default)s). "
                             "Abbassarlo accelera in modo proporzionale.")
    parser.add_argument("--max_rejection", type=float, default=0.5)
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

    rng = np.random.default_rng(SEED)
    all_rows = []
    for (model_name, dataset_name), cell in df.groupby(["model", "dataset"], sort=False):
        rows = compare_cell(cell, args.n_resamples, rng, args.max_rejection)
        all_rows.extend(rows)
        if rows:
            n_tied = sum(r["indistinguishable_from_best"] for r in rows)
            print(f"{model_name}/{dataset_name}: migliore = {rows[0]['best_method']} "
                  f"(PRR {rows[0]['best_prr']:.3f}); {n_tied} metodi su {len(rows)} "
                  f"non distinguibili da esso.")
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
    print(f"\nSalvato: {out_path} ({len(result)} confronti)")

    # Riepilogo leggibile: quanti metodi restano in testa per cella. E' il
    # numero da citare quando si scrive una classifica.
    summary = (result.groupby(["model", "dataset"])
               .agg(migliore=("best_method", "first"),
                    prr_migliore=("best_prr", "first"),
                    metodi_confrontati=("method", "count"),
                    equivalenti_al_migliore=("indistinguishable_from_best", "sum"))
               .reset_index())
    summary_path = os.path.join(args.results_dir,
                                out_name.replace(".csv", "_summary.csv"))
    summary.to_csv(summary_path, index=False)
    print(f"Salvato: {summary_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
