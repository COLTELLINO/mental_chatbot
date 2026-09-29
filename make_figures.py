"""Figure di lettura del benchmark, generate dai CSV gia' salvati.

Perche' e' uno script separato da main.py: le figure prodotte a fine run
richiedono un job GPU da ore per essere rigenerate, anche quando si vuole solo
cambiare un asse o aggiungere un pannello. Questo legge i CSV e disegna in
pochi secondi, quante volte serve.

Tutti i PRR sono quelli del paper (normalizzati: 0 = punteggio casuale,
1 = oracolo), dal 29/09. L'elenco completo delle figure e' in generate_all().

Uso:
    python3.11 make_figures.py /workspace/results
"""
import argparse
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_lib import (
    FAMILY_ORDER,
    VERBALIZED_LABELS,
    aggregate_across_datasets,
    family_of,
    find_main_py,
    labels_for_figure,
    load_paper_methods,
    severity_matched,
)

# Soglia del caso per i task a scelta multipla: sotto questa un'accuracy non
# indica ignoranza del modello ma un problema di estrazione della risposta.
CHANCE_LEVEL = {"MMLU": 0.25, "MedQAbstain-LT": 0.25, "MedQAbstain-Safe": 0.25}

# Miliardi di parametri, per l'asse della scala.
MODEL_PARAMS_B = {"LFM2-350M": 0.35, "LFM2-1.2B": 1.2,
                  "MedGemma-4B-it": 4.0, "Gemma3-4B-it": 4.0,
                  "Mistral-7B-it": 7.0}

ANCHOR_MODEL = "Mistral-7B-it"

# Etichetta dell'asse per il PRR.
PRR_AXIS = "PRR (0 = casuale, 1 = oracolo; max_rejection=0.5)"

# Valori di riferimento del paper per l'ancora. Fino al 29/09 erano due numeri
# letti a occhio da una figura (TriviaQA 0.60, MMLU 0.50). Ora si leggono da
# paper_reference_prr.csv (colonne: dataset, paper_label, prr), da riempire con
# i valori delle Tabelle 6-7 di Vashurin et al. per Mistral 7B v0.2: il file
# viene cercato nella cartella dei risultati e poi accanto a questo script.
PAPER_REFERENCE_FILE = "paper_reference_prr.csv"


def load_paper_reference(results_dir):
    for base in (results_dir, os.path.dirname(os.path.abspath(__file__))):
        path = os.path.join(base, PAPER_REFERENCE_FILE)
        if os.path.exists(path):
            ref = pd.read_csv(path).dropna(subset=["prr"])
            if not ref.empty:
                return ref
    return None


def _drop_phase_rows(t):
    """Toglie da estimator_timings.csv le righe con i tempi per fase
    ("__fase__:<nome>"), che non sono metodi."""
    if "estimator" in t.columns:
        t = t[~t["estimator"].astype(str).str.startswith("__fase__:")]
    return t


def _ordina_modelli(nomi):
    """Modelli nell'ordine di scala di MODEL_PARAMS_B, seguiti da quelli non in
    elenco (prima venivano scartati, e un modello nuovo spariva dalle figure)."""
    nomi = list(dict.fromkeys(nomi))
    noti = [m for m in MODEL_PARAMS_B if m in nomi]
    return noti + sorted(m for m in nomi if m not in MODEL_PARAMS_B)


def load_stats(results_dir):
    """Unisce le statistiche per-istanza della pipeline principale e della
    griglia severita': stessa struttura, dataset diversi."""
    frames = []
    for name in ("instance_level_stats.csv", "results_severity_grid_instance_stats.csv"):
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            try:
                frames.append(pd.read_csv(path))
            except Exception as e:
                print(f"  {name}: illeggibile ({e}), salto.")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def add_note(fig, text, **adjust):
    """Nota esplicativa sotto la figura, con lo spazio calcolato in pollici e
    non come frazione.

    Un `subplots_adjust(bottom=0.1)` fisso funziona su una figura alta 5 pollici
    e sovrappone la nota all'etichetta dell'asse su una alta 12: la frazione e'
    la stessa, lo spazio assoluto no. Qui le altezze variano con il numero di
    righe, quindi lo spazio va riservato in valore assoluto."""
    n_righe = text.count("\n") + 1
    pollici_necessari = 0.16 * n_righe + 0.45
    altezza = fig.get_size_inches()[1]
    fig.subplots_adjust(bottom=min(0.4, pollici_necessari / altezza), **adjust)
    fig.text(0.01, 0.012, text, fontsize=7, va="bottom")


def save(fig, results_dir, filename):
    path = os.path.join(results_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvato: {path}")


def fig_accuracy(stats, results_dir):
    """Accuracy per modello e dataset, con la soglia del caso dove ha senso."""
    acc = (stats.groupby(["model", "dataset"], as_index=False)["mean_quality"]
           .first().pivot(index="dataset", columns="model", values="mean_quality"))
    if acc.empty:
        return
    fig, ax = plt.subplots(figsize=(11, max(4, len(acc) * 0.9)))
    acc.plot(kind="barh", ax=ax, width=0.8)

    # La soglia del caso vale solo per gli MCQ: la disegno come segmento sulla
    # singola riga invece che come linea verticale su tutto il grafico, che
    # suggerirebbe erroneamente che valga anche per i task a risposta libera.
    for i, ds in enumerate(acc.index):
        if ds in CHANCE_LEVEL:
            ax.plot([CHANCE_LEVEL[ds]] * 2, [i - 0.42, i + 0.42],
                    color="tab:red", linewidth=1.6, linestyle="--", zorder=5)
    ax.plot([], [], color="tab:red", linewidth=1.6, linestyle="--",
            label="livello del caso (solo MCQ)")

    ax.set_xlabel("Qualita' media delle generazioni (accuracy o AlignScore)")
    ax.set_title("Accuracy di base per modello e dataset")
    ax.legend(loc="lower right", fontsize=8)
    add_note(fig,
             "Un'accuracy sotto la linea rossa su un MCQ non e' ignoranza del modello: tirando a "
             "indovinare ne prenderebbe di piu'.\nIndica che la risposta non viene estratta "
             "correttamente (vedi il parse-failure rate in instance_level_stats.csv).",)
    save(fig, results_dir, "fig_accuracy.png")


def fig_accuracy_vs_prr(stats, results_dir):
    """Accuracy contro PRR, prima e dopo la correzione del 29/09.

    Ogni punto e' una coppia modello-dataset, con il PRR mediano fra i metodi.
    A sinistra l'area grezza riportata fino al 29/09: sta quasi esattamente
    sulla diagonale, cioe' misura l'accuracy del modello e non i metodi UQ. A
    destra il PRR del paper, normalizzato fra caso e oracolo: la dipendenza
    meccanica dall'accuracy sparisce."""
    cell = stats.groupby(["model", "dataset"], as_index=False).agg(
        accuracy=("mean_quality", "first"),
        prr_mediano=("prr", "median"),
        **({"raw_mediano": ("prr_raw", "median")} if "prr_raw" in stats.columns else {}),
    ).dropna(subset=["accuracy", "prr_mediano"])
    if cell.empty:
        return
    pannelli = ([("raw_mediano", "Area grezza (riportata fino al 29/09)")]
                if "raw_mediano" in cell.columns else []) + \
               [("prr_mediano", "PRR del paper (normalizzato)")]
    fig, axes = plt.subplots(1, len(pannelli), figsize=(8 * len(pannelli), 7), squeeze=False)
    for ax, (col, titolo) in zip(axes[0], pannelli):
        for model_name, group in cell.groupby("model"):
            ax.scatter(group["accuracy"], group[col], s=60, label=model_name, zorder=3)
            for _, row in group.iterrows():
                ax.annotate(row["dataset"], (row["accuracy"], row[col]),
                            textcoords="offset points", xytext=(5, 3), fontsize=6)
        r = cell["accuracy"].corr(cell[col])
        ax.axhline(0, color="black", linewidth=0.8)
        if col == "raw_mediano":
            ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accuracy di base del modello sul dataset")
        ax.set_ylabel("Valore mediano fra i metodi UQ")
        ax.set_title(f"{titolo}\ncorrelazione con l'accuracy: r = {r:.2f}", fontsize=10)
        ax.grid(linewidth=0.3, alpha=0.5)
    axes[0][-1].legend(fontsize=7)
    add_note(fig,
             "A sinistra l'area sotto la curva prediction-rejection: parte dal livello dell'accuracy "
             "e ci aggiunge poco, quindi confronta i modelli per accuracy.\nA destra il PRR del paper, "
             "(area - area casuale) / (area oracolo - area casuale): misura quanto il metodo UQ "
             "migliora rispetto a scartare a caso.\nCon pochissime risposte giuste o sbagliate il "
             "PRR normalizzato resta calcolabile ma rumoroso: guardare gli intervalli di confidenza.")
    save(fig, results_dir, "fig_accuracy_vs_prr.png")


def fig_anchor_replication(stats, results_dir):
    """Ancora di replica: i PRR di Mistral-7B contro i valori del paper
    (paper_reference_prr.csv), metodo per metodo."""
    anchor = stats[stats["model"] == ANCHOR_MODEL]
    if anchor.empty:
        print(f"  {ANCHOR_MODEL} assente dai risultati, salto la figura dell'ancora.")
        return
    ref = load_paper_reference(results_dir)
    datasets = sorted(anchor["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 8), sharey=True,
                             squeeze=False)
    ordine = (anchor.groupby("paper_label")["prr"].mean().sort_values().index.tolist())
    y = np.arange(len(ordine))
    for ax, ds in zip(axes[0], datasets):
        sub = anchor[anchor["dataset"] == ds].set_index("paper_label").reindex(ordine)
        err = np.vstack([(sub["prr"] - sub["prr_ci_low"]).clip(lower=0).fillna(0),
                         (sub["prr_ci_high"] - sub["prr"]).clip(lower=0).fillna(0)])
        ax.barh(y, sub["prr"], xerr=err, color="tab:blue", alpha=0.8,
                error_kw={"elinewidth": 0.6, "ecolor": "0.35"}, label="ottenuto (IC 95%)")
        if ref is not None:
            r = ref[ref["dataset"] == ds].set_index("paper_label")["prr"].reindex(ordine)
            ax.scatter(r, y, color="tab:red", marker="D", s=22, zorder=4, label="paper")
        acc = sub["mean_quality"].dropna()
        ax.set_title(f"{ds}\n(accuracy {acc.iloc[0]:.2f})" if len(acc) else ds, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("PRR", fontsize=8)
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels(ordine, fontsize=7)
    axes[0][-1].legend(fontsize=7, loc="lower right")
    precisione = "?"
    cond = os.path.join(results_dir, "run_conditions.csv")
    if os.path.exists(cond):
        c = pd.read_csv(cond)
        c = c[c["model"] == ANCHOR_MODEL]
        if len(c):
            precisione = ", ".join(sorted(c["precision"].astype(str).unique()))
    fig.suptitle(f"Ancora di replica: {ANCHOR_MODEL} ({precisione}) contro Vashurin et al.")
    nota = ("L'ancora serve a distinguere gli effetti di scala dagli artefatti della nostra "
            "pipeline: ha valore se riproduce i valori del paper nel suo stesso regime.\n")
    if ref is None:
        nota += ("Valori del paper NON disponibili: riempire paper_reference_prr.csv (dataset, "
                 "paper_label, prr) con le Tabelle 6-7 di Vashurin et al. per Mistral 7B v0.2.")
    else:
        nota += ("Rombi rossi: valori delle Tabelle 6-7 del paper (paper_reference_prr.csv). "
                 "Criterio di successo da fissare PRIMA di guardare i risultati.")
    add_note(fig, nota, left=0.2, top=0.9)
    save(fig, results_dir, "fig_anchor_replication.png")


def fig_ties_vs_scale(results_dir):
    """Quanti metodi restano indistinguibili dal migliore, per scala.

    E' la lettura che qualifica il Kendall tau: se a una certa scala nessun
    metodo e' distinguibile dagli altri, il suo ranking e' rumore e un tau
    vicino a zero contro l'ancora e' garantito per costruzione, non e' un
    risultato sul trasferimento."""
    frames = []
    for name in ("per_instance_scores_paired_comparisons_summary.csv",
                 "results_severity_grid_paired_comparisons_summary.csv"):
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        print("  Nessun riepilogo dei confronti appaiati, salto la figura dei ties. "
              "Esegui prima paired_comparisons.py.")
        return
    summary = pd.concat(frames, ignore_index=True)
    agg = summary.groupby("model").agg(
        ties=("equivalenti_al_migliore", "mean"),
        confrontati=("metodi_confrontati", "mean"),
        celle=("dataset", "count"),
        **({"non_testabili": ("non_testabili", "mean")} if "non_testabili" in summary.columns else {}),
    ).reset_index()
    agg["params_B"] = agg["model"].map(MODEL_PARAMS_B)
    agg = agg.dropna(subset=["params_B"]).sort_values("params_B")
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    # Solo punti: modelli di famiglie diverse (LFM2, Gemma, Mistral) non
    # stanno su una stessa curva, e una linea che li unisce suggerirebbe
    # un andamento che il disegno sperimentale non puo' mostrare.
    for i, (_, r) in enumerate(agg.iterrows()):
        ax.scatter([r["params_B"]], [r["ties"]], s=60, color=f"C{i}", zorder=3)
    for _, r in agg.iterrows():
        extra = (f", {r['non_testabili']:.1f} non testabili" if "non_testabili" in agg.columns
                 and pd.notna(r.get("non_testabili")) else "")
        ax.annotate(f"{r['model']}\n({int(r['celle'])} dataset{extra})",
                    (r["params_B"], r["ties"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_ylim(0, max(agg["confrontati"].max(), agg["ties"].max()) * 1.15)
    ax.set_xlabel("Parametri del modello (miliardi, scala log)")
    ax.set_ylabel("Metodi indistinguibili dal migliore (media sui dataset)")
    ax.set_title("Capacita' di distinguere i metodi UQ, al variare della scala")
    add_note(fig,
             "Un valore alto significa che il benchmark, a quella scala, non separa i metodi: la loro "
             "classifica e' rumore.\nVa letto insieme al Kendall tau, perche' un ranking indistinguibile "
             "produce tau vicino a zero per costruzione.\nMetodo 'peggiore' solo se risulta il "
             "migliore in meno di 0.05/(m-1) dei ricampionamenti (migliore scelto dentro ogni "
             "ricampionamento, Bonferroni).",)
    save(fig, results_dir, "fig_ties_vs_scale.png")


def fig_timing_marginal(results_dir):
    """Costo MARGINALE per metodo e modello (sola aritmetica dello stimatore,
    sommata su batch e dataset). Era disegnata da main.py a fine run."""
    path = os.path.join(results_dir, "estimator_timings.csv")
    if not os.path.exists(path):
        return
    t = _drop_phase_rows(pd.read_csv(path))
    if "seconds" not in t.columns:
        return
    piv = t.pivot_table(index="paper_label", columns="model", values="seconds", aggfunc="sum")
    piv = piv.reindex(columns=_ordina_modelli(piv.columns))
    if piv.empty:
        return
    piv = piv.loc[piv.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(10, max(8, len(piv) * 0.4)))
    piv.plot(kind="barh", ax=ax, width=0.8, logx=True)
    ax.set_xlabel("Tempo di sola aritmetica dello stimatore, sommato su batch e dataset "
                  "(secondi, scala log)")
    ax.set_ylabel("")
    ax.set_title("Costo MARGINALE per metodo UQ e modello")
    ax.legend(loc="lower right", fontsize=8)
    add_note(fig,
             "Costo marginale = solo il calcolo dello stimatore su statistiche gia' pronte. NON e' il "
             "costo di eseguire la tecnica da sola:\nper quello vedi fig_timing_full_cost.png, la "
             "frontiera di Pareto ed estimator_cost_table.csv.")
    save(fig, results_dir, "estimator_timing_chart.png")


def fig_verbalized_by_style(results_dir):
    """PRR dei metodi verbalized per modello, uno per stile di confidenza, con
    intervalli di confidenza e la quota di confidenze non estraibili accanto al
    nome del modello. Era disegnata da main.py a fine run."""
    for style in ("numeric", "linguistic"):
        raw = _read_mapped(results_dir, f"results_verbalized_{style}_mapped.csv")
        if raw is None or raw.empty:
            continue
        agg = aggregate_across_datasets(raw)
        modelli = _ordina_modelli(agg["model"])
        if not modelli:
            continue
        agg = agg.set_index("model").reindex(modelli).reset_index()
        pf = (raw.groupby("model")["nan_rate"].mean()
              if "nan_rate" in raw.columns else pd.Series(dtype=float))
        etichette = [f"{m}\n(confidenza non estraibile: {pf[m]:.0%})" if m in pf else m
                     for m in modelli]
        datasets = ", ".join(sorted(raw["dataset"].unique()))
        fig, ax = plt.subplots(figsize=(9, 1.2 + 0.7 * len(modelli)))
        xerr = None
        if {"prr_ci_low", "prr_ci_high"}.issubset(agg.columns):
            xerr = np.stack([(agg["value"] - agg["prr_ci_low"]).clip(lower=0).fillna(0),
                             (agg["prr_ci_high"] - agg["value"]).clip(lower=0).fillna(0)])
        ax.barh(etichette, agg["value"], xerr=xerr, color="tab:purple", alpha=0.8,
                error_kw={"elinewidth": 0.6, "ecolor": "0.35"})
        ax.invert_yaxis()
        ax.set_xlabel(f"Mean PRR (0 = casuale, 1 = oracolo; media su {datasets})")
        titolo = "numerica" if style == "numeric" else "verbale"
        ax.set_title(f"Metodi verbalized: confidenza {titolo} dichiarata dal modello")
        add_note(fig,
                 "Prompt e max_new_tokens diversi dalla pipeline principale: valori NON confrontabili "
                 "con le Figure A/B (vedi fig_verbalized_accuracy_cost.png).\nlm-polygraph tratta una "
                 "confidenza non estraibile come massima confidenza: con una quota alta fra parentesi, "
                 "il PRR di quel modello\nnon misura quasi nulla.")
        save(fig, results_dir, f"fig_verbalized_{style}.png")


def fig_verbalized_accuracy_cost(results_dir):
    """Quanto costa, in qualita' delle risposte, chiedere al modello di
    dichiarare la propria confidenza.

    Perche' e' una figura a se'. La letteratura sui metodi verbalized discute
    il parse-failure rate, cioe' quante volte il modello non produce il formato
    richiesto. Ma il costo arriva prima: cambiando il prompt per chiedere la
    confidenza, la RISPOSTA stessa peggiora. Misurato qui fino a -0.45 di
    accuracy media su CoQA, con LFM2-1.2B che passa da 0.594 a 0.120.

    Serve anche a giustificare una scelta di lettura delle Figure A e B: il PRR
    dei metodi verbalized e' calcolato su un insieme di risposte in gran parte
    diverso da quello degli altri metodi, quindi le loro barre non sono
    confrontabili una a una con le altre.
    """
    base_path = os.path.join(results_dir, "accuracy_table.csv")
    if not os.path.exists(base_path):
        print("  accuracy_table.csv assente, salto la figura del costo verbalized.")
        return
    base = pd.read_csv(base_path).set_index("model")

    varianti = {}
    for style in ("numeric", "linguistic"):
        path = os.path.join(results_dir, f"accuracy_table_verbalized_{style}.csv")
        if os.path.exists(path):
            varianti[style] = pd.read_csv(path).set_index("model")
    if not varianti:
        print("  Nessuna tabella verbalized, salto la figura del costo.")
        return

    righe = []
    for model_name in base.index:
        for col in base.columns:
            senza = base.loc[model_name, col]
            if pd.isna(senza):
                continue
            con = {s: v.loc[model_name, col]
                   for s, v in varianti.items()
                   if model_name in v.index and col in v.columns and not pd.isna(v.loc[model_name, col])}
            if con:
                righe.append({"cella": f"{model_name}\n{col.split(' (')[0]}",
                              "senza": senza, **{f"con_{k}": v for k, v in con.items()}})
    if not righe:
        print("  Nessuna cella confrontabile, salto la figura del costo verbalized.")
        return

    df = pd.DataFrame(righe)
    con_cols = [c for c in df.columns if c.startswith("con_")]
    df["peggiore"] = df[con_cols].min(axis=1)
    df = df.sort_values("senza", ascending=True).reset_index(drop=True)

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.34)))
    for i, row in df.iterrows():
        ax.plot([row["peggiore"], row["senza"]], [i, i],
                color="tab:red" if row["peggiore"] < row["senza"] else "tab:green",
                linewidth=1.4, zorder=1, alpha=0.7)
    ax.scatter(df["senza"], y, s=48, color="tab:blue", zorder=3,
               label="prompt normale")
    marcatori = {"con_numeric": ("o", "tab:orange", "con richiesta di confidenza (numerica)"),
                 "con_linguistic": ("s", "tab:purple", "con richiesta di confidenza (verbale)")}
    for col in con_cols:
        marker, color, label = marcatori.get(col, ("^", "tab:gray", col))
        ax.scatter(df[col], y, s=40, marker=marker, color=color, zorder=3, label=label)

    ax.set_yticks(y)
    ax.set_yticklabels(df["cella"], fontsize=7)
    ax.set_xlabel("Qualita' media delle risposte (accuracy o AlignScore)")
    ax.set_title("Il costo nascosto dei metodi verbalized: chiedere la confidenza peggiora la risposta")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    add_note(fig,
             "Ogni riga e' una coppia modello-dataset. Il segmento va dalla qualita' con il prompt "
             "normale a quella con la richiesta di confidenza.\nDove il segmento e' lungo, il PRR dei "
             "metodi verbalized e' misurato su risposte molto piu' sbagliate del resto del benchmark, "
             "e non\ne' confrontabile barra a barra con gli altri metodi.", left=0.2)
    save(fig, results_dir, "fig_verbalized_accuracy_cost.png")


# Nomi leggibili per gli stimatori della sezione verbalized, che nei CSV
# compaiono con il nome interno della classe piu' il suffisso dello stile.
VERBALIZED_RENAME = {
    "Verbalized1S_numeric": "Verbalized 1S (confidenza numerica)",
    "Linguistic1S_linguistic": "Linguistic 1S (confidenza verbale)",
}


def _read_mapped(results_dir, filename):
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  {filename}: illeggibile ({e}), salto.")
        return None


def _barh_con_ci(ax, pivot, err_low=None, err_high=None):
    """Barre orizzontali raggruppate per modello, con barre d'errore."""
    xerr = None
    if err_low is not None:
        xerr = np.stack([
            np.stack([np.nan_to_num(err_low[c].to_numpy(dtype=float)),
                      np.nan_to_num(err_high[c].to_numpy(dtype=float))])
            for c in pivot.columns
        ])
    pivot.plot(kind="barh", ax=ax, width=0.8, xerr=xerr,
               error_kw={"elinewidth": 0.6, "ecolor": "0.35"})


def fig_paper_figure(results_dir, figure_letter, out_name, titolo):
    """Figura A o B: tutte le righe del paper, non solo quelle che calcoliamo.

    Tre gruppi separati visivamente, perche' rappresentano cose diverse:
      - i metodi inclusi, con i loro valori e intervalli;
      - i metodi verbalized, che girano con un prompt e un max_new_tokens
        diversi e le cui barre NON sono confrontabili una a una con le altre
        (la richiesta di confidenza peggiora le risposte fino a -0.45 di
        accuracy: vedi fig_verbalized_accuracy_cost.png);
      - i metodi del paper che non calcoliamo, mostrati come riga vuota con il
        motivo, perche' una figura che replica quella del paper deve avere le
        sue stesse righe: chi confronta le due deve poter vedere subito cosa
        manca e perche', invece di dover contare le barre.
    """
    main_py = find_main_py(results_dir)
    if main_py is None:
        print("  main.py non trovato, salto Figure A/B (serve la mappa dei metodi).")
        return
    paper_methods = load_paper_methods(main_py)
    righe_figura = labels_for_figure(paper_methods, figure_letter)

    raw = _read_mapped(results_dir, "results_paper_mapped.csv")
    if raw is None:
        print(f"  results_paper_mapped.csv assente, salto {out_name}.")
        return
    agg = aggregate_across_datasets(raw)

    # Sezione verbalized: stessi calcoli, file diversi.
    verb_frames = []
    for style in ("numeric", "linguistic"):
        v = _read_mapped(results_dir, f"results_verbalized_{style}_mapped.csv")
        if v is not None:
            verb_frames.append(v)
    verb_agg = aggregate_across_datasets(pd.concat(verb_frames, ignore_index=True)) \
        if verb_frames else None
    if verb_agg is not None:
        verb_agg["paper_label"] = verb_agg["paper_label"].replace(VERBALIZED_RENAME)

    modelli = _ordina_modelli(agg["model"])
    if not modelli:
        return

    inclusi = righe_figura[righe_figura["status"].isin(["incluso", "alias"])]["paper_label"].tolist()
    pivot_inc = agg.pivot_table(index="paper_label", columns="model",
                                values="value", aggfunc="first").reindex(
        index=[l for l in inclusi if l in set(agg["paper_label"])], columns=modelli)
    if pivot_inc.empty:
        return
    pivot_inc = pivot_inc.loc[pivot_inc.mean(axis=1).sort_values().index]

    lo_inc = agg.pivot_table(index="paper_label", columns="model", values="prr_ci_low",
                             aggfunc="first").reindex(index=pivot_inc.index, columns=modelli)
    hi_inc = agg.pivot_table(index="paper_label", columns="model", values="prr_ci_high",
                             aggfunc="first").reindex(index=pivot_inc.index, columns=modelli)

    blocchi = [("inclusi", pivot_inc, (pivot_inc - lo_inc).clip(lower=0),
                (hi_inc - pivot_inc).clip(lower=0))]

    verb_labels = [l for l in righe_figura["paper_label"] if l in VERBALIZED_LABELS]
    if verb_agg is not None and verb_labels:
        pv = verb_agg.pivot_table(index="paper_label", columns="model",
                                  values="value", aggfunc="first").reindex(columns=modelli)
        if not pv.empty:
            blocchi.append(("verbalized", pv, None, None))

    # Anche le righe verbalized del paper restano come "non incluso": le nostre
    # due varianti calcolate (numerica e verbale) non corrispondono una a una
    # alle sei del paper, e sostituirle in silenzio renderebbe la figura non
    # confrontabile riga per riga con l'originale -- che e' tutto il punto di
    # replicarla.
    esclusi = righe_figura[righe_figura["status"] == "escluso"]["paper_label"].tolist()

    n_righe = sum(len(b[1]) for b in blocchi) + len(esclusi)
    fig, ax = plt.subplots(figsize=(12, max(7, n_righe * 0.42)))

    # barh cresce verso l'alto, quindi si disegna dal basso: prima gli esclusi,
    # poi i verbalized, poi gli inclusi. Letta dall'alto la figura mostra i
    # metodi migliori per primi e le righe vuote in fondo, che e' l'ordine in
    # cui la si guarda.
    etichette, y_corrente = [], 0
    separatori = []
    if esclusi:
        for lab in esclusi:
            etichette.append(f"{lab}  (non incluso)")
        y_corrente += len(esclusi)
        separatori.append(y_corrente - 0.5)

    for nome, pivot, el, eh in reversed(blocchi):
        if nome == "inclusi" and len(blocchi) > 1:
            separatori.append(y_corrente - 0.5)
        sotto = ax.inset_axes([0, 0, 1, 1]) if False else ax
        indici = np.arange(y_corrente, y_corrente + len(pivot))
        n_mod = len(pivot.columns)
        altezza = 0.8 / n_mod
        for j, col in enumerate(pivot.columns):
            offset = (j - (n_mod - 1) / 2) * altezza
            xerr = None
            if el is not None:
                xerr = np.vstack([np.nan_to_num(el[col].to_numpy(dtype=float)),
                                  np.nan_to_num(eh[col].to_numpy(dtype=float))])
            sotto.barh(indici + offset, pivot[col].to_numpy(dtype=float),
                       height=altezza, xerr=xerr,
                       error_kw={"elinewidth": 0.6, "ecolor": "0.35"},
                       label=col if nome == "inclusi" else None,
                       hatch="//" if nome == "verbalized" else None,
                       color=f"C{j}")
        etichette.extend(pivot.index.tolist())
        y_corrente += len(pivot)

    for s in separatori:
        ax.axhline(s, color="0.4", linewidth=1.0, linestyle=":")

    ax.set_yticks(np.arange(len(etichette)))
    ax.set_yticklabels(etichette, fontsize=7.5)
    for tick, lab in zip(ax.get_yticklabels(), etichette):
        if "(non incluso)" in lab:
            tick.set_color("0.55")
    ax.set_ylim(-0.7, len(etichette) - 0.3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean PRR (0 = casuale, 1 = oracolo; media sui dataset di selective QA)")
    ax.set_title(titolo, fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    add_note(fig,
             "Le righe sono quelle della figura corrispondente del paper. Tratteggiate: metodi "
             "verbalized, che girano con un prompt e un\nmax_new_tokens diversi -- la richiesta di "
             "confidenza peggiora le risposte fino a -0.45 di accuracy, quindi le loro barre non "
             "sono\nconfrontabili una a una con le altre (vedi fig_verbalized_accuracy_cost.png). "
             "In grigio i metodi del paper che non calcoliamo:\nil motivo per ciascuno e' in "
             "excluded_methods.md.", left=0.34)
    save(fig, results_dir, out_name)


def fig_severity_grid(results_dir):
    """Griglia 2x2 severita' clinica x formato della risposta."""
    raw = _read_mapped(results_dir, "results_severity_grid_mapped.csv")
    if raw is None:
        print("  results_severity_grid_mapped.csv assente, salto la griglia severita'.")
        return
    posizioni = {("alta", "MCQ"): "MedQAbstain-LT", ("bassa", "MCQ"): "MedQAbstain-Safe",
                 ("alta", "libera"): "MedicationQA", ("bassa", "libera"): "MedQuAD"}
    presenti = set(raw["dataset"])
    modelli = _ordina_modelli(raw["model"])

    ordine = (raw.groupby("paper_label")["value"].mean().sort_values().index.tolist())
    fig, axes = plt.subplots(2, 2, figsize=(17, max(9, len(ordine) * 0.45)), sharey=True)
    for r, sev in enumerate(["alta", "bassa"]):
        for c, fmt in enumerate(["MCQ", "libera"]):
            ax = axes[r][c]
            ds = posizioni[(sev, fmt)]
            if ds not in presenti:
                ax.set_visible(False)
                continue
            sub = raw[raw["dataset"] == ds]
            pivot = sub.pivot_table(index="paper_label", columns="model", values="value",
                                    aggfunc="first").reindex(index=ordine, columns=modelli)
            acc = sub.groupby("model")["mean_quality"].first() if "mean_quality" in sub else None
            pivot.plot(kind="barh", ax=ax, width=0.8, legend=False)
            titolo = f"{ds} -- severita' {sev}, formato {fmt}"
            if acc is not None and len(acc):
                titolo += "\naccuracy: " + ", ".join(f"{m} {acc.get(m, float('nan')):.2f}"
                                                     for m in modelli if m in acc.index)
            ax.set_title(titolo, fontsize=7.5)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel(PRR_AXIS, fontsize=8)
            ax.tick_params(labelsize=7)

    # Asse x condiviso per colonna: dentro una colonna il formato e' lo stesso,
    # quindi la metrica di qualita' e' la stessa e le scale sono commensurabili.
    # Fra colonne no: AccuracyMetric e' binaria, AlignScore e' continua.
    for c in range(2):
        col_axes = [axes[r][c] for r in range(2) if axes[r][c].get_visible()]
        if len(col_axes) == 2:
            lo = min(a.get_xlim()[0] for a in col_axes)
            hi = max(a.get_xlim()[1] for a in col_axes)
            for a in col_axes:
                a.set_xlim(lo, hi)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][-1].legend(handles, labels, loc="lower right", fontsize=7)
    fig.suptitle("PRR per metodo UQ: griglia severita' clinica x formato della risposta")
    add_note(fig,
             "Il confronto che isola la severita' e' VERTICALE dentro una colonna: stesso formato, "
             "stessa metrica di correttezza.\nFra colonne no, perche' accuracy binaria e AlignScore "
             "continuo non sono commensurabili. Le domande dei due strati sono DIVERSE: se quelle "
             "pericolose sono anche piu' difficili,\nuna differenza di PRR non e' attribuibile alla "
             "severita' (vedi fig_severity_matched.png, stesso confronto a parita' di difficolta').",
             top=0.9, wspace=0.05, hspace=0.25, left=0.2)
    save(fig, results_dir, "fig_severity_grid.png")


def fig_quant_comparison(results_dir):
    """Quantizzato 4-bit contro bf16, sullo stesso modello."""
    raw = _read_mapped(results_dir, "results_quant_comparison_mapped.csv")
    if raw is None:
        print("  results_quant_comparison_mapped.csv assente, salto il confronto quantizzazione.")
        return
    agg = aggregate_across_datasets(raw)
    varianti = sorted(agg["model"].unique())
    pivot = agg.pivot_table(index="paper_label", columns="model", values="value",
                            aggfunc="first").reindex(columns=varianti)
    if pivot.empty:
        return
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    lo = agg.pivot_table(index="paper_label", columns="model", values="prr_ci_low",
                         aggfunc="first").reindex(index=pivot.index, columns=varianti)
    hi = agg.pivot_table(index="paper_label", columns="model", values="prr_ci_high",
                         aggfunc="first").reindex(index=pivot.index, columns=varianti)

    fig, ax = plt.subplots(figsize=(11, max(7, len(pivot) * 0.42)))
    _barh_con_ci(ax, pivot, (pivot - lo).clip(lower=0), (hi - pivot).clip(lower=0))
    ax.set_xlabel("Mean PRR (0 = casuale, 1 = oracolo; media sui dataset)")
    ax.set_title("Effetto della quantizzazione sulla qualita' delle stime di incertezza")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=8, loc="lower right")
    add_note(fig,
             "Un metodo peggiora davvero con la quantizzazione solo se il suo intervallo di confidenza "
             "non si sovrappone a quello\ndell'altra variante. Per un confronto piu' stringente serve "
             "il test appaiato (paired_comparisons.py), perche' le due varianti\nsono valutate sulle "
             "stesse domande.", left=0.34)
    save(fig, results_dir, "fig_quant_comparison.png")


def fig_timing_full_cost(results_dir):
    """Costo pieno standalone per istanza, accanto al costo marginale.

    E' la figura che il TODO 6 chiede e che mancava. Il grafico dei tempi
    esistente misura solo l'aritmetica dello stimatore su statistiche gia'
    pronte: risponde a "quanto costa aggiungere questa tecnica se sto gia'
    calcolando tutto il resto?". La domanda del deployment on-device e'
    l'opposta -- "quanto costa se e' l'unica che faccio girare?" -- e la
    risposta include la generazione greedy, i K campioni se servono e i forward
    del modello NLI se servono.
    """
    path = os.path.join(results_dir, "estimator_timings.csv")
    if not os.path.exists(path):
        print("  estimator_timings.csv assente, salto la figura dei costi.")
        return
    t = _drop_phase_rows(pd.read_csv(path))
    attese = {"seconds_full_per_instance", "seconds_marginal_per_instance", "paper_label"}
    if not attese.issubset(t.columns):
        print(f"  estimator_timings.csv ha uno schema vecchio (mancano "
              f"{sorted(attese - set(t.columns))}), salto la figura dei costi.")
        return

    costo = t.groupby("paper_label", as_index=False).agg(
        pieno=("seconds_full_per_instance", "mean"),
        marginale=("seconds_marginal_per_instance", "mean"),
        campionamento=("needs_sampling", "max"),
        nli=("needs_nli", "max"),
    ).sort_values("pieno")

    y = np.arange(len(costo))
    fig, ax = plt.subplots(figsize=(11, max(6, len(costo) * 0.4)))
    ax.barh(y + 0.2, costo["pieno"], height=0.38, color="tab:blue",
            label="costo pieno standalone (greedy + tutti i calcolatori da cui dipende)")
    ax.barh(y - 0.2, costo["marginale"], height=0.38, color="tab:orange",
            label="costo marginale (sola aritmetica)")
    ax.set_yticks(y)
    etichette = [f"{r.paper_label}"
                 + ("  [K campioni]" if r.campionamento else "")
                 + ("  [NLI]" if r.nli else "")
                 for r in costo.itertuples()]
    ax.set_yticklabels(etichette, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("Secondi per istanza (scala log)")
    ax.set_title("Costo per metodo UQ: pieno standalone contro marginale")
    ax.legend(fontsize=8, loc="lower right")
    add_note(fig,
             "La distanza fra le due barre e' il costo che il metodo si fa pagare dal magazzino "
             "condiviso: enorme per i metodi a\ndiversita' campionaria, nulla per quelli a passaggio "
             "singolo. Per la scelta on-device conta la barra blu -- e' quella\nusata anche sull'asse "
             "x della frontiera di Pareto.", left=0.36)
    save(fig, results_dir, "fig_timing_full_cost.png")


def _righe_di_testo(celle):
    """Numero massimo di righe di testo fra le celle date."""
    celle = [str(c) for c in celle]
    return max(c.count("\n") + 1 for c in celle) if celle else 1


def _tabella_immagine(fig, ax, df, col_widths=None, fontsize=8):
    ax.axis("off")
    tab = ax.table(cellText=df.values.tolist(), colLabels=df.columns.tolist(),
                   cellLoc="center", loc="center", colWidths=col_widths)
    tab.auto_set_font_size(False)
    tab.set_fontsize(fontsize)
    # L'altezza delle righe cresce con il numero di righe di testo per cella:
    # con un fattore fisso le celle su due righe (valore + metrica) si
    # sovrapponevano a quelle vicine.
    righe_corpo = _righe_di_testo(df.values.ravel())
    righe_intest = _righe_di_testo(df.columns)
    tab.scale(1, 1.45 * righe_corpo)
    for (r, c), cell in tab.get_celld().items():
        if r == 0 and righe_intest != righe_corpo:
            cell.set_height(cell.get_height() * righe_intest / righe_corpo)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
    return tab


def table_accuracy(stats, results_dir):
    """Tabella accuracy con la metrica dichiarata per ogni cella."""
    cell = stats.groupby(["model", "dataset"], as_index=False).agg(
        acc=("mean_quality", "first"), metrica=("quality_metric", "first"))
    if cell.empty:
        return
    cell["testo"] = cell.apply(
        lambda r: f"{r['acc']:.3f}\n({r['metrica']})" if pd.notna(r["acc"]) else "-", axis=1)
    tabella = cell.pivot(index="model", columns="dataset", values="testo").fillna("-")
    tabella = tabella.reindex(_ordina_modelli(tabella.index))
    df = tabella.reset_index().rename(columns={"model": "modello"})

    fig, ax = plt.subplots(figsize=(2.1 * len(df.columns), 1.4 + 0.75 * len(df)))
    _tabella_immagine(fig, ax, df)
    ax.set_title("Qualita' di base per modello e dataset, con la metrica usata",
                 fontsize=11, pad=16)
    add_note(fig,
             "Su MCQ a quattro opzioni il livello del caso e' 0.25: un valore sotto non indica "
             "ignoranza del modello ma un problema\ndi estrazione della risposta. AlignScore e' "
             "continuo e severo sul testo libero medico: valori bassi li' significano che\nquasi ogni "
             "risposta conta come sbagliata, e il PRR di quelle celle ha intervalli di confidenza "
             "larghi.")
    save(fig, results_dir, "fig_tabella_accuracy.png")


def table_prr_vs_accuracy(stats, results_dir):
    """PRR affiancato all'accuracy della stessa cella, con l'intervallo di
    confidenza del metodo migliore al posto delle vecchie "bande di regime".

    Le bande (accuracy fra 0.30 e 0.85 = interpretabile) erano una conseguenza
    dell'area grezza, che dipende meccanicamente dall'accuracy, e le soglie non
    avevano una giustificazione. Con il PRR normalizzato quello che resta vero
    e' che con pochissime risposte giuste o sbagliate la stima e' rumorosa: lo
    dice l'ampiezza dell'intervallo, e la tabella riporta il numero di
    risposte della classe minoritaria."""
    cell = stats.groupby(["model", "dataset"], as_index=False).agg(
        accuracy=("mean_quality", "first"),
        metrica=("quality_metric", "first"),
        n=("n_instances", "first"),
        prr_mediano=("prr", "median"),
    ).dropna(subset=["accuracy"])
    if cell.empty:
        return
    idx = stats.dropna(subset=["prr"]).groupby(["model", "dataset"])["prr"].idxmax()
    migliori = stats.loc[idx, ["model", "dataset", "paper_label", "prr", "prr_ci_low",
                               "prr_ci_high"]].rename(columns={"paper_label": "metodo_migliore",
                                                               "prr": "prr_migliore"})
    cell = cell.merge(migliori, on=["model", "dataset"], how="left")
    err_col = stats.groupby(["model", "dataset"])["error_rate_overall"].first() \
        if "error_rate_overall" in stats.columns else None
    if err_col is not None:
        cell["minoritaria"] = [int(round(min(e, 1 - e) * n)) if pd.notna(e) and pd.notna(n) else -1
                               for e, n in zip(err_col.reindex(list(zip(cell["model"], cell["dataset"]))).values,
                                               cell["n"])]
    else:
        cell["minoritaria"] = -1
    cell = cell.sort_values(["model", "dataset"])

    def _ci(r):
        if pd.isna(r["prr_ci_low"]):
            return "-"
        return f"[{r['prr_ci_low']:.2f}, {r['prr_ci_high']:.2f}]"

    df = pd.DataFrame({
        "modello": cell["model"],
        "dataset": cell["dataset"],
        "accuracy": cell["accuracy"].map("{:.3f}".format),
        "metrica": cell["metrica"],
        "PRR mediano": cell["prr_mediano"].map("{:.3f}".format),
        "PRR migliore": cell["prr_migliore"].map("{:.3f}".format),
        "IC 95% migliore": cell.apply(_ci, axis=1),
        "metodo migliore": cell["metodo_migliore"].fillna("-"),
        "classe minoritaria": cell["minoritaria"].map(lambda v: "-" if v < 0 else str(v)),
    })
    larghezze = [0.12, 0.11, 0.07, 0.10, 0.08, 0.08, 0.11, 0.24, 0.09]
    fig, ax = plt.subplots(figsize=(16, 0.6 + 0.30 * len(df)))
    _tabella_immagine(fig, ax, df, col_widths=larghezze, fontsize=7.5)
    ax.set_title("PRR affiancato all'accuracy della stessa cella", fontsize=11, pad=16)
    add_note(fig,
             "PRR del paper: 0 = come scartare a caso, 1 = come un oracolo. 'Classe minoritaria' = "
             "numero di risposte giuste o sbagliate, quale delle due e' piu' rara:\ncon poche "
             "decine di casi il PRR e' rumoroso, come mostra l'intervallo di confidenza. Il "
             "metodo migliore e' quello con il PRR piu' alto sui dati completi,\nche non vuol dire "
             "distinguibile dagli altri: per quello vedi paired_comparisons e fig_ties_vs_scale.png.")
    save(fig, results_dir, "fig_tabella_prr_accuracy.png")


def fig_prr_vs_scale_by_family(results_dir):
    """PRR medio contro scala del modello, una curva per famiglia di metodi.

    Completa l'artefatto del TODO 1: il Kendall tau dice se il ranking dei
    metodi si conserva scendendo di scala, questa dice se il LIVELLO di
    affidabilita' si conserva, e se qualche famiglia regge meglio delle altre.
    Sono due domande diverse: il ranking puo' restare identico mentre tutti i
    valori crollano, e viceversa.
    """
    raw = _read_mapped(results_dir, "results_paper_mapped.csv")
    if raw is None:
        print("  results_paper_mapped.csv assente, salto PRR-vs-scala.")
        return
    df = raw.copy()
    df["famiglia"] = df["paper_label"].map(family_of)
    df["params_B"] = df["model"].map(MODEL_PARAMS_B)
    df = df.dropna(subset=["params_B", "value"])
    df = df[df["famiglia"] != "non classificato"]
    if df.empty:
        return

    curve = df.groupby(["famiglia", "params_B"], as_index=False).agg(
        prr=("value", "mean"), sd=("value", "std"), n=("value", "size"))

    fig, ax = plt.subplots(figsize=(9, 6))
    for famiglia in FAMILY_ORDER:
        sub = curve[curve["famiglia"] == famiglia].sort_values("params_B")
        if sub.empty:
            continue
        ax.errorbar(sub["params_B"], sub["prr"], yerr=sub["sd"].fillna(0),
                    marker="o", capsize=3, linestyle="none", label=famiglia)
    ax.set_xscale("log")
    ax.set_xlabel("Parametri del modello (miliardi, scala log)")
    ax.set_ylabel("Mean PRR (media sui metodi della famiglia e sui dataset)")
    ax.set_title("Affidabilita' per famiglia di metodi, al variare della scala")
    ax.legend(fontsize=8)
    ax.grid(linewidth=0.3, alpha=0.5)
    add_note(fig,
             "Le barre verticali sono la dispersione fra i metodi della stessa famiglia, non un "
             "intervallo di confidenza.\nLe famiglie seguono la ripartizione standard della "
             "letteratura, da confrontare con la Sezione 3 del paper: vedi\nMETHOD_FAMILIES in "
             "analysis_lib.py. Solo punti, niente linee: i modelli appartengono a famiglie diverse "
             "(LFM2, Gemma, Mistral)\ne non stanno su una stessa curva di scala.")
    save(fig, results_dir, "fig_prr_vs_scale_by_family.png")


def fig_rank_transfer(stats, results_dir):
    """Kendall tau della classifica dei metodi contro il modello-ancora, un
    punto per dataset, con intervallo bootstrap appaiato sulle domande (da
    rank_transfer_kendall_tau.csv, scritto da main.py o da recompute_stats.py)."""
    path = os.path.join(results_dir, "rank_transfer_kendall_tau.csv")
    if not os.path.exists(path):
        print("  rank_transfer_kendall_tau.csv assente, salto il rank transfer.")
        return
    tau = pd.read_csv(path)
    if "dataset" not in tau.columns:
        print("  rank_transfer_kendall_tau.csv ha lo schema vecchio (tau sulla media fra dataset): "
              "rilanciare recompute_stats.py.")
        return
    modelli = _ordina_modelli(tau["model"])
    datasets = sorted(tau["dataset"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    larghezza = 0.8 / max(len(datasets), 1)
    for j, ds in enumerate(datasets):
        sub = tau[tau["dataset"] == ds].set_index("model").reindex(modelli)
        x = np.arange(len(modelli)) + (j - (len(datasets) - 1) / 2) * larghezza
        err = None
        if {"tau_ci_low", "tau_ci_high"}.issubset(sub.columns):
            err = np.vstack([(sub["kendall_tau_vs_anchor"] - sub["tau_ci_low"]).clip(lower=0).fillna(0),
                             (sub["tau_ci_high"] - sub["kendall_tau_vs_anchor"]).clip(lower=0).fillna(0)])
        ax.errorbar(x, sub["kendall_tau_vs_anchor"], yerr=err, fmt="o", capsize=3,
                    color=f"C{j}", label=ds)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="0.7", linewidth=0.8, linestyle=":")
    ax.set_xticks(np.arange(len(modelli)))
    ax.set_xticklabels([f"{m}\n({MODEL_PARAMS_B.get(m, '?')}B)" for m in modelli], fontsize=8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(f"Kendall tau della classifica dei metodi vs {ANCHOR_MODEL}")
    ax.set_title("Trasferimento della classifica dei metodi UQ, dataset per dataset")
    ax.legend(fontsize=8, title="dataset")
    add_note(fig,
             "Un punto per dataset; barre = intervallo bootstrap al 95% ricampionando le domande "
             "(le due classifiche sono calcolate sulle stesse domande),\nche tiene conto della "
             "dipendenza fra metodi che condividono gli stessi dati. Niente linee fra modelli: sono "
             "famiglie diverse.\nSe a una scala i metodi sono indistinguibili fra loro (vedi "
             "fig_ties_vs_scale.png), un tau vicino a zero e' garantito per costruzione.")
    save(fig, results_dir, "fig_rank_transfer.png")


def fig_pareto(results_dir):
    """Frontiera di Pareto costo pieno per istanza contro affidabilita'."""
    path = os.path.join(results_dir, "estimator_timings.csv")
    raw = _read_mapped(results_dir, "results_paper_mapped.csv")
    if not os.path.exists(path) or raw is None:
        print("  Dati insufficienti per la frontiera di Pareto, salto.")
        return
    t = _drop_phase_rows(pd.read_csv(path))
    if "seconds_full_per_instance" not in t.columns:
        print("  estimator_timings.csv ha uno schema vecchio, salto la Pareto.")
        return
    costo = t.groupby("paper_label", as_index=False)["seconds_full_per_instance"].mean()
    agg = aggregate_across_datasets(raw)
    # Intervallo al 95% della media sui modelli, combinando gli errori standard
    # dei singoli modelli come indipendenti (stessa regola di
    # aggregate_across_datasets).
    if {"prr_ci_low", "prr_ci_high"}.issubset(agg.columns):
        agg = agg.assign(_se2=((agg["prr_ci_high"] - agg["prr_ci_low"]) / (2 * 1.96)) ** 2)
        qualita = agg.groupby("paper_label", as_index=False).agg(
            value=("value", "mean"), _se2=("_se2", "sum"), _k=("value", "size"))
        qualita["ci"] = 1.96 * np.sqrt(qualita["_se2"]) / qualita["_k"]
        qualita = qualita.drop(columns=["_se2", "_k"])
    else:
        qualita = agg.groupby("paper_label", as_index=False)["value"].mean()
        qualita["ci"] = np.nan
    punti = qualita.merge(costo, on="paper_label", how="inner").dropna(
        subset=["value", "seconds_full_per_instance"])
    if punti.empty:
        return
    punti = punti.sort_values("seconds_full_per_instance")

    frontiera_idx, migliore = [], -np.inf
    for idx, row in punti.iterrows():
        if row["value"] > migliore:
            frontiera_idx.append(idx)
            migliore = row["value"]
    frontiera = punti.loc[frontiera_idx]

    fig, ax = plt.subplots(figsize=(11, 7))
    if punti["ci"].notna().any():
        ax.errorbar(punti["seconds_full_per_instance"], punti["value"], yerr=punti["ci"],
                    fmt="none", ecolor="0.75", elinewidth=0.8, capsize=2, zorder=1,
                    label="intervallo di confidenza al 95%")
    ax.scatter(punti["seconds_full_per_instance"], punti["value"], s=34,
               color="0.6", label="metodi dominati", zorder=2)
    ax.scatter(frontiera["seconds_full_per_instance"], frontiera["value"], s=68,
               color="tab:red", label="frontiera di Pareto", zorder=3)
    ax.plot(frontiera["seconds_full_per_instance"], frontiera["value"],
            color="tab:red", linewidth=1.0, linestyle="--", zorder=1)
    for _, r in frontiera.iterrows():
        ax.annotate(r["paper_label"], (r["seconds_full_per_instance"], r["value"]),
                    textcoords="offset points", xytext=(7, -3), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Costo pieno standalone per istanza (secondi, scala log)")
    ax.set_ylabel("Mean PRR")
    ax.set_title("Frontiera di Pareto: quanto costa l'affidabilita'")
    ax.legend(fontsize=8)
    ax.grid(linewidth=0.3, alpha=0.5)
    add_note(fig,
             "Un metodo e' dominato se ne esiste un altro insieme piu' economico e piu' affidabile: "
             "sceglierlo non e' mai razionale.\nLa frontiera dice, per ogni budget di calcolo, la "
             "migliore affidabilita' raggiungibile e con quale metodo -- e' la figura\nche risponde "
             "alla domanda del deployment on-device. Le barre grigie sono intervalli al 95%: due "
             "metodi le cui barre si sovrappongono\nlargamente non sono distinguibili, e la "
             "frontiera fra loro va letta come indicativa, non come una classifica."
             + ("" if "needs_cross_encoder" in t.columns else
                "\nATTENZIONE: costi prodotti prima della correzione del calcolo delle dipendenze: "
                "CCP, TokenSAR, SAR, SentenceSAR,\nLabel Prob., PMI e P(True) risultano piu' "
                "economici di quanto sono (modelli ausiliari e forward extra non attribuiti)."))
    save(fig, results_dir, "fig_pareto_cost_quality.png")


def table_costi(results_dir):
    """Tabella costi per metodo: tempo marginale, tempo pieno, memoria di picco."""
    path = os.path.join(results_dir, "estimator_timings.csv")
    if not os.path.exists(path):
        return
    t = _drop_phase_rows(pd.read_csv(path))
    attese = {"seconds_marginal_per_instance", "seconds_full_per_instance", "peak_memory_gb"}
    if not attese.issubset(t.columns):
        print("  estimator_timings.csv senza le colonne di costo, salto la tabella costi.")
        return
    extra = [c for c in ("needs_cross_encoder", "needs_extra_forward") if c in t.columns]
    costo = t.groupby("paper_label", as_index=False).agg(
        marginale=("seconds_marginal_per_instance", "mean"),
        pieno=("seconds_full_per_instance", "mean"),
        campioni=("needs_sampling", "max"),
        nli=("needs_nli", "max"),
        **{c: (c, "max") for c in extra},
    ).sort_values("pieno", ascending=False)

    # La memoria di picco NON compare: e' misurata sull'intera cella modello x
    # dataset, quindi e' identica per tutti i metodi e non dice nulla sul
    # singolo metodo (resta in estimator_cost_table.csv come dato per modello).
    colonne = {
        "metodo": costo["paper_label"],
        "famiglia": costo["paper_label"].map(family_of),
        "s/istanza (marginale)": costo["marginale"].map("{:.4f}".format),
        "s/istanza (pieno)": costo["pieno"].map("{:.3f}".format),
        "K campioni": np.where(costo["campioni"], "si", "-"),
        "NLI": np.where(costo["nli"], "si", "-"),
    }
    larghezze = [0.28, 0.16, 0.13, 0.12, 0.09, 0.07]
    if "needs_cross_encoder" in costo:
        colonne["cross-encoder"] = np.where(costo["needs_cross_encoder"], "si", "-")
        larghezze.append(0.10)
    if "needs_extra_forward" in costo:
        colonne["forward extra"] = np.where(costo["needs_extra_forward"], "si", "-")
        larghezze.append(0.10)
    df = pd.DataFrame(colonne)
    fig, ax = plt.subplots(figsize=(13, 0.6 + 0.30 * len(df)))
    _tabella_immagine(fig, ax, df, col_widths=larghezze, fontsize=7.5)
    ax.set_title("Costo per metodo UQ: tempo marginale e tempo pieno standalone",
                 fontsize=11, pad=16)
    nota_extra = ("" if extra else
                  "\nATTENZIONE: risultati prodotti prima della correzione del calcolo delle "
                  "dipendenze: il costo pieno di CCP, TokenSAR,\nSAR, SentenceSAR, Label Prob., PMI e "
                  "P(True) e' sottostimato (modelli ausiliari e forward extra non attribuiti).")
    add_note(fig,
             "Il tempo marginale e' la sola aritmetica su statistiche gia' pronte, utile a chi "
             "calcola molte tecniche insieme. Il tempo pieno\ne' quello che conta per il deployment "
             "on-device: somma i tempi misurati di tutti i calcolatori da cui il metodo dipende\n"
             "(generazione greedy, K campioni, modello NLI, cross-encoder, forward extra del modello)."
             + nota_extra)
    save(fig, results_dir, "fig_tabella_costi.png")


def table_silent_failure(results_dir):
    """Errori fra le risposte date con piu' fiducia, per metodo, dataset della
    griglia e modello, accanto all'error rate complessivo della cella.

    Sostituisce la vecchia tabella del silent failure rate ("quale frazione
    degli errori finisce nel 10% piu' confidente"), che ha un tetto: con il 90%
    di errori, come sui dataset clinici a risposta libera, vale al massimo ~0.11
    e il caso da' 0.10, quindi non distingueva nulla proprio li'. La nuova
    lettura: "fra le risposte date con piu' sicurezza, quante sono sbagliate",
    da confrontare con quante sono sbagliate in tutto."""
    path = os.path.join(results_dir, "results_severity_grid_instance_stats.csv")
    if not os.path.exists(path):
        print("  results_severity_grid_instance_stats.csv assente, salto la tabella degli errori.")
        return
    st = pd.read_csv(path)
    if "error_rate_top10" not in st.columns or st.empty:
        print("  statistiche senza error_rate_top10 (run precedente al 29/09): rilanciare "
              "recompute_stats.py.")
        return
    abbrev = {"MedQAbstain-LT": "LT", "MedQAbstain-Safe": "Safe",
              "MedicationQA": "MedicationQA", "MedQuAD": "MedQuAD"}
    ordine_ds = [d for d in abbrev if d in set(st["dataset"])]
    ordine_m = _ordina_modelli(st["model"])
    colonne = [(d, m) for d in ordine_ds for m in ordine_m
               if ((st["dataset"] == d) & (st["model"] == m)).any()]
    piv = st.pivot_table(index="paper_label", columns=["dataset", "model"],
                         values="error_rate_top10", aggfunc="first")
    colonne = [c for c in colonne if c in piv.columns and not piv[c].isna().all()]
    if not colonne:
        return
    piv = piv.reindex(columns=pd.MultiIndex.from_tuples(colonne))
    overall = st.groupby(["dataset", "model"])["error_rate_overall"].first()
    piv = piv.loc[piv.mean(axis=1).sort_values().index]

    intest = [f"{abbrev[d]}\n{m.replace('-it', '')}\n(tutte: {overall.get((d, m), np.nan):.2f})"
              for d, m in colonne]
    testo = piv.apply(lambda col: col.map(lambda v: "-" if pd.isna(v) else f"{v:.2f}"))
    df = pd.DataFrame(testo.values, columns=intest)
    df.insert(0, "metodo", piv.index)

    fig, ax = plt.subplots(figsize=(max(12, 1.05 * len(df.columns) + 3), 1.9 + 0.30 * len(df)))
    larghezze = [0.25] + [0.75 / len(colonne)] * len(colonne)
    tab = _tabella_immagine(fig, ax, df, col_widths=larghezze, fontsize=6.5)
    for (r, c), cell in tab.get_celld().items():
        if r == 0 or c == 0:
            continue
        v = piv.iloc[r - 1, c - 1]
        base = overall.get(colonne[c - 1], np.nan)
        if pd.isna(v) or pd.isna(base):
            continue
        if v <= 0.5 * base:
            cell.set_facecolor("#dff0d8")
        elif v >= base:
            cell.set_facecolor("#f8d7da")
    ax.set_title("Errori fra le risposte date con piu' fiducia (10% piu' confidente), per strato "
                 "di severita'", fontsize=11, pad=16)
    add_note(fig,
             "Ogni cella: frazione di risposte SBAGLIATE fra il 10% su cui il metodo e' piu' "
             "sicuro. In intestazione, fra parentesi, la frazione di risposte sbagliate in tutto.\n"
             "Un metodo che non sa nulla da' in media l'error rate complessivo; uno utile, meno. "
             "Verde: al massimo meta' dell'error rate complessivo. Rosso: non meglio\ndel caso. "
             "'-' = metodo con punteggio costante (non ordina nulla). Pareggi sul bordo del 10% "
             "risolti in valore atteso.")
    save(fig, results_dir, "fig_tabella_silent_failure.png")


def fig_severity_matched(results_dir):
    """Severita' a parita' di difficolta' (vedi analysis_lib.severity_matched):
    per ogni coppia di strati (LT/Safe, MedicationQA/MedQuAD) e ogni modello, il
    PRR mediano fra i metodi sui dati completi e sui sottoinsiemi con la stessa
    distribuzione di difficolta'. Scrive anche severity_matched_difficulty.csv."""
    path = os.path.join(results_dir, "results_severity_grid_per_instance.csv")
    if not os.path.exists(path):
        print("  results_severity_grid_per_instance.csv assente, salto la severita' a parita' "
              "di difficolta'.")
        return
    per = pd.read_csv(path)
    coppie = [("MedQAbstain-LT", "MedQAbstain-Safe"), ("MedicationQA", "MedQuAD")]
    frames = [severity_matched(per, a, b) for a, b in coppie
              if a in set(per["dataset"]) and b in set(per["dataset"])]
    frames = [f for f in frames if not f.empty]
    if not frames:
        print("  severita' a parita' di difficolta': servono almeno tre modelli su entrambi gli "
              "strati, salto.")
        return
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(os.path.join(results_dir, "severity_matched_difficulty.csv"), index=False)
    print(f"Salvato: {os.path.join(results_dir, 'severity_matched_difficulty.csv')}")

    agg = res.groupby(["dataset_a", "dataset_b", "model"], as_index=False).agg(
        a_full=("prr_a_full", "median"), b_full=("prr_b_full", "median"),
        a_matched=("prr_a_matched", "median"), b_matched=("prr_b_matched", "median"),
        acc_a_full=("acc_a_full", "first"), acc_b_full=("acc_b_full", "first"),
        acc_a_matched=("acc_a_matched", "first"), acc_b_matched=("acc_b_matched", "first"))
    n_coppie = agg[["dataset_a", "dataset_b"]].drop_duplicates()
    fig, axes = plt.subplots(1, len(n_coppie), figsize=(8 * len(n_coppie), 6), squeeze=False)
    for ax, (_, cp) in zip(axes[0], n_coppie.iterrows()):
        sub = agg[(agg["dataset_a"] == cp["dataset_a"]) & (agg["dataset_b"] == cp["dataset_b"])]
        modelli = _ordina_modelli(sub["model"])
        sub = sub.set_index("model").reindex(modelli)
        x = np.arange(len(modelli))
        for off, col, lab, mk, colr in ((-0.15, "a_full", f"{cp['dataset_a']} (tutte)", "o", "tab:red"),
                                        (-0.05, "a_matched", f"{cp['dataset_a']} (stessa difficolta')", "s", "tab:red"),
                                        (0.05, "b_full", f"{cp['dataset_b']} (tutte)", "o", "tab:blue"),
                                        (0.15, "b_matched", f"{cp['dataset_b']} (stessa difficolta')", "s", "tab:blue")):
            ax.scatter(x + off, sub[col], marker=mk, color=colr, s=45, label=lab,
                       facecolors="none" if "matched" in col else colr)
        for i, m in enumerate(modelli):
            r = sub.loc[m]
            ax.annotate(f"acc {r['acc_a_full']:.2f}/{r['acc_b_full']:.2f}\n"
                        f"pari diff. {r['acc_a_matched']:.2f}/{r['acc_b_matched']:.2f}",
                        (i, np.nanmin([r["a_full"], r["b_full"], r["a_matched"], r["b_matched"]])),
                        textcoords="offset points", xytext=(0, -26), ha="center", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(modelli, fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("PRR mediano fra i metodi")
        ax.set_title(f"{cp['dataset_a']} contro {cp['dataset_b']}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    fig.suptitle("Severita' clinica a parita' di difficolta' delle domande")
    add_note(fig,
             "Difficolta' di una domanda = quanti ALTRI modelli la azzeccano (escluso quello "
             "valutato, per non rendere l'analisi circolare). Per ogni livello si tengono tante "
             "domande\nquante ne ha lo strato piu' povero (sottocampionamento ripetuto 50 volte). "
             "Se la differenza fra gli strati resta anche a parita' di difficolta', e' "
             "attribuibile alla severita';\nse sparisce, era la difficolta'. Sotto ogni modello: "
             "accuracy dei due strati, su tutte le domande e a parita' di difficolta'.", top=0.88)
    save(fig, results_dir, "fig_severity_matched.png")


def fig_phase_timings(results_dir):
    """Tempo per fase (generazione greedy, K campioni, NLI, cross-encoder,
    metrica di qualita', ...) per modello e dataset, da phase_timings.csv."""
    path = os.path.join(results_dir, "phase_timings.csv")
    if not os.path.exists(path):
        print("  phase_timings.csv assente (run precedente al 29/09), salto i tempi per fase.")
        return
    ph = pd.read_csv(path)
    ph["cella"] = ph["model"] + " / " + ph["dataset"]
    piv = ph.pivot_table(index="cella", columns="phase", values="seconds_per_instance",
                         aggfunc="sum").fillna(0)
    piv = piv.loc[piv.sum(axis=1).sort_values().index]
    piv = piv[piv.sum().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(piv))))
    piv.plot(kind="barh", stacked=True, ax=ax, width=0.8)
    ax.set_xlabel("Secondi per istanza")
    ax.set_ylabel("")
    ax.set_title("Dove va il tempo: fasi del calcolo per modello e dataset")
    ax.legend(fontsize=7, loc="lower right")
    add_note(fig,
             "Somma su tutti i metodi della cella: ogni fase e' calcolata una volta e condivisa. "
             "'metrica_qualita' e' il calcolo della correttezza (AlignScore o\nestrazione della "
             "lettera), che sul telefono non esiste. 'generazione_interna_stimatore' e' BB P(True), "
             "che genera da se' le proprie risposte.")
    save(fig, results_dir, "fig_phase_timings.png")


def _tabella_da_csv(results_dir, filename, titolo, out_name, nota, indice=None):
    """Rende leggibile come immagine una tabella gia' salvata come CSV."""
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        print(f"  {filename} assente, salto {out_name}.")
        return
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  {filename}: illeggibile ({e}), salto.")
        return
    if df.empty:
        return
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
    df = df.astype(str)
    fig, ax = plt.subplots(figsize=(max(9, 1.7 * len(df.columns)), 0.7 + 0.30 * len(df)))
    _tabella_immagine(fig, ax, df, fontsize=7.5)
    ax.set_title(titolo, fontsize=11, pad=16)
    add_note(fig, nota)
    save(fig, results_dir, out_name)


def generate_all(results_dir):
    """Disegna tutte le figure e le tabelle-immagine a partire dai CSV di
    results_dir. E' l'UNICO punto in cui si disegna: main.py la richiama a fine
    run, e la si puo' rilanciare a mano quante volte serve."""
    args = argparse.Namespace(results_dir=results_dir)
    stats = load_stats(args.results_dir)
    if stats is None:
        print(f"!!! Nessuna statistica per-istanza in {args.results_dir}.")
        return 1
    print(f"Celle modello x dataset disponibili: "
          f"{stats.groupby(['model', 'dataset']).ngroups}")

    rd = args.results_dir
    nota_parse_failure = ("Frazione di istanze in cui la confidenza non e' estraibile dal testo generato. Va letta "
             "ACCANTO al PRR: lm-polygraph\nconverte i punteggi non estraibili in -1e7, cioe' li tratta "
             "come massima confidenza, quindi un modello che non rispetta il\nformato non viene "
             "penalizzato dal PRR ma premiato. Un parse-failure rate alto rende il PRR di quel metodo "
             "non informativo.")
    # Ogni figura gira isolata: un errore su una (dati parziali, un CSV di
    # una versione vecchia) non deve impedire di disegnare le altre.
    figure = [
        ("fig_accuracy", lambda: fig_accuracy(stats, rd)),
        ("fig_accuracy_vs_prr", lambda: fig_accuracy_vs_prr(stats, rd)),
        ("fig_anchor_replication", lambda: fig_anchor_replication(stats, rd)),
        ("fig_ties_vs_scale", lambda: fig_ties_vs_scale(rd)),
        ("fig_timing_full_cost", lambda: fig_timing_full_cost(rd)),
        ("fig_a_white_box", lambda: fig_paper_figure(
            rd, "A", "fig_a_white_box.png",
            "Mean PRR aggregato su selective QA (~ Fig. 2 Vashurin et al., white-box full-access)")),
        ("fig_b_reflexive", lambda: fig_paper_figure(
            rd, "B", "fig_b_reflexive.png",
            "Mean PRR aggregato su selective QA (~ Fig. 3 Vashurin et al., reflexive/black-box)")),
        ("fig_quant_comparison", lambda: fig_quant_comparison(rd)),
        ("fig_severity_grid", lambda: fig_severity_grid(rd)),
        ("fig_tabella_accuracy", lambda: table_accuracy(stats, rd)),
        ("fig_tabella_prr_accuracy", lambda: table_prr_vs_accuracy(stats, rd)),
        ("fig_verbalized_accuracy_cost", lambda: fig_verbalized_accuracy_cost(rd)),
        ("fig_verbalized_numeric/linguistic", lambda: fig_verbalized_by_style(rd)),
        ("estimator_timing_chart", lambda: fig_timing_marginal(rd)),
        ("fig_prr_vs_scale_by_family", lambda: fig_prr_vs_scale_by_family(rd)),
        ("fig_rank_transfer", lambda: fig_rank_transfer(stats, rd)),
        ("fig_pareto_cost_quality", lambda: fig_pareto(rd)),
        ("fig_tabella_costi", lambda: table_costi(rd)),
    ]
    for style in ("numeric", "linguistic"):
        figure.append((f"fig_tabella_parse_failure_{style}", lambda style=style: _tabella_da_csv(
            rd, f"parse_failure_rate_{style}.csv", f"Parse-failure rate, confidenza {style}",
            f"fig_tabella_parse_failure_{style}.png", nota_parse_failure)))
    figure.append(("fig_tabella_silent_failure", lambda: table_silent_failure(rd)))
    figure.append(("fig_severity_matched", lambda: fig_severity_matched(rd)))
    figure.append(("fig_phase_timings", lambda: fig_phase_timings(rd)))

    fallite = []
    for nome, disegna in figure:
        try:
            disegna()
        except Exception:
            fallite.append(nome)
            print(f"!!! {nome} fallita:")
            traceback.print_exc()
        finally:
            plt.close("all")
    if fallite:
        print(f"\n!!! Figure non prodotte ({len(fallite)}): {', '.join(fallite)}")
        return 1
    print(f"\nTutte le {len(figure)} figure prodotte.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    args = parser.parse_args()
    return generate_all(args.results_dir)


if __name__ == "__main__":
    sys.exit(main())
