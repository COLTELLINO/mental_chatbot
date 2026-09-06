"""Figure di lettura del benchmark, generate dai CSV gia' salvati.

Perche' e' uno script separato da main.py: le figure prodotte a fine run
richiedono un job GPU da ore per essere rigenerate, anche quando si vuole solo
cambiare un asse o aggiungere un pannello. Questo legge i CSV e disegna in
pochi secondi, quante volte serve.

Figure prodotte (le sole per cui esistono i dati necessari):
  fig_accuracy.png              accuracy per modello e dataset, con la soglia
                                del caso sui task a scelta multipla
  fig_accuracy_vs_prr.png       la relazione che conta davvero: il PRR e'
                                interpretabile solo a certe accuracy
  fig_anchor_replication.png    Mistral-7B contro i valori attesi dal paper
  fig_ties_vs_scale.png         quanti metodi restano indistinguibili dal
                                migliore, al variare della scala

Uso:
    python3.11 make_figures.py /workspace/results
"""
import argparse
import os
import sys

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
)

# Soglia del caso per i task a scelta multipla: sotto questa un'accuracy non
# indica ignoranza del modello ma un problema di estrazione della risposta.
CHANCE_LEVEL = {"MMLU": 0.25, "MedQAbstain-LT": 0.25, "MedQAbstain-Safe": 0.25}

# Miliardi di parametri, per l'asse della scala.
MODEL_PARAMS_B = {"LFM2-350M": 0.35, "LFM2-1.2B": 1.2,
                  "MedGemma-4B-it": 4.0, "Gemma3-4B-it": 4.0,
                  "Mistral-7B-it": 7.0}

# Valori riportati da Vashurin et al. per il regime di scala del paper, usati
# come termine di paragone per l'ancora di replica (TODO 1).
PAPER_REFERENCE_PRR = {"TriviaQA": 0.60, "MMLU": 0.50}
ANCHOR_MODEL = "Mistral-7B-it"

# Zona in cui il PRR non e' interpretabile: troppo poche risposte corrette o
# troppo poche sbagliate da ordinare (vedi build_accuracy_table in main.py).
INTERPRETABLE_ACCURACY = (0.30, 0.85)


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
             "correttamente, e il PRR misurato su quelle etichette non e' interpretabile.",)
    save(fig, results_dir, "fig_accuracy.png")


def fig_accuracy_vs_prr(stats, results_dir):
    """Accuracy contro PRR: la relazione che decide se una barra e' leggibile.

    Ogni punto e' una coppia modello-dataset. Il PRR e' la mediana fra i
    metodi, perche' la domanda qui non e' quale metodo vince ma quanto segnale
    ci sia in quella cella."""
    cell = stats.groupby(["model", "dataset"], as_index=False).agg(
        accuracy=("mean_quality", "first"),
        prr_mediano=("prr", "median"),
        prr_massimo=("prr", "max"),
    ).dropna(subset=["accuracy", "prr_mediano"])
    if cell.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    lo, hi = INTERPRETABLE_ACCURACY
    ax.axvspan(0, lo, color="tab:red", alpha=0.07)
    ax.axvspan(hi, 1.0, color="tab:red", alpha=0.07)
    ax.text(lo / 2, 0.97, "troppo pochi\nsuccessi", ha="center", va="top",
            fontsize=7, color="tab:red", transform=ax.get_xaxis_transform())
    ax.text((hi + 1) / 2, 0.97, "troppi pochi\nerrori", ha="center", va="top",
            fontsize=7, color="tab:red", transform=ax.get_xaxis_transform())

    for model_name, group in cell.groupby("model"):
        ax.scatter(group["accuracy"], group["prr_mediano"], s=70, label=model_name, zorder=3)
        for _, row in group.iterrows():
            ax.annotate(row["dataset"], (row["accuracy"], row["prr_mediano"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=6.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy di base del modello sul dataset")
    ax.set_ylabel("PRR mediano fra i metodi UQ")
    ax.set_title("Il PRR e' interpretabile solo in un intervallo di accuracy")
    ax.legend(fontsize=8)
    add_note(fig,
             "Fuori dalle bande rosse il PRR misura la tecnica. Dentro, misura il regime: con "
             "quasi nessuna risposta corretta non c'e' nulla\nda ordinare, con quasi nessun errore "
             "la stima e' dominata dal rumore. Un PRR basso a sinistra non e' un metodo debole.",)
    save(fig, results_dir, "fig_accuracy_vs_prr.png")


def fig_anchor_replication(stats, results_dir):
    """Ancora di replica: i PRR di Mistral-7B contro i valori del paper."""
    anchor = stats[stats["model"] == ANCHOR_MODEL]
    if anchor.empty:
        print(f"  {ANCHOR_MODEL} assente dai risultati, salto la figura dell'ancora.")
        return
    agg = anchor.groupby("dataset").agg(prr_massimo=("prr", "max"),
                                        prr_mediano=("prr", "median"),
                                        accuracy=("mean_quality", "first"))
    datasets = [d for d in agg.index if d in PAPER_REFERENCE_PRR] + \
               [d for d in agg.index if d not in PAPER_REFERENCE_PRR]
    agg = agg.loc[datasets]

    y = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(10, max(4, len(agg) * 0.85)))
    ax.barh(y + 0.2, agg["prr_massimo"], height=0.35, label="miglior metodo (ottenuto)")
    ax.barh(y - 0.2, agg["prr_mediano"], height=0.35, label="metodo mediano (ottenuto)")
    for i, ds in enumerate(agg.index):
        if ds in PAPER_REFERENCE_PRR:
            ax.plot([PAPER_REFERENCE_PRR[ds]] * 2, [i - 0.45, i + 0.45],
                    color="tab:red", linewidth=2, zorder=5)
    ax.plot([], [], color="tab:red", linewidth=2, label="atteso dal paper")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{d}\n(accuracy {agg.loc[d, 'accuracy']:.2f})" for d in agg.index],
                       fontsize=8)
    ax.set_xlabel("PRR (raw, max_rejection=0.5)")
    ax.set_title(f"Ancora di replica: {ANCHOR_MODEL} contro i valori attesi da Vashurin et al.")
    ax.legend(loc="lower right", fontsize=8)
    add_note(fig,
             "L'ancora serve a distinguere gli effetti di scala dagli artefatti della nostra pipeline: "
             "ha valore solo se riproduce i pattern\ndel paper nel suo stesso regime. Nota che qui e' "
             "caricata in 4-bit e non in bf16 (vincolo dei 24GB della 3090), quindi uno scostamento\n"
             "e' attribuibile anche alla quantizzazione.",)
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
    ).reset_index()
    agg["params_B"] = agg["model"].map(MODEL_PARAMS_B)
    agg = agg.dropna(subset=["params_B"]).sort_values("params_B")
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(agg["params_B"], agg["ties"], "o-", color="tab:blue")
    for _, r in agg.iterrows():
        ax.annotate(f"{r['model']}\n({int(r['celle'])} dataset)",
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
             "produce tau vicino a zero per costruzione.",)
    save(fig, results_dir, "fig_ties_vs_scale.png")


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

    modelli = [m for m in MODEL_PARAMS_B if m in set(agg["model"])]
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
    ax.set_xlabel("Mean PRR (raw, max_rejection=0.5, aggregato sui dataset di selective QA)")
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
    modelli = [m for m in MODEL_PARAMS_B if m in set(raw["model"])]

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
            ax.set_xlabel("Mean PRR (raw, max_rejection=0.5)", fontsize=8)
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
             "continuo non sono commensurabili. Leggere sempre insieme alle accuracy\nin titolo: dove "
             "sono molto basse, il PRR misura il regime e non la tecnica.",
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
    ax.set_xlabel("Mean PRR (raw, max_rejection=0.5, aggregato sui dataset)")
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
    t = pd.read_csv(path)
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
            label="costo pieno standalone (greedy + campioni + NLI + aritmetica)")
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


def _tabella_immagine(fig, ax, df, col_widths=None, fontsize=8):
    ax.axis("off")
    tab = ax.table(cellText=df.values.tolist(), colLabels=df.columns.tolist(),
                   cellLoc="center", loc="center", colWidths=col_widths)
    tab.auto_set_font_size(False)
    tab.set_fontsize(fontsize)
    tab.scale(1, 1.45)
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
    tabella = tabella.reindex([m for m in MODEL_PARAMS_B if m in tabella.index])
    df = tabella.reset_index().rename(columns={"model": "modello"})

    fig, ax = plt.subplots(figsize=(2.1 * len(df.columns), 0.9 + 0.55 * len(df)))
    _tabella_immagine(fig, ax, df)
    ax.set_title("Qualita' di base per modello e dataset, con la metrica usata",
                 fontsize=11, pad=16)
    add_note(fig,
             "Su MCQ a quattro opzioni il livello del caso e' 0.25: un valore sotto non indica "
             "ignoranza del modello ma un problema\ndi estrazione della risposta. AlignScore e' "
             "continuo e severo sul testo libero medico: valori bassi li' significano che\nquasi ogni "
             "risposta conta come sbagliata, e il PRR misurato in quel regime non e' interpretabile.")
    save(fig, results_dir, "fig_tabella_accuracy.png")


def table_prr_vs_accuracy(stats, results_dir):
    """PRR affiancato all'accuracy della stessa cella."""
    cell = stats.groupby(["model", "dataset"], as_index=False).agg(
        accuracy=("mean_quality", "first"),
        metrica=("quality_metric", "first"),
        prr_mediano=("prr", "median"),
        prr_migliore=("prr", "max"),
        n=("prr", "count"),
    ).dropna(subset=["accuracy"])
    if cell.empty:
        return
    idx = stats.dropna(subset=["prr"]).groupby(["model", "dataset"])["prr"].idxmax()
    migliori = stats.loc[idx, ["model", "dataset", "paper_label"]].rename(
        columns={"paper_label": "metodo_migliore"})
    cell = cell.merge(migliori, on=["model", "dataset"], how="left")
    cell["interpretabile"] = np.where(
        (cell["accuracy"] >= INTERPRETABLE_ACCURACY[0])
        & (cell["accuracy"] <= INTERPRETABLE_ACCURACY[1]), "si", "NO")
    cell = cell.sort_values(["model", "dataset"])

    df = pd.DataFrame({
        "modello": cell["model"],
        "dataset": cell["dataset"],
        "accuracy": cell["accuracy"].map("{:.3f}".format),
        "metrica": cell["metrica"],
        "PRR mediano": cell["prr_mediano"].map("{:.3f}".format),
        "PRR migliore": cell["prr_migliore"].map("{:.3f}".format),
        "metodo migliore": cell["metodo_migliore"].fillna("-"),
        "regime": cell["interpretabile"],
    })

    # Larghezze esplicite: "metodo migliore" contiene nomi lunghi come
    # "EigValLaplacian NLI Score Entail." che con le colonne uniformi vengono
    # troncati a meta'.
    larghezze = [0.13, 0.12, 0.09, 0.11, 0.10, 0.10, 0.28, 0.07]
    fig, ax = plt.subplots(figsize=(15, 0.6 + 0.30 * len(df)))
    tab = _tabella_immagine(fig, ax, df, col_widths=larghezze, fontsize=7.5)
    for r in range(len(df)):
        if df["regime"].iloc[r] == "NO":
            for c in range(len(df.columns)):
                tab[(r + 1, c)].set_facecolor("#fbe9e7")
    ax.set_title("PRR affiancato all'accuracy della stessa cella", fontsize=11, pad=16)
    add_note(fig,
             "Le righe evidenziate sono fuori dall'intervallo di accuracy in cui il PRR e' "
             "interpretabile: troppo pochi successi o troppo\npochi errori da ordinare. Su queste "
             "celle un PRR basso non significa che i metodi UQ siano deboli. Misurato sul complesso "
             "delle\ncelle, l'accuracy da sola spiega circa il 97% della varianza del PRR: leggere "
             "sempre le due colonne insieme.")
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
                    marker="o", capsize=3, linewidth=1.6, label=famiglia)
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
             "analysis_lib.py. Da leggere insieme alla tabella PRR-accuracy: dove l'accuracy e' fuori "
             "regime,\nil PRR non misura la tecnica.")
    save(fig, results_dir, "fig_prr_vs_scale_by_family.png")


def fig_rank_transfer(stats, results_dir):
    """Kendall tau del ranking dei metodi contro il modello-ancora a 7B."""
    if ANCHOR_MODEL not in set(stats["model"]):
        print(f"  {ANCHOR_MODEL} assente, salto il rank transfer.")
        return
    try:
        from scipy.stats import kendalltau
    except ImportError:
        print("  scipy non disponibile, salto il rank transfer.")
        return

    agg = stats.groupby(["model", "paper_label"], as_index=False)["prr"].mean()
    pivot = agg.pivot_table(index="paper_label", columns="model", values="prr")
    if ANCHOR_MODEL not in pivot.columns:
        return
    righe = []
    for model_name in pivot.columns:
        if model_name == ANCHOR_MODEL:
            continue
        pair = pivot[[ANCHOR_MODEL, model_name]].dropna()
        if len(pair) < 3:
            continue
        tau, p = kendalltau(pair[ANCHOR_MODEL].rank(ascending=False),
                            pair[model_name].rank(ascending=False))
        righe.append({"model": model_name, "params_B": MODEL_PARAMS_B.get(model_name, np.nan),
                      "tau": tau, "p": p, "n": len(pair)})
    if not righe:
        return
    tau_df = pd.DataFrame(righe).dropna(subset=["params_B"]).sort_values("params_B")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(tau_df["params_B"], tau_df["tau"], "o-", color="tab:blue")
    for _, r in tau_df.iterrows():
        significativo = "significativo" if r["p"] < 0.05 else f"p={r['p']:.2f}"
        ax.annotate(f"{r['model']}\n{significativo}, {int(r['n'])} metodi",
                    (r["params_B"], r["tau"]), textcoords="offset points",
                    xytext=(8, 4), fontsize=7.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="0.7", linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Parametri del modello (miliardi, scala log)")
    ax.set_ylabel(f"Kendall tau del ranking dei metodi vs {ANCHOR_MODEL}")
    ax.set_title("Trasferimento del ranking dei metodi UQ al calare della scala")
    add_note(fig,
             "tau = 1: stesso ordine di preferenza dei metodi del modello a 7B; tau = 0: ranking "
             "scorrelato.\nAttenzione: se a una certa scala i metodi sono statisticamente "
             "indistinguibili fra loro (vedi fig_ties_vs_scale.png),\nil loro ordinamento e' rumore e "
             "un tau vicino a zero e' garantito per costruzione, non e' un risultato sul "
             "trasferimento.")
    save(fig, results_dir, "fig_rank_transfer.png")


def fig_pareto(results_dir):
    """Frontiera di Pareto costo pieno per istanza contro affidabilita'."""
    path = os.path.join(results_dir, "estimator_timings.csv")
    raw = _read_mapped(results_dir, "results_paper_mapped.csv")
    if not os.path.exists(path) or raw is None:
        print("  Dati insufficienti per la frontiera di Pareto, salto.")
        return
    t = pd.read_csv(path)
    if "seconds_full_per_instance" not in t.columns:
        print("  estimator_timings.csv ha uno schema vecchio, salto la Pareto.")
        return
    costo = t.groupby("paper_label", as_index=False)["seconds_full_per_instance"].mean()
    qualita = aggregate_across_datasets(raw).groupby("paper_label", as_index=False)["value"].mean()
    punti = qualita.merge(costo, on="paper_label", how="inner").dropna()
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
             "alla domanda del deployment on-device.")
    save(fig, results_dir, "fig_pareto_cost_quality.png")


def table_costi(results_dir):
    """Tabella costi per metodo: tempo marginale, tempo pieno, memoria di picco."""
    path = os.path.join(results_dir, "estimator_timings.csv")
    if not os.path.exists(path):
        return
    t = pd.read_csv(path)
    attese = {"seconds_marginal_per_instance", "seconds_full_per_instance", "peak_memory_gb"}
    if not attese.issubset(t.columns):
        print(f"  estimator_timings.csv senza le colonne di costo, salto la tabella costi.")
        return
    costo = t.groupby("paper_label", as_index=False).agg(
        marginale=("seconds_marginal_per_instance", "mean"),
        pieno=("seconds_full_per_instance", "mean"),
        memoria=("peak_memory_gb", "max"),
        campioni=("needs_sampling", "max"),
        nli=("needs_nli", "max"),
    ).sort_values("pieno", ascending=False)

    df = pd.DataFrame({
        "metodo": costo["paper_label"],
        "famiglia": costo["paper_label"].map(family_of),
        "s/istanza (marginale)": costo["marginale"].map("{:.4f}".format),
        "s/istanza (pieno)": costo["pieno"].map("{:.3f}".format),
        "memoria picco (GB)": costo["memoria"].map("{:.1f}".format),
        "K campioni": np.where(costo["campioni"], "si", "-"),
        "NLI": np.where(costo["nli"], "si", "-"),
    })
    fig, ax = plt.subplots(figsize=(13, 0.6 + 0.30 * len(df)))
    _tabella_immagine(fig, ax, df, col_widths=[0.28, 0.16, 0.13, 0.12, 0.13, 0.09, 0.07],
                      fontsize=7.5)
    ax.set_title("Costo per metodo UQ: tempo marginale, tempo pieno standalone, memoria di picco",
                 fontsize=11, pad=16)
    add_note(fig,
             "Il tempo marginale e' la sola aritmetica su statistiche gia' pronte, utile a chi "
             "calcola molte tecniche insieme. Il tempo pieno\ne' quello che conta per il deployment "
             "on-device: include generazione greedy, K campioni e forward NLI se il metodo li "
             "richiede.\nLa memoria di picco e' quella dell'intera combinazione, quindi e' un limite "
             "superiore per il singolo metodo.")
    save(fig, results_dir, "fig_tabella_costi.png")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    args = parser.parse_args()

    stats = load_stats(args.results_dir)
    if stats is None:
        raise SystemExit(f"!!! Nessuna statistica per-istanza in {args.results_dir}.")
    print(f"Celle modello x dataset disponibili: "
          f"{stats.groupby(['model', 'dataset']).ngroups}")

    fig_accuracy(stats, args.results_dir)
    fig_accuracy_vs_prr(stats, args.results_dir)
    fig_anchor_replication(stats, args.results_dir)
    fig_ties_vs_scale(args.results_dir)

    # Le otto figure richieste (vedi figure_richieste.txt).
    fig_timing_full_cost(args.results_dir)                                    # 1
    fig_paper_figure(args.results_dir, "A", "fig_a_white_box.png",            # 2
                     "Mean PRR aggregato su selective QA "
                     "(~ Fig. 2 Vashurin et al., white-box full-access)")
    fig_paper_figure(args.results_dir, "B", "fig_b_reflexive.png",            # 3
                     "Mean PRR aggregato su selective QA "
                     "(~ Fig. 3 Vashurin et al., reflexive/black-box)")
    fig_quant_comparison(args.results_dir)                                    # 4
    fig_severity_grid(args.results_dir)                                       # 5
    table_accuracy(stats, args.results_dir)                                   # 6
    table_prr_vs_accuracy(stats, args.results_dir)                            # 7
    fig_verbalized_accuracy_cost(args.results_dir)                            # 8

    # Artefatti chiesti dal TODO ma non compresi nelle otto.
    fig_prr_vs_scale_by_family(args.results_dir)                              # TODO 1
    fig_rank_transfer(stats, args.results_dir)                                # TODO 1
    fig_pareto(args.results_dir)                                              # TODO 6
    table_costi(args.results_dir)                                             # TODO 6
    for style in ("numeric", "linguistic"):                                   # TODO 3
        _tabella_da_csv(
            args.results_dir, f"parse_failure_rate_{style}.csv",
            f"Parse-failure rate, confidenza {style}",
            f"fig_tabella_parse_failure_{style}.png",
            "Frazione di istanze in cui la confidenza non e' estraibile dal testo generato. Va letta "
            "ACCANTO al PRR: lm-polygraph\nconverte i punteggi non estraibili in -1e7, cioe' li tratta "
            "come massima confidenza, quindi un modello che non rispetta il\nformato non viene "
            "penalizzato dal PRR ma premiato. Un parse-failure rate alto rende il PRR di quel metodo "
            "non informativo.")
    _tabella_da_csv(                                                          # TODO 4
        args.results_dir, "silent_failure_rate.csv",
        "Silent failure rate per strato di severita'",
        "fig_tabella_silent_failure.png",
        "Frazione delle risposte SBAGLIATE che finisce nel decile piu' confidente del metodo: gli "
        "errori che passano\ninosservati, presentati con la massima sicurezza. E' la metrica che "
        "interessa in ambito clinico, dove non conta\nquanto bene il metodo ordina in media ma quanti "
        "errori pericolosi lascia passare.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
