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
    fig.text(0.01, 0.01,
             "Un'accuracy sotto la linea rossa su un MCQ non e' ignoranza del modello: tirando a "
             "indovinare ne prenderebbe di piu'.\nIndica che la risposta non viene estratta "
             "correttamente, e il PRR misurato su quelle etichette non e' interpretabile.",
             fontsize=7, va="bottom")
    fig.subplots_adjust(bottom=0.16)
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
    fig.text(0.01, 0.01,
             "Fuori dalle bande rosse il PRR misura la tecnica. Dentro, misura il regime: con "
             "quasi nessuna risposta corretta non c'e' nulla\nda ordinare, con quasi nessun errore "
             "la stima e' dominata dal rumore. Un PRR basso a sinistra non e' un metodo debole.",
             fontsize=7, va="bottom")
    fig.subplots_adjust(bottom=0.14)
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
    fig.text(0.01, 0.01,
             "L'ancora serve a distinguere gli effetti di scala dagli artefatti della nostra pipeline: "
             "ha valore solo se riproduce i pattern\ndel paper nel suo stesso regime. Nota che qui e' "
             "caricata in 4-bit e non in bf16 (vincolo dei 24GB della 3090), quindi uno scostamento\n"
             "e' attribuibile anche alla quantizzazione.",
             fontsize=7, va="bottom")
    fig.subplots_adjust(bottom=0.2)
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
    fig.text(0.01, 0.01,
             "Un valore alto significa che il benchmark, a quella scala, non separa i metodi: la loro "
             "classifica e' rumore.\nVa letto insieme al Kendall tau, perche' un ranking indistinguibile "
             "produce tau vicino a zero per costruzione.",
             fontsize=7, va="bottom")
    fig.subplots_adjust(bottom=0.16)
    save(fig, results_dir, "fig_ties_vs_scale.png")


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
