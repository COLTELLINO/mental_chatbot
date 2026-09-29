"""Rimuove uno o piu' modelli dai checkpoint in results/, cosi' che il
prossimo run li riesegua da capo lasciando intatti gli altri.

A cosa serve: main.py salta le combinazioni modello x dataset gia' presenti in
results_final.csv (vedi load_checkpoint_csv). E' quello che si vuole dopo un
job interrotto, ma NON dopo aver cambiato la configurazione di un modello: in
quel caso il checkpoint contiene risultati prodotti con i parametri vecchi, e
il resume li conserverebbe.

Caso concreto per cui e' nato (run 14978811): Gemma3-4B-it e MedGemma-4B-it
erano passati su CoQA ma con solo ~10 stimatori su 26 -- tutti quelli a
passaggio singolo, nessuno di quelli a campionamento -- perche' a batch 4 la
memoria non bastava. Dopo aver ridotto il batch vanno rieseguiti interamente,
mentre LFM2-350M, LFM2-1.2B e Mistral-7B-it sono completi e rifarli
costerebbe ore di GPU per riottenere gli stessi numeri.

Serve anche per dataset, non solo per modelli: se cambia la metrica di
qualita' di un task, tutti i risultati di quel task vanno rifatti per ogni
modello. E' il caso di MMLU e MedQAbstain-LT/Safe dopo la sostituzione della
regex di estrazione della lettera con MCQAccuracyMetric.

Uso:
    python3.11 drop_models_from_checkpoints.py results Gemma3-4B-it
        -> tutte le celle di Gemma3-4B-it
    python3.11 drop_models_from_checkpoints.py results --datasets MMLU MedQAbstain-LT
        -> tutte le celle di MMLU e MedQAbstain-LT, per ogni modello
    python3.11 drop_models_from_checkpoints.py results Gemma3-4B-it --datasets MMLU
        -> solo la cella Gemma3-4B-it x MMLU

Scrive una copia .bak di ogni file toccato prima di modificarlo. I checkpoint a
blocchi delle stesse celle (results/chunks/...) vengono SPOSTATI in
results/chunks_rimossi/, non cancellati: se restassero al loro posto, il run
successivo li ricaricherebbe invece di rifare la cella.
"""
import argparse
import os
import re
import shutil
import sys
import time

import pandas as pd

# Tutti i checkpoint che contengono una colonna "model" e da cui quindi va
# tolto il modello. Se un file non esiste viene semplicemente saltato.
CHECKPOINT_FILES = [
    "results_final.csv",
    "results_partial.csv",
    "instance_level_stats.csv",
    "per_instance_scores.csv",
    "estimator_timings.csv",
    "estimator_timings_partial.csv",
    "results_severity_grid.csv",
    "results_severity_grid_instance_stats.csv",
    "results_severity_grid_per_instance.csv",
    "results_verbalized_numeric.csv",
    "results_verbalized_numeric_instance_stats.csv",
    "results_verbalized_numeric_per_instance.csv",
    "results_verbalized_linguistic.csv",
    "results_verbalized_linguistic_instance_stats.csv",
    "results_verbalized_linguistic_per_instance.csv",
    "results_quant_comparison.csv",
    "results_quant_comparison_instance_stats.csv",
    "results_quant_comparison_per_instance.csv",
]


def _safe_name(x):
    # Stessa regola di main._safe_name, usata per i nomi delle cartelle dei blocchi.
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x))


def _drop_row(model, dataset, models, datasets):
    """Con modelli E dataset si toglie solo la loro combinazione; con uno solo
    dei due, tutte le celle che lo riguardano."""
    if models and datasets:
        return model in models and dataset in datasets
    return (bool(models) and model in models) or (bool(datasets) and dataset in datasets)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("models", nargs="*", default=[],
                        help="Modelli da rimuovere dai checkpoint.")
    parser.add_argument("--datasets", nargs="+", default=[],
                        help="Dataset da rimuovere, per ogni modello. Usare quando cambia "
                             "la metrica di qualita' di un task.")
    args = parser.parse_args()

    results_dir = args.results_dir
    models_to_drop = set(args.models)
    datasets_to_drop = set(args.datasets)
    if not models_to_drop and not datasets_to_drop:
        parser.error("specifica almeno un modello o --datasets.")

    print(f"Cartella: {results_dir}")
    if models_to_drop:
        print(f"Modelli da rimuovere: {sorted(models_to_drop)}")
    if datasets_to_drop:
        print(f"Dataset da rimuovere: {sorted(datasets_to_drop)}")
    print()

    for filename in CHECKPOINT_FILES:
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  {filename}: illeggibile ({e}), salto.")
            continue
        if "model" not in df.columns and "dataset" not in df.columns:
            print(f"  {filename}: nessuna colonna 'model'/'dataset', salto.")
            continue

        models_col = df["model"] if "model" in df.columns else pd.Series([None] * len(df), index=df.index)
        datasets_col = df["dataset"] if "dataset" in df.columns else pd.Series([None] * len(df), index=df.index)
        drop_mask = pd.Series(
            [_drop_row(m, d, models_to_drop, datasets_to_drop)
             for m, d in zip(models_col, datasets_col)], index=df.index)
        keep = ~drop_mask
        removed = int(drop_mask.sum())
        if removed == 0:
            print(f"  {filename}: nessuna riga da rimuovere.")
            continue

        shutil.copy2(path, path + ".bak")
        df[keep].to_csv(path, index=False)
        print(f"  {filename}: rimosse {removed} righe su {len(df)} "
              f"(backup in {filename}.bak).")

    chunks_root = os.path.join(results_dir, "chunks")
    moved = 0
    if os.path.isdir(chunks_root):
        safe_models = {_safe_name(m) for m in models_to_drop}
        safe_datasets = {_safe_name(d) for d in datasets_to_drop}
        dest_root = os.path.join(results_dir, "chunks_rimossi", time.strftime("%Y%m%d-%H%M%S"))
        for section in sorted(os.listdir(chunks_root)):
            section_dir = os.path.join(chunks_root, section)
            if not os.path.isdir(section_dir):
                continue
            for cell in sorted(os.listdir(section_dir)):
                parts = cell.split("__")
                if len(parts) < 3:
                    continue
                if _drop_row(parts[0], parts[1], safe_models, safe_datasets):
                    dest = os.path.join(dest_root, section)
                    os.makedirs(dest, exist_ok=True)
                    shutil.move(os.path.join(section_dir, cell), os.path.join(dest, cell))
                    moved += 1
                    print(f"  chunks/{section}/{cell}: spostata in {os.path.relpath(dest, results_dir)}/")
    if moved:
        print(f"  {moved} cartelle di blocchi spostate (non cancellate).")

    print("\nFatto. Rilancia il benchmark SENZA --no_resume: le combinazioni rimosse "
          "verranno rieseguite, le altre riprese dai checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
