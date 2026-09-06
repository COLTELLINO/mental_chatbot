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
    python3.11 drop_models_from_checkpoints.py results --datasets MMLU MedQAbstain-LT
    python3.11 drop_models_from_checkpoints.py results Gemma3-4B-it --datasets MMLU

Scrive una copia .bak di ogni file toccato prima di modificarlo.
"""
import argparse
import os
import shutil
import sys

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
]


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

        drop_mask = pd.Series(False, index=df.index)
        if models_to_drop and "model" in df.columns:
            drop_mask |= df["model"].isin(models_to_drop)
        if datasets_to_drop and "dataset" in df.columns:
            drop_mask |= df["dataset"].isin(datasets_to_drop)
        keep = ~drop_mask
        removed = int(drop_mask.sum())
        if removed == 0:
            print(f"  {filename}: nessuna riga da rimuovere.")
            continue

        shutil.copy2(path, path + ".bak")
        df[keep].to_csv(path, index=False)
        print(f"  {filename}: rimosse {removed} righe su {len(df)} "
              f"(backup in {filename}.bak).")

    print("\nFatto. Rilancia il benchmark SENZA --no_resume: le combinazioni rimosse "
          "verranno rieseguite, le altre riprese dai checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
