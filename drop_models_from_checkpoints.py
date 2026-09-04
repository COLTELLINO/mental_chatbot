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

Uso:
    python3.11 drop_models_from_checkpoints.py results Gemma3-4B-it MedGemma-4B-it

Scrive una copia .bak di ogni file toccato prima di modificarlo.
"""
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
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    results_dir = sys.argv[1]
    models_to_drop = set(sys.argv[2:])
    print(f"Cartella: {results_dir}")
    print(f"Modelli da rimuovere: {sorted(models_to_drop)}\n")

    for filename in CHECKPOINT_FILES:
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  {filename}: illeggibile ({e}), salto.")
            continue
        if "model" not in df.columns:
            print(f"  {filename}: nessuna colonna 'model', salto.")
            continue

        keep = ~df["model"].isin(models_to_drop)
        removed = int((~keep).sum())
        if removed == 0:
            print(f"  {filename}: nessuna riga da rimuovere.")
            continue

        shutil.copy2(path, path + ".bak")
        df[keep].to_csv(path, index=False)
        print(f"  {filename}: rimosse {removed} righe su {len(df)} "
              f"(backup in {filename}.bak).")

    print("\nFatto. Rilancia il benchmark SENZA --no_resume: i modelli rimossi "
          "verranno rieseguiti, gli altri ripresi dai checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
