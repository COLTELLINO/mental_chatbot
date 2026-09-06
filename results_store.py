"""Lo strato dati del benchmark: CSV con semantica di sovrascrittura per riga.

Il problema che risolve. Finora ogni sezione di main.py gestiva i propri
checkpoint a modo suo: leggere il CSV esistente, calcolare quali combinazioni
mancano, saltare quelle gia' presenti, concatenare e riscrivere. Tre
implementazioni della stessa idea, con tre modi diversi di sbagliare. E la
semantica era "salta se gia' presente", non "sovrascrivi se ricalcolato":
per rifare una cella bisognava prima cancellarla a mano da tutti i file che la
contenevano, e dimenticarne uno significava mescolare in una stessa figura
risultati prodotti da versioni diverse del codice. E' esattamente quello che e'
successo il 2026-09-01, quando una run ha ripreso in silenzio checkpoint di
agosto.

La regola qui e' una sola: ogni tabella ha una CHIAVE. Scrivere righe con una
chiave che esiste gia' sostituisce quelle righe; le altre non vengono toccate.
Nient'altro da ricordare.

Scrittura atomica: si scrive su un file temporaneo e poi lo si rinomina. Un job
SLURM ucciso a meta' scrittura lascia il CSV precedente intatto invece di uno
troncato che sembra valido.
"""
import os
import shutil
import tempfile

import pandas as pd

# Chiave di ogni tabella: le colonne che identificano univocamente una riga.
# Scrivere una riga con la stessa chiave sostituisce quella vecchia.
TABLE_KEYS = {
    "results": ["model", "dataset", "estimator", "ue_metric"],
    "instance_stats": ["model", "dataset", "estimator"],
    "per_instance": ["model", "dataset"],
    "timings": ["model", "dataset", "estimator"],
    "accuracy": ["model", "dataset"],
    "paired_comparisons": ["model", "dataset", "method"],
    "rank_transfer": ["model"],
}


def _atomic_write(df, path):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=directory)
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read(path, required_columns=None):
    """Rilegge una tabella. Ritorna None se assente, illeggibile, o se le
    colonne attese non ci sono -- quest'ultimo caso significa "prodotta da una
    versione precedente del codice", e fidarsene mescolerebbe risultati di due
    versioni diverse nella stessa figura."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"!!! {os.path.basename(path)} illeggibile ({e}): lo ignoro.")
        return None
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            print(f"!!! {os.path.basename(path)} ha uno schema vecchio "
                  f"(mancano {sorted(missing)}): lo ignoro e ricalcolo.")
            return None
    return df


def upsert(path, new_rows, key):
    """Inserisce o sostituisce righe in base alla chiave.

    Le righe esistenti la cui chiave compare in `new_rows` vengono rimosse e
    rimpiazzate; tutte le altre restano dove sono. Ritorna la tabella completa
    risultante.
    """
    if new_rows is None or len(new_rows) == 0:
        return read(path)

    missing = set(key) - set(new_rows.columns)
    if missing:
        raise ValueError(f"Le righe da scrivere in {os.path.basename(path)} non hanno "
                         f"le colonne chiave {sorted(missing)}.")

    existing = read(path, required_columns=key)
    if existing is None or existing.empty:
        combined = new_rows.copy()
    else:
        # Allineo i tipi delle colonne chiave prima del confronto: un modello
        # letto da CSV e' str, ma una colonna numerica potrebbe tornare int64
        # da una parte e object dall'altra, e le chiavi non combacerebbero.
        left = existing.copy()
        right = new_rows.copy()
        for col in key:
            left[col] = left[col].astype(str)
            right[col] = right[col].astype(str)
        incoming = set(map(tuple, right[key].to_numpy()))
        mask_replaced = left[key].apply(lambda r: tuple(r) in incoming, axis=1)
        n_replaced = int(mask_replaced.sum())
        combined = pd.concat([existing[~mask_replaced], new_rows], ignore_index=True)
        if n_replaced:
            print(f"  {os.path.basename(path)}: sostituite {n_replaced} righe, "
                  f"aggiunte {len(new_rows) - n_replaced if len(new_rows) > n_replaced else 0}, "
                  f"conservate {len(existing) - n_replaced}.")

    _atomic_write(combined, path)
    return combined


def done_keys(path, key):
    """Combinazioni gia' presenti, per decidere cosa ricalcolare. Vuoto se il
    file non esiste o ha uno schema vecchio."""
    df = read(path, required_columns=key)
    if df is None or df.empty:
        return set()
    return set(map(tuple, df[key].astype(str).to_numpy()))
