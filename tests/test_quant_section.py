"""Sezione --run_quant_comparison con il modello minuscolo. Sulla CPU non c'e'
bitsandbytes, quindi la variante "4-bit" viene caricata non quantizzata: qui si
verifica la logica della sezione (celle, ripresa, etichette), non l'effetto
della quantizzazione. Due lanci nella stessa cartella con due
--quant_compare_model diversi devono produrre quattro serie distinte (prima il
secondo lancio riusava in silenzio le celle del primo)."""
import sys, os, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
src = open("test_orchestration.py").read()
exec(src.split("def launch")[0])          # stesso setup dell'orchestrazione
M.CANNOT_QUANTIZE_MODELS = set()
_orig_load = M.load_whitebox_model
def load_no_bnb(*a, **kw):
    kw["use_quantization"] = False
    return _orig_load(*a, **kw)
M.load_whitebox_model = load_no_bnb

def launch(qmodel):
    sys.argv = ["main.py", "--results_dir", "quant_t", "--cache_dir", "hfcache", "--n_bootstrap", "20",
                "--chunk_size", "3", "--models", "tinyA", "--run_quant_comparison",
                "--quant_compare_model", qmodel]
    M.main()

shutil.rmtree("quant_t", ignore_errors=True)
launch("tinyA")
launch("tinyB")
q = pd.read_csv("quant_t/results_quant_comparison.csv")
serie = sorted(q["model"].unique())
print("serie:", serie)
assert serie == ["tinyA 4-bit nf4", "tinyA bf16", "tinyB 4-bit nf4", "tinyB bf16"], serie
for f in ("results_quant_comparison_instance_stats.csv", "results_quant_comparison_per_instance.csv"):
    d = pd.read_csv(os.path.join("quant_t", f))
    assert sorted(d["model"].unique()) == serie, (f, sorted(d["model"].unique()))
chunks = sorted(os.listdir("quant_t/chunks/quant"))
assert all(any(s.split()[0] in c for s in serie) for c in chunks) and len(chunks) == 8, chunks
print("OK: quattro serie distinte, una cartella di blocchi per (modello, variante, dataset)")
print("QUANT OK")
