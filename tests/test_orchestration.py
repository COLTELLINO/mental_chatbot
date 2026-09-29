"""main() completo con modello minuscolo e dataset finti: verifica sezioni, checkpoint,
file per-istanza e ripresa (un secondo lancio con --models diversi non deve cancellare
le celle del primo)."""
import sys, os, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
exec(open("test_integration.py").read().split("class Boom")[0])   # modello pilotato + import
import main as M
from dataset_prep import MCQAccuracyMetric
def estimators():
    return [MaximumSequenceProbability(), Perplexity(), MeanTokenEntropy(),
            MonteCarloSequenceEntropy(), LexicalSimilarity(metric="rougeL")]

def fake_loader(tag):
    def load(n_test, seed, cache_dir=None):
        k = min(n_test, 7)
        return [{"content": f"Domanda: {tag} {i}\nA) uno\nB) due", "reference": "A" if i % 3 else "B"}
                for i in range(k)]
    return load

def cfg(tag, **extra):
    c = {"loader": fake_loader(tag), "n_test": 7, "max_new_tokens": 12, "plain_suffix": "\nRisposta:",
         "generation_metric_factory": MCQAccuracyMetric, "stop_strings": ["\n"],
         "continuation_stop_strings": ["\nDomanda:"]}
    c.update(extra); return c

M.MODELS = {"tinyA": "tiny", "tinyB": "tiny"}
M.CANNOT_QUANTIZE_MODELS = {"tinyA", "tinyB"}
M.MODEL_PARAMS_B.update({"tinyA": 0.001, "tinyB": 0.002})
M.DATASETS = {"CoQA": cfg("coqa"), "MMLU": cfg("mmlu")}
M.VERBALIZED_DATASETS = dict(M.DATASETS)
M.SEVERITY_DATASETS = {
    "MedQAbstain-LT": cfg("lt", severity="alta", answer_format="MCQ", severity_label="alta"),
    "MedQAbstain-Safe": cfg("safe", severity="bassa", answer_format="MCQ", severity_label="bassa"),
    "MedicationQA": cfg("med", severity="alta", answer_format="libera", severity_label="alta"),
    "MedQuAD": cfg("quad", severity="bassa", answer_format="libera", severity_label="bassa"),
}
M.build_estimators = estimators
M.QUANT_COMPARE_MODEL_DEFAULT = "tinyA"

def launch(models):
    sys.argv = ["main.py", "--results_dir", "orch", "--cache_dir", "hfcache", "--n_bootstrap", "20",
                "--chunk_size", "3", "--run_severity_grid", "--run_verbalized", "--models", *models]
    M.main()

shutil.rmtree("orch", ignore_errors=True)
launch(["tinyA"])
launch(["tinyB"])   # secondo job, altro modello, stessa cartella: non deve perdere tinyA

def cells(f):
    d = pd.read_csv(os.path.join("orch", f)); return sorted(set(zip(d["model"], d["dataset"])))
for f in ["per_instance_scores.csv", "instance_level_stats.csv",
          "results_severity_grid_per_instance.csv", "results_severity_grid_instance_stats.csv",
          "results_verbalized_numeric_per_instance.csv"]:
    c = cells(f); print(f, "->", c)
    assert {m for m, _ in c} == {"tinyA", "tinyB"}, f
pi = pd.read_csv("orch/results_severity_grid_per_instance.csv")
assert pi.groupby(["model", "dataset"]).size().eq(7).all()
print("OK: dopo due job con --models diversi, tutti i file per-istanza contengono entrambi i modelli (7 istanze per cella)")
print("figure:", sorted(f for f in os.listdir("orch") if f.endswith(".png"))[:12], "...")
print("ORCHESTRAZIONE OK")
