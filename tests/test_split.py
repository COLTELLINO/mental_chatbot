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
M.MODEL_PARAMS_B.update({"tinyA": 0.001, "tinyB": 0.002})
M.DATASETS = {"CoQA": cfg("coqa"), "MMLU": cfg("mmlu"), "GSM8k": cfg("gsm")}
M.VERBALIZED_DATASETS = {k: v for k, v in M.DATASETS.items() if k != "GSM8k"}
M.SEVERITY_DATASETS = {
    "MedQAbstain-LT": cfg("lt", severity="alta", answer_format="MCQ", severity_label="alta"),
    "MedQAbstain-Safe": cfg("safe", severity="bassa", answer_format="MCQ", severity_label="bassa"),
    "MedicationQA": cfg("med", severity="alta", answer_format="libera", severity_label="alta"),
    "MedQuAD": cfg("quad", severity="bassa", answer_format="libera", severity_label="bassa"),
}
M.build_estimators = estimators
_lw = M.load_whitebox_model
M.load_whitebox_model = lambda *a, **k: _lw(*a, **{**k, 'use_quantization': False})
M.QUANT_COMPARE_MODEL_DEFAULT = "tinyA"


def launch(*extra):
    sys.argv = ["main.py", "--results_dir", "split", "--cache_dir", "hfcache", "--n_bootstrap", "20",
                "--chunk_size", "3", *extra]
    M.main()
shutil.rmtree("split", ignore_errors=True)
launch("--datasets", "CoQA", "MMLU", "--run_severity_grid", "--run_verbalized", "--run_quant_comparison")
launch("--datasets", "GSM8k", "--run_quant_comparison")
def cells(f):
    d = pd.read_csv(os.path.join("split", f)); return sorted(set(zip(d["model"], d["dataset"])))
fin = cells("results_final.csv"); print("results_final ->", fin)
assert {d for _, d in fin} == {"CoQA", "MMLU", "GSM8k"} and {m for m, _ in fin} == {"tinyA", "tinyB"}
pi = cells("per_instance_scores.csv"); assert len(pi) == 6, pi
q = cells("results_quant_comparison.csv"); print("quant ->", q)
assert {d for _, d in q} == {"CoQA", "MMLU", "GSM8k"}
qpi = cells("results_quant_comparison_per_instance.csv"); assert len(qpi) == len(q), (qpi, q)
v = cells("results_verbalized_numeric.csv"); print("verbalized ->", v)
assert {d for _, d in v} == {"CoQA", "MMLU"}
g = cells("results_severity_grid.csv"); assert len(g) == 8
print("SPLIT OK: run 1 (tutto tranne GSM8k) + run 2 (solo GSM8k) -> risultati completi")
