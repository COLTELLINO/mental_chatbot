"""Replica del paper (paper_replica.py).

1. Le domande valutate sono ESATTAMENTE quelle che sceglie lm-polygraph
   (Dataset.load + Dataset.subsample(2000, seed=1), come scripts/polygraph_eval),
   stesso ordine, per ogni dataset e variante di prompt. Serve la rete (Hugging Face).
2. Le metriche di qualita' riproducono la post-elaborazione del protocollo.
3. Esecuzione completa con il modello minuscolo (white-box, black-box, due
   varianti verbalized), confronto con il paper e figura.
"""
import os, sys, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import paper_replica as PR
from lm_polygraph.utils.dataset import Dataset as LPDataset

cfg = PR.load_configs()
cache = "hfcache_datasets"

# 1. stessa selezione di lm-polygraph
checked = 0
todo = [("whitebox", n, e) for n, e in cfg["whitebox"].items()]
todo += [(f"blackbox/{s}", n, e) for s in ("empirical_baselines", "verb_2s_topk")
         for n, e in cfg["blackbox"][s].items()]
for where, name, entry in todo:
    ours = PR.make_loader(entry, name)(None, cache_dir=cache)
    ds = LPDataset.load([entry["hf_dataset"].replace("LM-polygraph/", "LM-Polygraph/"), entry["subset"]],
                        "input", "output", batch_size=1, split=entry["split"], cache_dir=cache)
    if name not in PR.PAPER_NO_SUBSAMPLE:
        ds.subsample(PR.PAPER_MAX_INSTANCES, seed=1)
    assert [e["content"] for e in ours] == list(ds.x), (where, name)
    assert [e["reference"] for e in ours] == list(ds.y), (where, name)
    checked += 1
    print(f"  {where}/{name}: {len(ours)} domande, identiche a lm-polygraph")
n = {e[1]: len(PR.make_loader(e[2], e[1])(None, cache_dir=cache)) for e in todo if e[0] == "whitebox"}
assert n == {"CoQA": 2000, "TriviaQA": 2000, "MMLU": 5700, "GSM8k": 1319}, n
print(f"OK 1: selezione delle domande identica a lm-polygraph ({checked} combinazioni); numerosita' {n}")

# 2. metriche (accuracy; AlignScore e' verificata solo come catena di chiamate)
wb = cfg["whitebox"]
gsm = PR.make_metric_factory(wb["GSM8k"], "GSM8k")()
v = gsm({"greedy_texts": ["She has 9 eggs. 9 * 2 = 18. The answer is 18.", "The answer is 17."]},
        ["...\n#### 18", "...\n#### 18"])
assert list(v) == [1, 0], v
mmlu = PR.make_metric_factory(wb["MMLU"], "MMLU")()
assert list(mmlu({"greedy_texts": [" B", "C"]}, ["B", "B"])) == [1, 0]
bb_mmlu = PR.make_metric_factory(cfg["blackbox"]["verb_1s_top1"]["MMLU"], "MMLU")()
assert list(bb_mmlu({"greedy_texts": ["Guess: B\nProbability: 0.8", "Guess: A."]}, ["B", "B"])) == [1, 0]
cot = PR.make_metric_factory(cfg["blackbox"]["verb_2s_cot"]["MMLU"], "MMLU")()
assert list(cot({"greedy_texts": ["Explanation: because x, the answer is C. Guess: C"]}, ["C"])) == [1]
print("OK 2: metriche di qualita' con la post-elaborazione del protocollo")

# 3. esecuzione completa con il modello minuscolo
import main as M
from lm_polygraph.estimators import MaximumSequenceProbability, Perplexity, LexicalSimilarity, NumSemSets
from lm_polygraph.generation_metrics import AccuracyMetric

class FakeAlignScore(AccuracyMetric):
    """AlignScore senza scaricare il modello: stessa interfaccia."""
    def __init__(self, *a, **k):
        super().__init__(normalize=True)
    def __str__(self):
        return "AlignScore"
PR.AlignScore = FakeAlignScore
PR.WHITEBOX_MODEL = ("Mistral-7B-v0.2-base", "tiny")
PR.BLACKBOX_MODEL = ("Mistral-7B-v0.2-it", "tiny_chat")
PR.whitebox_estimators = lambda: [MaximumSequenceProbability(), Perplexity(), LexicalSimilarity(metric="rougeL")]
_orig = PR.subset_estimators_factory
def small_factory(configs, subset):
    full = _orig(configs, subset)
    if subset == "empirical_baselines":
        return lambda: [e for e in full() if str(e) in ("LexicalSimilarity_rougeL", "LexicalSimilarity_BLEU")]
    return full
PR.subset_estimators_factory = small_factory
# modello minuscolo con un chat template (serve alla parte instruct e a Verbalized2S)
if not os.path.isdir("tiny_chat"):
    shutil.copytree("tiny", "tiny_chat")
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("tiny_chat")
    t.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                       "{% if add_generation_prompt %}assistant:{% endif %}")
    t.save_pretrained("tiny_chat")
_orig_load = M.load_whitebox_model
def load_cpu(*a, **k):
    k["use_quantization"] = False
    k["attn_implementation"] = "sdpa"
    return _orig_load(*a, **k)
M.load_whitebox_model = load_cpu
M.attn_implementation_for = lambda *a, **k: "sdpa"

shutil.rmtree("replica_t", ignore_errors=True)
sys.argv = ["paper_replica.py", "--results_dir", "replica_t", "--cache_dir", "hfcache",
            "--datasets_cache_dir", cache, "--n_test_samples", "5", "--chunk_size", "3",
            "--n_bootstrap", "20", "--parts", "whitebox", "blackbox", "verbalized",
            "--verbalized_variants", "verb_1s_top1", "verb_2s_top1", "ling_1s"]
PR.main()
files = sorted(os.listdir("replica_t"))
for f in ("results_paper_whitebox_instance_stats.csv", "results_paper_blackbox_empirical_baselines_instance_stats.csv",
          "results_paper_blackbox_verb_2s_top1_instance_stats.csv", "paper_replica_comparison.csv",
          "paper_replica_summary.csv", "fig_paper_replica_whitebox.png", "fig_paper_replica_blackbox.png",
          "run_conditions.csv"):
    assert f in files, (f, files)
wbst = pd.read_csv("replica_t/results_paper_whitebox_instance_stats.csv")
assert set(wbst["dataset"]) == {"CoQA", "TriviaQA", "MMLU", "GSM8k"} and wbst["n_instances"].eq(5).all()
comp = pd.read_csv("replica_t/paper_replica_comparison.csv")
assert comp["prr_paper"].notna().sum() > 0
assert {"Verbalized 2S top-1", "Linguistic 1S", "Verbalized 1S top-1", "Maximum Sequence Probability"} <= set(comp["paper_label"])
rc = pd.read_csv("replica_t/run_conditions.csv")
fmt = rc.groupby("model")["prompt_format"].unique().to_dict()
assert list(fmt["Mistral-7B-v0.2-base"]) == ["plain"] and list(fmt["Mistral-7B-v0.2-it"]) == ["chat_template_libreria"], fmt
v2 = pd.read_csv("replica_t/results_paper_blackbox_verb_2s_top1_per_instance.csv")
assert "Verbalized2S_top1" in v2.columns
print("OK 3: replica completa (4 dataset white-box, 3 black-box, varianti verbalized) con confronto e figure")
print("REPLICA OK")

# 4. confronto con il paper su dati sintetici: replica = paper -> tau 1, scarto 0
import make_figures as MF
syn = "replica_syn"
shutil.rmtree(syn, ignore_errors=True); os.makedirs(syn)
ref = PR.load_reference()
rows = []
for setting, model in (("whitebox", PR.WHITEBOX_MODEL[0]), ("blackbox", PR.BLACKBOX_MODEL[0])):
    for _, r in ref[ref["setting"] == setting].iterrows():
        rows.append({"model": model, "dataset": r["dataset"], "estimator": r["paper_label"],
                     "paper_label": r["paper_label"], "prr": r["prr"], "prr_ci_low": r["prr"] - 0.02,
                     "prr_ci_high": r["prr"] + 0.02, "mean_quality": 0.6, "n_instances": 2000})
st = pd.DataFrame(rows)
st[st["model"] == PR.WHITEBOX_MODEL[0]].to_csv(f"{syn}/results_paper_whitebox_instance_stats.csv", index=False)
st[st["model"] == PR.BLACKBOX_MODEL[0]].to_csv(f"{syn}/results_paper_blackbox_empirical_baselines_instance_stats.csv", index=False)
PR.compare_with_paper(syn)
summ = pd.read_csv(f"{syn}/paper_replica_summary.csv")
assert np.allclose(summ["mean_abs_diff"], 0) and summ["kendall_tau_vs_paper"].gt(0.99).all(), summ
assert summ["share_paper_in_ci"].eq(1).all()
assert MF.generate_all(syn) == 0 and os.path.exists(f"{syn}/fig_paper_replica_whitebox.png")
print("OK 4: confronto con il paper (tau 1, scarto 0 quando replica = paper) e figura")
print("REPLICA SINTETICA OK")
