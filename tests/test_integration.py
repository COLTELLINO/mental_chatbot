import os, sys, shutil, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from transformers import LlamaForCausalLM, AutoTokenizer
import main as M
from dataset_prep import MCQAccuracyMetric, GSM8kAccuracyMetric
from lm_polygraph.estimators import (MaximumSequenceProbability, Perplexity, MeanTokenEntropy,
                                     MonteCarloSequenceEntropy, LexicalSimilarity)
from lm_polygraph.estimators.estimator import Estimator

tok = AutoTokenizer.from_pretrained("tiny")
SCRIPT = tok.encode(" A\n\nDomanda: altro\nRisposta: B\n\nDomanda: altro\nRisposta: A\n\nDomanda:")

# --- logit pilotati: il modello "vuole" rispondere A, andare a capo e inventare un nuovo esempio
_orig_forward = LlamaForCausalLM.forward
state = {"start": None}
def scripted_forward(self, *a, **kw):
    out = _orig_forward(self, *a, **kw)
    ids = kw.get("input_ids", a[0] if a else None)
    cp = kw.get("cache_position")
    if ids is None or cp is None:
        return out
    if ids.shape[1] > 1:
        state["start"] = int(cp[-1]) + 1
    step = int(cp[-1]) + 1 - state["start"]
    if 0 <= step < len(SCRIPT):
        out.logits[:, -1, SCRIPT[step]] += 12.0
    return out
LlamaForCausalLM.forward = scripted_forward

class Boom(Estimator):
    """Fallisce quando nel batch c'e' un prompt con 'BOOM' (errore di singola istanza)."""
    def __init__(self): super().__init__(["greedy_log_likelihoods", "input_texts"], "sequence")
    def __str__(self): return "Boom"
    def __call__(self, stats):
        if any("BOOM" in t for t in stats["input_texts"]):
            raise RuntimeError("errore simulato su una singola istanza")
        return np.array([-np.sum(l) for l in stats["greedy_log_likelihoods"]])

class OOMIfBatch(Estimator):
    """Simula un OOM quando il batch di generazione e' > 1."""
    def __init__(self): super().__init__(["greedy_log_likelihoods", "input_texts"], "sequence")
    def __str__(self): return "OOMIfBatch"
    def __call__(self, stats):
        if len(stats["input_texts"]) > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulato)")
        return np.array([-np.sum(l) for l in stats["greedy_log_likelihoods"]])

def estimators():
    return [MaximumSequenceProbability(), Perplexity(), MeanTokenEntropy(),
            MonteCarloSequenceEntropy(), LexicalSimilarity(metric="rougeL")]

def make_args(rd, **kw):
    a = argparse.Namespace(batch_size=2, cache_dir="hfcache", max_rejection=0.5, n_bootstrap=50,
                           results_dir=rd, chunk_size=0, no_resume=False)
    for k, v in kw.items(): setattr(a, k, v)
    os.makedirs(rd, exist_ok=True)
    return a

examples = [{"content": f"Domanda: domanda numero {i}\nA) uno\nB) due", "reference": "A" if i % 3 else "B"}
            for i in range(7)]
cfg = {"max_new_tokens": 12, "plain_suffix": "\nRisposta:", "generation_metric_factory": MCQAccuracyMetric,
       "stop_strings": ["\n"], "continuation_stop_strings": ["\nDomanda:"]}

model = M.load_whitebox_model("tiny", "hfcache", use_quantization=False, attn_implementation="sdpa")
assert model.model.generation_config.pad_token_id == tok.eos_token_id, "pad != eos"
print("OK pad_token_id di generazione = eos")

def run(rd, key="stop_strings", exs=examples, **argkw):
    return M.run_model_on_dataset(model, "tiny", "DS", exs, cfg, make_args(rd, **argkw), {},
                                  use_chat_template=False, estimators_factory=estimators,
                                  stop_strings_key=key)

def generations(rd):
    return pd.read_csv(os.path.join(rd, "sample_generations.csv"))["generazione"].tolist()

# T1: arresto al primo a capo, batch 2 (sequenze che finiscono insieme o no), nessun <pad> nel testo
shutil.rmtree("r1", ignore_errors=True)
prr1, tim1, st1, pi1 = run("r1")
g = generations("r1"); print("T1 generazioni:", g[:3])
assert all(x.rstrip("\n") == " A" for x in g), g
assert not any("<pad>" in x for x in g)
assert pi1.shape[0] == 7 and st1["mean_quality"].iloc[0] > 0
print("OK T1: generazione fermata al primo a capo, niente <pad>, accuracy", round(st1["mean_quality"].iloc[0], 3))

# T2: senza arresto (chiave assente) il modello inventa il nuovo esempio -> il difetto originale
shutil.rmtree("r2", ignore_errors=True)
run("r2", key="chiave_assente")
g2 = generations("r2"); print("T2 generazioni:", g2[:1])
assert any("Domanda" in x for x in g2)
print("OK T2: senza stringhe di arresto riappare la coda inventata")

# T3: verbalized-like, arresto solo sulla continuazione
shutil.rmtree("r3", ignore_errors=True)
run("r3", key="continuation_stop_strings")
g3 = generations("r3"); print("T3 generazioni:", g3[:1])
assert all(x.endswith("\nDomanda:") and x.count("Domanda") == 1 for x in g3), g3
print("OK T3: arresto sull'esempio inventato, righe successive conservate")

# T4: blocchi == esecuzione unica (stimatori deterministici), e ripresa da checkpoint
shutil.rmtree("r4", ignore_errors=True)
prrC, timC, stC, piC = M.run_cell_chunked("main", model, "tiny", "DS", examples, cfg,
                                          make_args("r4", chunk_size=3), {}, use_chat_template=False,
                                          estimators_factory=estimators)
det = ["MaximumSequenceProbability", "Perplexity", "MeanTokenEntropy"]
a = prr1.set_index(["estimator", "ue_metric"])["value"]; b = prrC.set_index(["estimator", "ue_metric"])["value"]
assert set(a.index) == set(b.index), (set(a.index) ^ set(b.index))
for (e, um) in a.index:
    if e in det:
        assert np.isclose(a[(e, um)], b[(e, um)]), (e, um, a[(e, um)], b[(e, um)])
print("   ue_metric confrontate:", sorted(set(prrC["ue_metric"])))
assert np.allclose(pi1[det].to_numpy(), piC[det].to_numpy())
assert timC["n_instances"].iloc[0] == 7
nfiles = len(os.listdir("r4/chunks/main/tiny__DS__n7_c3"))
print("OK T4: PRR a blocchi identico all'esecuzione unica;", nfiles, "file di checkpoint")
calls = {"n": 0}
orig = M.run_model_on_dataset
def counting(*a, **k):
    calls["n"] += 1; return orig(*a, **k)
M.run_model_on_dataset = counting
prrR, _, _, _ = M.run_cell_chunked("main", model, "tiny", "DS", examples, cfg,
                                   make_args("r4", chunk_size=3), {}, use_chat_template=False,
                                   estimators_factory=estimators)
M.run_model_on_dataset = orig
assert calls["n"] == 0 and np.allclose(prrR["value"].astype(float), prrC["value"].astype(float), equal_nan=True)
print("OK T4b: seconda esecuzione interamente ripresa dai checkpoint (0 blocchi ricalcolati)")

# T5: errore su una singola istanza -> blocco rieseguito istanza per istanza, istanza saltata e registrata
shutil.rmtree("r5", ignore_errors=True)
ex5 = [dict(e) for e in examples]; ex5[4]["content"] += " BOOM"
prr5, _, st5, pi5 = M.run_cell_chunked("main", model, "tiny", "DS", ex5, cfg,
                                       make_args("r5", chunk_size=3), {}, use_chat_template=False,
                                       estimators_factory=lambda: estimators() + [Boom()])
assert pi5.shape[0] == 6 and st5["n_skipped_instances"].iloc[0] == 1
print("OK T5: istanza difettosa saltata e registrata, 6/7 istanze nella cella")

# T6: errore sistematico -> cella abbandonata, non riempita a meta'
shutil.rmtree("r6", ignore_errors=True)
ex6 = [dict(e, content=e["content"] + " BOOM") for e in examples] * 2
res6 = M.run_cell_chunked("main", model, "tiny", "DS", ex6, cfg, make_args("r6", chunk_size=3), {},
                          use_chat_template=False, estimators_factory=lambda: estimators() + [Boom()])
assert all(r is None for r in res6)
print("OK T6: errore sistematico -> cella abbandonata in modo esplicito")

# T7: OOM con batch 2 -> nuovo tentativo con batch 1 riuscito
shutil.rmtree("r7", ignore_errors=True)
prr7, _, st7, pi7 = run("r7", key="stop_strings") if False else M.run_model_on_dataset(
    model, "tiny", "DS", examples, cfg, make_args("r7", batch_size=2), {}, use_chat_template=False,
    estimators_factory=lambda: estimators() + [OOMIfBatch()])
assert prr7 is not None and "OOMIfBatch" in set(prr7["estimator"]) and pi7.shape[0] == 7
print("OK T7: OOM simulato -> ripetuto con batch 1, cella completa")

# T8: estrattore GSM8k
m = GSM8kAccuracyMetric()
assert m._extract_number("5 + 3 = 8. The answer is 8.\n\nQuestion: Luca has 12 apples and 4 pears") == 8.0
assert m._extract_number("il totale e' 1,250") == 1250.0
print("OK T8: GSM8k legge 'The answer is' e non l'ultimo numero del problema inventato")
# T9: file per-istanza con indice dell'istanza e testo generato; valori PRR = PRR normalizzato
assert {"instance_index", "greedy_text"} <= set(pi1.columns), pi1.columns
assert list(pi1["instance_index"]) == list(range(7))
assert list(piC["instance_index"]) == list(range(7)), piC["instance_index"].tolist()
import analysis_lib as AL
row = prr1[(prr1["estimator"] == "Perplexity") & (prr1["ue_metric"] == "prr_0.5_normalized")]["value"].iloc[0]
assert np.isclose(row, AL.prr_normalized(pi1["Perplexity"], pi1["quality"]), equal_nan=True)
assert "prr_raw" in st1.columns and "error_rate_top10" in st1.columns
print("OK T9: indice istanza e testo generato salvati; PRR = PRR normalizzato ricalcolato")
print("\nTUTTI I TEST SUPERATI")
