"""Test delle correzioni del 29/09 che non passano dal modello minuscolo."""
import os, sys, json, shutil, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import datasets as HD
import analysis_lib as AL
import dataset_prep as D
import paired_comparisons as PC

# 1. PRR: casuale -> 0, oracolo -> 1, pareggi, equivalenza con lm-polygraph
AL.check_prr_implementation()
print("OK 1: PRR normalizzato (casuale ~0, oracolo 1, invertito <0, pareggi in valore atteso)")

# 2. doppio BOS
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
from transformers import PreTrainedTokenizerFast
tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder = decoders.ByteLevel()
tok.train_from_iterator(["hello world user model"] * 20, trainers.BpeTrainer(
    vocab_size=280, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<start_of_turn>", "<end_of_turn>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 1)])
hf = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>")
hf.chat_template = ("{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}\n{{ m['content'] }}"
                    "<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")
assert D.tokenizer_adds_bos(hf)
text = D.format_chat_prompt(hf, "hello")
ids = hf(text)["input_ids"]
assert ids.count(hf.bos_token_id) == 1 and ids[0] == hf.bos_token_id, ids[:5]
raw = hf.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False, add_generation_prompt=True)
assert hf(raw)["input_ids"][:2] == [hf.bos_token_id] * 2  # il difetto che correggiamo
print("OK 2: un solo BOS dopo il chat template (prima erano due)")

# 3. prompt inglesi con i template ufficiali, su dataset finti
def fake_load(name, config=None, cache_dir=None):
    if "coqa" in name:
        v = HD.Dataset.from_dict({"story": ["Tom has a dog."] * 3, "questions": [["Who?", "What?"]] * 3,
                                  "answers": [{"input_text": ["Tom", "a dog"]}] * 3})
        return {"validation": v}
    if "trivia" in name:
        row = lambda q: {"question": q, "answer": {"value": "Paris", "normalized_value": "paris",
                                                   "aliases": ["Paris", "Paris, France"]}}
        d = HD.Dataset.from_list([row(f"Capital {i}?") for i in range(10)])
        return {"train": d, "validation": d}
    if "mmlu" in name:
        rows = [{"question": f"Q{s}{i}", "subject": s, "choices": ["w", "x", "y", "z"], "answer": i % 4}
                for s in ("anatomy", "law", "virology") for i in range(10)]
        d = HD.Dataset.from_list(rows)
        return {"dev": d, "test": d}
    if "gsm8k" in name:
        d = HD.Dataset.from_list([{"question": f"2+{i}?", "answer": f"2+{i}={2+i}\n#### {2+i}"} for i in range(5)])
        return {"train": d, "test": d}
    raise KeyError(name)
D.load_dataset = fake_load
co = D.prepare_coqa(3, 1)[0]
assert co["content"].startswith("Here's a short story:") and co["content"].endswith("Question: What?\nAnswer: ")
assert "Question: Who?\nAnswer: Tom" in co["content"] and co["reference"] == "a dog"
tq = D.prepare_triviaqa(4, 1)
assert tq[0]["content"].startswith("Answer the following question as briefly as possible.")
assert tq[0]["content"].endswith("\nAnswer: ") and tq[0]["reference"] == ["Paris", "Paris, France"]
mm = D.prepare_mmlu(6, 1)
assert sorted(e["subject"] for e in mm) == ["anatomy", "anatomy", "law", "law", "virology", "virology"]
assert mm[0]["content"].endswith("D. z\nAnswer:") and "Here are a few examples" in mm[0]["content"]
gs = D.prepare_gsm8k(2, 1)[0]
assert 'end with "The answer is [answer]"' in gs["content"] and gs["content"].endswith("\nAnswer:")
for e in (co, tq[0], mm[0], gs):
    for w in ("Domanda", "Risposta", "Rispondi", "Problema"):
        assert w not in e["content"], (w, e["content"][:80])
v = D.build_verbalized_content(tq[0]["content"], "numeric")
assert v.endswith("Question: Capital " + tq[0]["content"].split("Question: Capital ")[-1].split("\n")[0] + "\n" +
                  "Answer: ") or v.rstrip().endswith("Answer:")
assert v.index("Confidence:") < v.rindex("Answer:")
print("OK 3: prompt in inglese dai template ufficiali; TriviaQA con alias; MMLU stratificato; "
      "richiesta di confidenza prima della riga 'Answer:'")

# 4. riferimenti multipli: massimo sugli alias
class Exact(D.GenerationMetric):
    def __init__(self): super().__init__(["greedy_texts"], "sequence")
    def __str__(self): return "Exact"
    def __call__(self, stats, target_texts):
        return np.array([float(p.strip() == t) for p, t in zip(stats["greedy_texts"], target_texts)])
m = D.MaxOverReferences(Exact())
out = m({"greedy_texts": ["Paris, France", "Rome"]}, [["Paris", "Paris, France"], ["Paris"]])
assert list(out) == [1.0, 0.0], out
print("OK 4: TriviaQA multiref = massimo sugli alias")

# 5. precisione dell'ancora e controllo del confronto quantizzazione
import main as M
M.set_anchor_precision("bf16"); assert not M.uses_quantization("Mistral-7B-it") and M.uses_quantization("LFM2-1.2B")
M.set_anchor_precision("4bit"); assert M.uses_quantization("Mistral-7B-it")
assert not M.uses_quantization("Gemma3-4B-it")
assert "Mistral-7B-it" not in M.CANNOT_QUANTIZE_MODELS and "LFM2-350M" in M.CHAT_TEMPLATE_MODELS
assert M.QUANT_COMPARE_MODEL_DEFAULT == "LFM2-1.2B"
print("OK 5: ancora bf16 di default (4bit su richiesta); il confronto quantizzazione non esclude piu' Mistral; LFM2 con chat template")

# 6. attenzione: eager solo con attn_logit_softcapping
from transformers import LlamaConfig
for name, cap in (("cfg_cap", 50.0), ("cfg_nocap", None)):
    c = LlamaConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=8)
    c.attn_logit_softcapping = cap
    c.save_pretrained(name)
assert M.attn_implementation_for("x", "cfg_cap") == "eager"
assert M.attn_implementation_for("x", "cfg_nocap") == "sdpa"
print("OK 6: attenzione eager solo se il modello dichiara attn_logit_softcapping")

# 7. confronti appaiati: migliore dentro i ricampionamenti, Bonferroni, non testabile
rng = np.random.default_rng(0); n = 400
q = (rng.random(n) < 0.5).astype(float)
good = -q + rng.normal(0, 0.8, n)
cell = pd.DataFrame({"model": "m", "dataset": "d", "quality": q, "good": good,
                     "good_twin": good + rng.normal(0, 0.05, n), "random": rng.random(n),
                     "const": np.ones(n)})
rows = pd.DataFrame(PC.compare_cell(cell, 300, np.random.default_rng(1))).set_index("method")
assert rows.loc["random", "verdict"] == "peggiore", rows["verdict"]
assert abs(rows.loc["const", "prr"]) < 1e-12 and rows.loc["const", "verdict"] == "peggiore"  # costante = caso
print("OK 7: confronti appaiati ->", rows["verdict"].to_dict())

# 8. errori fra le risposte piu' confidenti: non ha il tetto del vecchio silent failure rate
q = np.r_[np.zeros(900), np.ones(100)]
assert AL.error_rate_most_confident(np.r_[np.zeros(900) + 1, np.zeros(100)], q) == 0.0
assert np.isclose(AL.error_rate_overall(q), 0.9)
print("OK 8: error rate fra le risposte piu' confidenti (0 per un metodo perfetto anche con 90% di errori)")

# 9. MCQ: il bug dell'articolo "a"
assert D.self_test_mcq_metric() >= 20
print("OK 9: estrazione della lettera MCQ (compresi 'I think it is a C' e 'A patient with... B')")

# 10. scheda delle condizioni e impronta, dall'orchestrazione
rc = pd.read_csv("orch/run_conditions.csv")
for col in ("precision", "attention", "prompt_format", "max_new_tokens", "stop_strings", "sampler",
            "version_lm-polygraph"):
    assert col in rc.columns, col
assert json.load(open("orch/run_fingerprint.json"))["fingerprint"]
ns = argparse.Namespace(results_dir="orch", no_resume=False, force_resume=False, max_rejection=0.5,
                        n_test_samples=None, sampler="batched", anchor_precision="bf16",
                        verbalized_max_new_tokens=41)   # impostazione diversa
try:
    M.check_run_fingerprint(ns); raise AssertionError("ripresa non rifiutata")
except SystemExit as e:
    assert "Ripresa rifiutata" in str(e), e
ns.force_resume = True; M.check_run_fingerprint(ns)
print("OK 10: run_conditions.csv scritto; ripresa con impostazioni diverse rifiutata (salvo --force_resume)")

# 11. ripresa dopo --no_resume: niente checkpoint della versione precedente
import tempfile
tmp = tempfile.mkdtemp()
os.makedirs(os.path.join(tmp, "chunks", "main", "LFM2-350M__CoQA__n500_c100"))
open(os.path.join(tmp, "chunks", "main", "LFM2-350M__CoQA__n500_c100", "chunk0000_prr.csv"), "w").write("x\n1\n")
pd.DataFrame({"a": [1]}).to_csv(os.path.join(tmp, "results_severity_grid_mapped.csv"), index=False)
open(os.path.join(tmp, "fig_old.png"), "w").write("x")
ns = argparse.Namespace(results_dir=tmp, no_resume=True, force_resume=False, max_rejection=0.5,
                        n_test_samples=None, sampler="batched", anchor_precision="bf16",
                        verbalized_max_new_tokens=40)
fp = M.check_run_fingerprint(ns)["fingerprint"]
left = sorted(os.listdir(tmp))
assert left == ["_superati", "run_fingerprint.json"], left   # tutto spostato, niente cancellato
moved = os.path.join(tmp, "_superati", os.listdir(os.path.join(tmp, "_superati"))[0])
assert os.path.exists(os.path.join(moved, "chunks", "main", "LFM2-350M__CoQA__n500_c100", "chunk0000_prr.csv"))
assert os.path.exists(os.path.join(moved, "results_severity_grid_mapped.csv"))
# una sola sezione basta perche' la cartella conti come "con risultati"
ns2 = argparse.Namespace(**{**vars(ns), "no_resume": False, "verbalized_max_new_tokens": 41})
pd.DataFrame({"a": [1]}).to_csv(os.path.join(tmp, "results_verbalized_numeric_mapped.csv"), index=False)
try:
    M.check_run_fingerprint(ns2); raise AssertionError("ripresa non rifiutata con solo una sezione")
except SystemExit:
    pass
# blocchi senza impronta (o con un'impronta diversa) non vengono ripresi
ns.run_fingerprint = fp; ns.no_resume = False
cdir = os.path.join(tmp, "chunks", "quant", "cella")
os.makedirs(cdir); open(os.path.join(cdir, "chunk0000_prr.csv"), "w").write("x\n1\n")
M.prepare_chunk_dir(ns, cdir)
assert os.listdir(cdir) == ["run_fingerprint.txt"], os.listdir(cdir)
open(os.path.join(cdir, "chunk0000_prr.csv"), "w").write("x\n1\n")
M.prepare_chunk_dir(ns, cdir)                      # stessa impronta: blocchi tenuti
assert "chunk0000_prr.csv" in os.listdir(cdir)
ns.run_fingerprint = "diversa"
M.prepare_chunk_dir(ns, cdir)
assert "chunk0000_prr.csv" not in os.listdir(cdir)
shutil.rmtree(tmp, ignore_errors=True)
print("OK 11: --no_resume sposta da parte tutti i risultati precedenti; blocchi di un'altra versione non ripresi")

# 12. confronto quantizzazione: le celle portano il nome del modello
la = [v for v, _ in M.quant_variants("LFM2-1.2B")]; lb = [v for v, _ in M.quant_variants("Mistral-7B-it")]
assert not set(la) & set(lb) and all("LFM2-1.2B" in v for v in la), (la, lb)
print("OK 12: le varianti del confronto quantizzazione sono distinte per modello:", la, lb)

# 13. punteggi NaN: error rate fra i piu' confidenti con la stessa convenzione del PRR
q = np.r_[np.zeros(50), np.ones(50)]                 # 50% di errori
ue = np.r_[np.full(40, np.nan), np.linspace(0, 1, 10), np.linspace(-1, 0, 50)]
e = AL.error_rate_most_confident(ue, q)
assert e == 1.0, e      # i 10 piu' "confidenti" sono parse failure su risposte sbagliate
assert AL.prr_normalized(ue, q, 0.5) < 0
print("OK 13: parse failure trattati come massima confidenza anche nell'error rate (", e, ")")
print("\nTEST DELLE CORREZIONI SUPERATI")
