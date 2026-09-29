"""Il campionatore in batch produce statistiche con la stessa struttura e la
stessa definizione di SamplingGenerationCalculator."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
import main as M
from lm_polygraph.stat_calculators import SamplingGenerationCalculator
from batched_sampling import BatchedSamplingCalculator

model = M.load_whitebox_model("tiny", "hfcache", use_quantization=False, attn_implementation="sdpa")
texts = ["Domanda: quale opzione?\nA) uno\nB) due\nRisposta:", "Problema: 3 + 4\nSoluzione:"]
for stop in (None, ["\n"]):
    n_stop = 0
    model.generation_parameters.stop_strings = stop
    torch.manual_seed(0); lib = SamplingGenerationCalculator(10)({}, texts, model, max_new_tokens=15)
    torch.manual_seed(0); mine = BatchedSamplingCalculator(10)({}, texts, model, max_new_tokens=15)
    for k in ("sample_log_likelihoods", "sample_log_probs", "sample_tokens", "sample_texts"):
        assert len(lib[k]) == len(mine[k]) == 2, k
        for a, b in zip(lib[k], mine[k]):
            assert len(a) == len(b) == 10, (k, len(a), len(b))
            assert type(a[0]) == type(b[0]), (k, type(a[0]), type(b[0]))
    # (tolleranza: modello in bf16, cache KV vs forward completo) le log-probabilita registrate = log-softmax del modello sul token scelto (teacher forcing)
    for i, text in enumerate(texts):
        enc = model.tokenizer(text, return_tensors="pt")
        for toks, ll, lp in zip(mine["sample_tokens"][i], mine["sample_log_likelihoods"][i], mine["sample_log_probs"][i]):
            ids = torch.cat([enc["input_ids"], torch.tensor([toks])], dim=1)
            with torch.no_grad():
                logits = model.model(input_ids=ids).logits[0].float().log_softmax(-1)
            L = enc["input_ids"].shape[1]
            tf = [float(logits[L - 1 + j, t]) for j, t in enumerate(toks)]
            assert np.allclose(tf, ll, atol=1e-2), (tf[:3], ll[:3])
            # stessa definizione della libreria: log_prob = somma delle ll, piu' il
            # log p(EOS) solo se il campione e' finito con un EOS vero. Un campione
            # fermato da una stringa di arresto non deve contenere un log p(EOS)
            # spurio dovuto al padding delle sequenze gia' finite.
            if stop and model.tokenizer.decode(toks).endswith("\n"):
                assert abs(lp - sum(ll)) < 1e-4, ("EOS spurio", lp, sum(ll))
                n_stop += 1
            else:
                assert lp <= sum(ll) + 1e-4
    # stop strings rispettate
    if stop:
        assert n_stop > 0, "nessun campione fermato dalla stringa di arresto: il test non verifica nulla"
        assert all("\n" not in t[:-1] for s in mine["sample_texts"] for t in s), mine["sample_texts"][0][:3]
    print("OK stop =", stop, "| esempio:", repr(mine["sample_texts"][0][0]))

# Modelli con piu' token di fine (Gemma 3: <eos> e <end_of_turn>) e padding = <eos>:
# un campione che finisce su un token di fine diverso da <eos> non deve ricevere il
# log p(<eos>) del padding che segue. La libreria, un campione alla volta, non lo ha.
eos = model.tokenizer.eos_token_id
EXTRA = set(range(100, 200))
orig_eos = model.model.generation_config.eos_token_id
model.model.generation_config.eos_token_id = [eos] + sorted(EXTRA)
model.generation_parameters.stop_strings = None
n_extra = 0
for seed in range(3):
    torch.manual_seed(seed); mine = BatchedSamplingCalculator(10)({}, texts[:1], model, max_new_tokens=15)
    torch.manual_seed(seed); lib = SamplingGenerationCalculator(10)({}, texts[:1], model, max_new_tokens=15)
    for r, name in ((mine, "batched"), (lib, "libreria")):
        for toks, ll, lp in zip(r["sample_tokens"][0], r["sample_log_likelihoods"][0], r["sample_log_probs"][0]):
            assert not any(t in EXTRA for t in toks[:-1]), (name, "token dopo la fine", toks)
            if toks[-1] in EXTRA:
                assert abs(lp - sum(ll)) < 1e-4, (name, "EOS spurio con piu' token di fine", lp, sum(ll))
                n_extra += name == "batched"
model.model.generation_config.eos_token_id = orig_eos
assert n_extra > 0, "nessun campione finito su un token di fine secondario: il test non verifica nulla"
print("OK piu' token di fine:", n_extra, "campioni verificati")
print("TEST CAMPIONATORE SUPERATO")
