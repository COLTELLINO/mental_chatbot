"""Generatore dei K campioni in una sola chiamata, senza conservare dati inutili.

Sostituisce SamplingGenerationCalculator di lm-polygraph 0.7.0
(lm_polygraph/stat_calculators/sample.py), che:

1. chiama `generate` K volte di fila, una per campione, invece di chiedere
   al modello K sequenze in una sola chiamata (num_return_sequences=K). Nel
   profiling il campionamento costava 7-10 volte la risposta principale;
2. chiede sempre `output_hidden_states=True`: per ogni campione e ogni token
   conserva gli stati di TUTTI gli strati, che servono solo a
   `sample_embeddings`, a sua volta usato solo dai metodi density-based (qui
   esclusi per design);
3. conserva, per ogni token di ogni campione, il log-softmax sull'intero
   vocabolario (262.000 voci per Gemma 3): con K=10 e 200 token sono ~2 GB
   per domanda, quando serve una sola probabilita' per token.

Qui: una sola chiamata con num_return_sequences=K, nessuno stato nascosto, e un
logits processor che tiene in memoria soltanto la distribuzione del passo
precedente e ne estrae subito la log-probabilita' del token scelto. Memoria
costante nella lunghezza della risposta.

Le statistiche prodotte hanno lo stesso nome, la stessa struttura e la stessa
definizione di quelle della libreria (verificato da test_batched_sampling.py):
- stessi parametri di campionamento (quelli di model.generation_parameters,
  con do_sample=True, num_beams=1, min_new_tokens=2, stesse stringhe di arresto);
- la log-probabilita' e' quella della distribuzione originale del modello,
  registrata nella stessa posizione della catena di logits processor in cui la
  registra WhiteboxModel.generate (dopo i processor di default, prima di
  temperatura / top-k / top-p);
- il taglio al primo EOS e la regola "sample_log_probs include la
  log-probabilita' dell'EOS, sample_log_likelihoods no" sono replicati uguali;
- un campione fermato da una stringa di arresto finisce li'. Con K sequenze in
  una sola chiamata, quelle gia' finite vengono allungate con il token di
  padding (qui = EOS) finche' non finisce l'ultima: quel padding non e' stato
  generato dal modello, quindi non entra in nessuna statistica. Senza questo
  accorgimento sample_log_probs conterrebbe un log p(EOS) spurio per ogni
  campione fermato da una stringa di arresto, cosa che la libreria (che genera
  un campione alla volta) non fa. Lo stesso vale per i modelli con piu' di un
  token di fine (Gemma 3 e MedGemma: <eos> e <end_of_turn>): un campione che
  chiude il turno con <end_of_turn> e' finito li', anche se il padding che
  segue e' <eos>.
Unica differenza: `sample_embeddings` e' vuoto (nessuno dei 26 metodi lo usa;
se uno stimatore lo chiedesse, fallirebbe subito in modo visibile).
"""
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
from transformers.generation.stopping_criteria import StopStringCriteria

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator


class _ChosenTokenLogProb:
    """Logits processor: a ogni passo t registra la log-probabilita' (sotto la
    distribuzione del passo t-1) del token appena scelto, che al passo t e'
    l'ultimo di input_ids. Tiene in memoria solo l'ultima distribuzione."""

    def __init__(self):
        self.prev = None
        self.steps = []

    def __call__(self, input_ids=None, scores=None):
        if self.prev is not None:
            chosen = input_ids[:, -1:]
            self.steps.append(self.prev.gather(1, chosen).squeeze(1).float())
        self.prev = scores.float().log_softmax(-1)
        return scores

    def finish(self, sequences):
        if self.prev is not None:
            self.steps.append(self.prev.gather(1, sequences[:, -1:]).squeeze(1).float())
            self.prev = None
        if not self.steps:
            return torch.empty((sequences.shape[0], 0))
        return torch.stack(self.steps, dim=1).cpu()


class _RecordStop(StoppingCriteria):
    """Avvolge un criterio di arresto e registra, per ogni sequenza, la
    lunghezza (prompt incluso) al primo passo in cui il criterio scatta. Dopo
    quel passo generate() aggiunge solo padding."""

    def __init__(self, inner):
        self.inner = inner
        self.stop_len = None

    def __call__(self, input_ids, scores, **kwargs):
        done = self.inner(input_ids, scores, **kwargs)
        done_cpu = torch.as_tensor(done).reshape(-1).bool().cpu()
        if self.stop_len is None:
            self.stop_len = torch.full((input_ids.shape[0],), -1, dtype=torch.long)
        newly = done_cpu & (self.stop_len < 0)
        self.stop_len[newly] = input_ids.shape[1]
        return done


class _SanitizeLogits:
    """Come WhiteboxModel._SanitizeLogitsProcessor: sostituisce inf/NaN con
    valori finiti, altrimenti torch.multinomial fallisce."""

    def __call__(self, input_ids=None, scores=None):
        if torch.isfinite(scores).all():
            return scores
        finite = torch.isfinite(scores)
        big = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
        small = torch.tensor(float("inf"), dtype=scores.dtype, device=scores.device)
        row_max = torch.where(finite, scores, big).max(dim=-1, keepdim=True).values
        row_min = torch.where(finite, scores, small).min(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
        scores = torch.where(torch.isposinf(scores), row_max, scores)
        scores = torch.where(torch.isneginf(scores), row_min, scores)
        return torch.nan_to_num(scores, nan=0.0)


class BatchedSamplingCalculator(StatCalculator):
    """Stesse statistiche di SamplingGenerationCalculator, calcolate in una
    sola chiamata a generate(). Vedi il docstring del modulo."""

    @staticmethod
    def meta_info():
        return [
            "sample_log_probs",
            "sample_tokens",
            "sample_texts",
            "sample_log_likelihoods",
            "sample_embeddings",
        ], []

    def __init__(self, samples_n: int = 10):
        super().__init__()
        self.samples_n = samples_n

    def __call__(self, dependencies: Dict[str, np.ndarray], texts: List[str], model,
                 max_new_tokens: int = 100) -> Dict[str, list]:
        batch = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        input_len = batch["input_ids"].shape[1]

        gp = asdict(model.generation_parameters)
        stop_strings = gp.get("stop_strings")
        recorder = _ChosenTokenLogProb()
        kwargs = dict(
            do_sample=True,
            num_beams=1,
            num_return_sequences=self.samples_n,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            temperature=gp["temperature"],
            top_k=gp["top_k"],
            top_p=gp["top_p"],
            repetition_penalty=gp["repetition_penalty"],
            return_dict_in_generate=True,
            output_scores=False,
            output_hidden_states=False,
            output_attentions=False,
            logits_processor=LogitsProcessorList([_SanitizeLogits(), recorder]),
        )
        if not gp.get("allow_newlines", True):
            kwargs["suppress_tokens"] = [t for t in range(len(model.tokenizer))
                                         if "\n" in model.tokenizer.decode([t])]
        stop_recorder = None
        if stop_strings:
            stop_recorder = _RecordStop(
                StopStringCriteria(stop_strings=list(stop_strings), tokenizer=model.tokenizer))
            kwargs["stopping_criteria"] = StoppingCriteriaList([stop_recorder])

        with torch.no_grad():
            out = model.model.generate(**batch, **kwargs)
        sequences = out.sequences.cpu()
        token_logprobs = recorder.finish(out.sequences)
        del out

        eos = model.tokenizer.eos_token_id
        # Tutti i token che chiudono la generazione: quello del tokenizer piu'
        # quelli di generation_config (per Gemma 3 anche <end_of_turn>).
        end_ids = {eos}
        gen_eos = getattr(getattr(model.model, "generation_config", None), "eos_token_id", None)
        if gen_eos is not None:
            end_ids |= {int(t) for t in np.atleast_1d(gen_eos)}
        end_ids.discard(None)
        n_inputs = len(texts)
        log_probs = [[] for _ in range(n_inputs)]
        tokens = [[] for _ in range(n_inputs)]
        sample_texts = [[] for _ in range(n_inputs)]
        log_likelihoods = [[] for _ in range(n_inputs)]
        gen = sequences[:, input_len:]
        for i in range(gen.shape[0]):
            owner = i // self.samples_n
            n_valid = gen.shape[1]
            if stop_recorder is not None and stop_recorder.stop_len is not None \
                    and stop_recorder.stop_len[i] >= 0:
                n_valid = int(stop_recorder.stop_len[i]) - input_len
            # dopo il primo token di fine c'e' solo padding
            for j in range(n_valid):
                if int(gen[i, j]) in end_ids:
                    n_valid = j + 1
                    break
            log_prob, ll, toks = 0.0, [], []
            for j in range(n_valid):
                cur = int(gen[i, j])
                lp = float(token_logprobs[i, j])
                log_prob += lp
                if cur == eos:
                    break
                ll.append(lp)
                toks.append(cur)
            if toks:
                log_likelihoods[owner].append(ll)
                log_probs[owner].append(log_prob)
                tokens[owner].append(toks)
                sample_texts[owner].append(model.tokenizer.decode(toks))
            # come la libreria: un campione vuoto viene saltato

        return {
            "sample_log_likelihoods": log_likelihoods,
            "sample_log_probs": log_probs,
            "sample_tokens": tokens,
            "sample_texts": sample_texts,
            "sample_embeddings": [[] for _ in range(n_inputs)],
        }


def load_stat_calculator(cfg, env):
    """Builder richiesto da lm-polygraph (StatCalculatorContainer.builder)."""
    return BatchedSamplingCalculator(samples_n=int(cfg.get("samples_n", 10)))
