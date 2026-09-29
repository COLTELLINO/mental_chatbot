# Costruisce un modello Llama minuscolo + tokenizer, salvati su disco, senza download.
import os, torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM
corpus = ["Domanda: quale opzione e' corretta?\nA) uno\nB) due\nRisposta: A\n\nDomanda: altro\nRisposta: B",
          "Rispondi SOLO con la lettera. Problema: 3 + 4\nSoluzione: 7\nRisposta finale: 7\n\nProblema:"] * 50
tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["<pad>", "<eos>", "<unk>"],
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>", model_input_names=["input_ids", "attention_mask"])
cfg = LlamaConfig(vocab_size=len(hf_tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048,
                  pad_token_id=hf_tok.pad_token_id, eos_token_id=hf_tok.eos_token_id, bos_token_id=None)
torch.manual_seed(0)
m = LlamaForCausalLM(cfg)
os.makedirs("tiny", exist_ok=True); m.save_pretrained("tiny"); hf_tok.save_pretrained("tiny")
print("pad", hf_tok.pad_token_id, "eos", hf_tok.eos_token_id, "vocab", len(hf_tok))
