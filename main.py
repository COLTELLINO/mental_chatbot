import os
import argparse
import torch
import pandas as pd
import wandb

from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# Argomenti da linea di comando (passati da train.sh via sbatch)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="MentalChat fine-tuning")
parser.add_argument("--model_checkpoint", type=str, default="unsloth/Qwen3-0.6B")
parser.add_argument("--dataset",          type=str, default="ShenLab/MentalChat16K")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------
EVAL_RATIO         = 0.30
DATASET_PARTITION  = 0.01
SEED               = 3407
MAX_STEPS          = 10
LEARNING_RATE      = 5e-5
BATCH_SIZE         = 1
GRAD_ACCUM_STEPS   = 1
SAVE_STEPS         = 5

# I checkpoint vengono salvati dentro /workspace (montato via -v in run_docker.sh)
CHECKPOINT_DIR = "/workspace/mentalchat_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Caricamento modello
# ---------------------------------------------------------------------------
print(f"[INFO] Caricamento modello: {args.model_checkpoint}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = args.model_checkpoint,
    max_seq_length = 1024,
    full_finetuning = True,
    qat_scheme     = "phone-deployment",
)

# ---------------------------------------------------------------------------
# Caricamento dataset
# ---------------------------------------------------------------------------
print(f"[INFO] Caricamento dataset: {args.dataset}")
dataset = load_dataset(args.dataset, split="train")

def generate_conversation(examples):
    conversations = []
    for p, s in zip(examples["input"], examples["output"]):
        if p is None or s is None:
            conversations.append([
                {"role": "user",      "content": ""},
                {"role": "assistant", "content": ""},
            ])
        else:
            conversations.append([
                {"role": "user",      "content": str(p)},
                {"role": "assistant", "content": str(s)},
            ])
    return {"conversations": conversations}

conversations = tokenizer.apply_chat_template(
    list(dataset.map(generate_conversation, batched=True)["conversations"]),
    tokenize=False,
)

# ---------------------------------------------------------------------------
# Split train / eval
# ---------------------------------------------------------------------------
data = pd.Series(conversations)
data.name = "text"

if DATASET_PARTITION < 1.0:
    data = data.sample(frac=DATASET_PARTITION, random_state=SEED).reset_index(drop=True)

eval_size     = int(len(data) * EVAL_RATIO)
data_shuffled = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

eval_data  = data_shuffled.iloc[:eval_size]
train_data = data_shuffled.iloc[eval_size:]

print(f"[INFO] Train: {len(train_data)} | Eval: {len(eval_data)}")

train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
eval_dataset  = Dataset.from_pandas(pd.DataFrame(eval_data))

# ---------------------------------------------------------------------------
# WandB  —  la chiave viene letta dalla variabile d'ambiente WANDB_API_KEY
# (impostata in run_docker.sh, non hardcoded qui)
# ---------------------------------------------------------------------------
wandb.login(key=os.environ.get("WANDB_API_KEY"))
wandb.init(
    project = "mentalchat-finetune",
    name    = f"run-{args.model_checkpoint.replace('/', '-')}",
    config  = {
        "model_checkpoint":  args.model_checkpoint,
        "dataset":           args.dataset,
        "eval_ratio":        EVAL_RATIO,
        "dataset_partition": DATASET_PARTITION,
        "learning_rate":     LEARNING_RATE,
        "max_steps":         MAX_STEPS,
        "batch_size":        BATCH_SIZE,
        "grad_accum_steps":  GRAD_ACCUM_STEPS,
    },
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = train_dataset,
    eval_dataset  = eval_dataset,
    args          = SFTConfig(
        dataset_text_field          = "text",
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        warmup_steps                = 5,
        max_steps                   = MAX_STEPS,
        learning_rate               = LEARNING_RATE,
        logging_steps               = 1,
        eval_strategy               = "steps",
        eval_steps                  = SAVE_STEPS,
        optim                       = "adamw_8bit",
        weight_decay                = 0.001,
        lr_scheduler_type           = "linear",
        seed                        = SEED,
        report_to                   = "wandb",
        output_dir                  = CHECKPOINT_DIR,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
    ),
)

# ---------------------------------------------------------------------------
# Info GPU
# ---------------------------------------------------------------------------
gpu_stats          = torch.cuda.get_device_properties(0)
start_gpu_memory   = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
max_memory         = round(gpu_stats.total_memory / 1024 ** 3, 3)
print(f"[INFO] GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"[INFO] {start_gpu_memory} GB of memory reserved.")

# ---------------------------------------------------------------------------
# Resume da checkpoint se disponibile
# ---------------------------------------------------------------------------
def find_valid_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = [
        os.path.join(checkpoint_dir, d)
        for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        print("[INFO] Nessun checkpoint trovato, training da zero.")
        return None

    checkpoints = sorted(checkpoints, key=os.path.getmtime, reverse=True)
    required_files = ["trainer_state.json", "config.json"]

    for ckpt in checkpoints:
        missing = [f for f in required_files if not os.path.isfile(os.path.join(ckpt, f))]
        if missing:
            print(f"[WARN] Checkpoint non valido (mancano {missing}): {ckpt}")
        else:
            print(f"[INFO] Checkpoint valido trovato: {ckpt}")
            return ckpt

    print("[INFO] Nessun checkpoint valido trovato, training da zero.")
    return None


last_checkpoint = find_valid_checkpoint(CHECKPOINT_DIR)
trainer_stats   = trainer.train(resume_from_checkpoint=last_checkpoint)

# ---------------------------------------------------------------------------
# Statistiche finali
# ---------------------------------------------------------------------------
used_memory            = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
used_memory_for_train  = round(used_memory - start_gpu_memory, 3)
used_percentage        = round(used_memory / max_memory * 100, 3)
train_percentage       = round(used_memory_for_train / max_memory * 100, 3)

print(f"[INFO] Training completato in {trainer_stats.metrics['train_runtime']:.1f} secondi "
      f"({trainer_stats.metrics['train_runtime']/60:.2f} minuti).")
print(f"[INFO] Peak reserved memory = {used_memory} GB ({used_percentage}% of max).")
print(f"[INFO] Peak reserved memory for training = {used_memory_for_train} GB ({train_percentage}% of max).")

wandb.finish()

# ---------------------------------------------------------------------------
# Salvataggio modello per deployment su telefono (TorchAO + ExecuTorch)
# ---------------------------------------------------------------------------
EXPORT_DIR = "/workspace/phone_model"
print(f"[INFO] Salvataggio modello quantizzato in {EXPORT_DIR} ...")
model.save_pretrained_torchao(EXPORT_DIR, tokenizer=tokenizer)

# Conversione pesi per ExecuTorch
os.system(
    f"python3 -m executorch.examples.models.qwen3.convert_weights "
    f"{EXPORT_DIR} /workspace/pytorch_model_converted.bin"
)

# Download config
os.system(
    "curl -L -o /workspace/0.6B_config.json "
    "https://raw.githubusercontent.com/pytorch/executorch/main/"
    "examples/models/qwen3/config/0_6b_config.json"
)

# Export .pte
os.system(
    "python3 -m executorch.examples.models.llama.export_llama "
    "--model qwen3_0_6b "
    "--checkpoint /workspace/pytorch_model_converted.bin "
    "--params /workspace/0.6B_config.json "
    "--output_name /workspace/qwen3_0.6B_model.pte "
    "-kv --use_sdpa_with_kv_cache -X --xnnpack-extended-ops "
    "--max_context_length 1024 --max_seq_length 128 --dtype fp32 "
    '--metadata \'{"get_bos_id":199999, "get_eos_ids":[200020,199999]}\''
)

print("[INFO] Export completato.")
os.system("ls -lh /workspace/qwen3_0.6B_model.pte")