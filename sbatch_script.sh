#!/bin/bash
set -u

# Le credenziali NON vanno scritte qui: questo file e' tracciato da git, quindi
# una chiave incollata dentro finisce nella storia del repository e resta
# leggibile anche dopo essere stata cancellata dal file. Vanno esportate nella
# propria shell prima di lanciare lo script:
#
#   export WANDB_API_KEY="..."   # opzionale, solo per il logging su W&B
#   export HF_TOKEN="..."        # necessario per i modelli gated
#                                # (google/medgemma-4b-it, google/gemma-3-4b-it:
#                                #  accettare prima la licenza su huggingface.co)
#
# In alternativa, tenerle in un file non tracciato e caricarlo:
#   [ -f ~/.uq_env ] && source ~/.uq_env

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ATTENZIONE: HF_TOKEN non impostato. I modelli gated falliranno al" >&2
    echo "caricamento e il run produrra' risultati incompleti." >&2
fi

sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh