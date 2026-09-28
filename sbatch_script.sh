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

# Gli argomenti di questo script vengono passati a main.py, es.:
#   bash sbatch_script.sh --run_severity_grid --run_verbalized --run_quant_comparison
# Per fissare il nodo (i checkpoint a blocchi stanno sul disco del nodo su cui
# gira il job, e faretra/moro232 non condividono la home):
#   SBATCH_NODE=faretra bash sbatch_script.sh ...
sbatch -N 1 ${SBATCH_NODE:+-w "$SBATCH_NODE"} --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh "$@"
