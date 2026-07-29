#!/bin/bash

export WANDB_API_KEY="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"

# HF_TOKEN must be exported in your shell too (needed for gated models:
# google/medgemma-4b-it, google/gemma-3-4b-it). Accept each model's license
# on huggingface.co first, then export HF_TOKEN="hf_..." before running this.

sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh