#!/bin/bash

export WANDB_API_KEY="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"

sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh "Qwen/Qwen3-0.6B" "ShenLab/MentalChat16K"