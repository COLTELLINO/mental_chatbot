#!/bin/bash
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh "Qwen/Qwen3-0.6B" "ShenLab/MentalChat16K"