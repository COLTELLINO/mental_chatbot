#!/bin/bash

PHYS_DIR="/home/patrignani/mental_chatbot"
LLM_CACHE_DIR="/llms"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":/llms \
    -e HF_HOME="/llms" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e HF_TOKEN="$HF_TOKEN" \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    mental-chatbot-image \
    "/workspace/train.sh" "$@"