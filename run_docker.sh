#!/bin/bash

PHYS_DIR="/home/patrignani/mental_chatbot"
LLM_CACHE_DIR="/llms"

# BUG CORRETTO il 01/10/2026: i token venivano passati come -e NOME="$VALORE",
# quindi il valore compariva nella riga di comando di docker run, leggibile da
# tutti gli utenti del nodo con ps per tutta la durata del job. Con -e NOME
# docker prende il valore dall'ambiente di questo script, senza scriverlo
# nella riga di comando. Le variabili vanno quindi esportate (vedi ~/.uq_env).

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":/llms \
    -e HF_HOME="/llms" \
    -e WANDB_API_KEY \
    -e HF_TOKEN \
    -e PYTORCH_JIT=0 \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    mental-chatbot-image \
    "/workspace/train.sh" "$@"