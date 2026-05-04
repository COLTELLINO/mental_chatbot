FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
LABEL maintainer="UniboNLP"

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace
ENV APP_PATH=/workspace

RUN apt-get update -y && \
    apt-get install -y curl git bash nano build-essential \
                       python3.11 python3.11-venv python3.11-dev python3-pip && \
    apt-get autoremove -y && apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip
RUN python3.11 -m pip install wrapt --upgrade --ignore-installed

RUN python3.11 -m pip install torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

COPY build/requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

ENV DEBIAN_FRONTEND=dialog