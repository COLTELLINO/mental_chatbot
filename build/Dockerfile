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

# System python3.11 has ensurepip disabled by Debian/Ubuntu policy, so we
# bootstrap pip via a venv instead (same pattern as the guide's RTX 5090 Dockerfile).
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed

RUN pip install torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

COPY build/requirements.txt build/requirements-nodeps.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# unsloth/weave have huge unpinned dependency trees; installing them together
# with the pinned core stack makes pip backtrack for minutes or fail outright.
# --no-deps keeps them from re-triggering a global resolve.
RUN pip install --no-cache-dir --no-deps -r requirements-nodeps.txt

ENV DEBIAN_FRONTEND=dialog