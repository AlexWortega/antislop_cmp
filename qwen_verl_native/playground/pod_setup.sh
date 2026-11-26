#!/usr/bin/env bash
# Runs on the playground GPU (nvidia/cuda image): install vLLM 0.32.0 and fetch the base model, then mark it ready.
set -euo pipefail
trap 'echo SETUP_FAILED' ERR
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3 python3-venv python3-dev curl rsync > /workspace/apt.log 2>&1
(curl -LsSf https://astral.sh/uv/0.11.16/install.sh | sh) || { python3 -m venv /opt/uvpip && /opt/uvpip/bin/pip install -q uv==0.11.16 && mkdir -p /root/.local/bin && ln -sf /opt/uvpip/bin/uv /root/.local/bin/uv; }
export PATH=/root/.local/bin:$PATH
uv venv -q --python 3.12 /opt/venv
uv pip install -q --python /opt/venv/bin/python vllm==0.32.0 huggingface_hub ninja
/opt/venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('rawmodels/Qwen3.8-27B-antislop',revision='395c2c057ac4d7d6fa333dd9774a652039ebddb6',local_dir='/workspace/base')"
touch /workspace/base/READY
