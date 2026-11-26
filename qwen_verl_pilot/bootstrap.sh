#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git curl rsync libnuma1 libnuma-dev ninja-build > /workspace/apt.log 2>&1
curl -LsSf https://astral.sh/uv/0.11.16/install.sh | sh
export PATH=/root/.local/bin:$PATH UV_CACHE_DIR=/workspace/uvcache HF_HOME=/workspace/hf
cd /workspace/pilot/vendor/verl
uv sync --frozen --all-packages --extra vllm --extra fsdp
.venv/bin/python - <<'PY'
import json,torch,transformers,vllm,peft
from huggingface_hub import snapshot_download
print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'vllm':vllm.__version__,'peft':peft.__version__,'gpu':torch.cuda.get_device_name(0)}),flush=True)
snapshot_download('Qwen/Qwen3.8-27B',revision=open('/workspace/pilot/model_revision.txt').read().strip(),local_dir='/workspace/model')
PY
mkdir -p /workspace/pilot/output
.venv/bin/python -m pip freeze > /workspace/pilot/output/environment.txt || uv pip freeze > /workspace/pilot/output/environment.txt
bash /workspace/pilot/run_train.sh 2>&1 | tee /workspace/pilot/output/train.log
