#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-dev python3-venv git curl rsync libnuma1 libnuma-dev ninja-build > /workspace/longrun/output/apt.log 2>&1
# PyPI fallback: the astral installer fetches from GitHub releases, which can be down.
(curl -LsSf https://astral.sh/uv/0.11.16/install.sh | sh) || { python3 -m venv /opt/uvpip && /opt/uvpip/bin/pip install -q uv==0.11.16 && mkdir -p /root/.local/bin && ln -sf /opt/uvpip/bin/uv /root/.local/bin/uv; }
export PATH=/root/.local/bin:$PATH UV_CACHE_DIR=/workspace/uvcache HF_HOME=/workspace/hf
cd /workspace/longrun/vendor/verl
uv sync --frozen --all-packages --extra vllm --extra fsdp
uv pip freeze > /workspace/longrun/output/environment.txt
.venv/bin/python - <<'PY'
import json,torch,transformers,vllm,peft
from huggingface_hub import snapshot_download
print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'vllm':vllm.__version__,'peft':peft.__version__,'gpu':torch.cuda.get_device_name(0)}),flush=True)
assert torch.__version__=='2.13.0+cu130' and transformers.__version__=='5.16.1' and vllm.__version__=='0.32.0'
snapshot_download('rawmodels/Qwen3.8-27B-antislop',revision='395c2c057ac4d7d6fa333dd9774a652039ebddb6',local_dir='/workspace/antislop_model')
PY
touch /workspace/longrun/BOOTSTRAP_DONE
