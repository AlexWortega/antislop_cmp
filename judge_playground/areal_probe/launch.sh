#!/usr/bin/env bash
# AReaL probe on the GPU box (2 x H200, Ubuntu 24.04, CUDA 13 driver). Not executed yet.
#
#   bash launch.sh install   # fresh box: system packages, uv, AReaL v2.1.1 (vLLM variant), model snapshot
#   bash launch.sh check     # imports + GPU + model config, no training
#   bash launch.sh run       # the 10-step probe
#
# What it expects, as the verl runner (vast_core.py) already sets up:
#   /workspace/lab/areal_probe/   this directory (agent.py, train.py, config.yaml, rows.json from make_rows.py)
#   /workspace/lab/native/serve.key   shared key for the shim (port 9001) and the coordinator
#   127.0.0.1:18975 on this box  -> laptop coordinator (reverse tunnel from the runner)
#   agent host :18974            -> 127.0.0.1:9001 on this box (to_agents.sh); agent.py serves 9001
#   AREAL_ADMIN_KEY              any random string, used only between AReaL processes
#
# Alternative to "install": rent with image ghcr.io/areal-project/areal-runtime:v2.1.1-vllm and set AREAL_PY to its python.
set -euo pipefail

AREAL_REF=${AREAL_REF:-v2.1.1}
AREAL_DIR=${AREAL_DIR:-/workspace/areal}
PROBE_DIR=${PROBE_DIR:-/workspace/lab/areal_probe}
MODEL_DIR=${MODEL_DIR:-/workspace/model}
MODEL_ID=${MODEL_ID:-rawmodels/Qwen3.8-27B-antislop}
MODEL_REV=${MODEL_REV:-395c2c057ac4d7d6fa333dd9774a652039ebddb6}
FA_WHEEL=${FA_WHEEL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl}
export PATH=/root/.local/bin:$PATH UV_CACHE_DIR=/workspace/uvcache HF_HOME=/workspace/hf
AREAL_PY=${AREAL_PY:-$AREAL_DIR/.venv/bin/python}

install() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq python3.12 python3.12-dev python3.12-venv git git-lfs curl rsync build-essential ninja-build libnuma1 libnuma-dev
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/0.11.16/install.sh | sh
  [ -d "$AREAL_DIR/.git" ] || git clone --depth 1 --branch "$AREAL_REF" https://github.com/areal-project/AReaL "$AREAL_DIR"
  cd "$AREAL_DIR"
  # vLLM variant (docs/en/tutorial/installation.md, "Using vLLM Instead of SGLang"): vllm==0.19.1, torch 2.10 (cu129 wheels;
  # they run on a CUDA 13 driver). SGLang 0.5.10.post1 is not used: its LoRA path crashes on in_proj_qkv / in_proj_z
  # (sgl-project/sglang#30168).
  cp pyproject.vllm.toml pyproject.toml
  cp uv.vllm.lock uv.lock
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python "$FA_WHEEL"     # prebuilt flash-attn, skips a ~30 min compile
  uv sync --extra cuda
  uv pip install --python .venv/bin/python fastapi uvicorn httpx   # shim deps for agent.py (normally already present)
  [ -s /root/.hf_token ] && export HF_TOKEN="$(cat /root/.hf_token)"
  .venv/bin/python - "$MODEL_ID" "$MODEL_REV" "$MODEL_DIR" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
repo, rev, out = sys.argv[1:]
for attempt in range(8):
    try:
        snapshot_download(repo, revision=rev, local_dir=out, max_workers=8)
        break
    except Exception as exc:
        if attempt == 7:
            raise
        print(f'download retry {attempt + 1}: {type(exc).__name__}', flush=True)
        time.sleep(30 * (attempt + 1))
PY
  rm -rf "$MODEL_DIR/.cache"
}

check() {
  cd "$PROBE_DIR"
  "$AREAL_PY" - "$MODEL_DIR" <<'PY'
import json, sys, torch, transformers, vllm, peft, areal
cfg = json.load(open(sys.argv[1] + '/config.json'))
text = cfg.get('text_config', {})
print(json.dumps({'areal': areal.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__,
                  'vllm': vllm.__version__, 'peft': peft.__version__, 'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                  'model_type': cfg.get('model_type'), 'arch': cfg.get('architectures'),
                  'layers': text.get('num_hidden_layers'), 'hidden': text.get('hidden_size'), 'vocab': text.get('vocab_size')}))
from areal.engine.core.model import is_qwen3_5_model, is_valid_vision_model
assert is_qwen3_5_model(cfg['model_type']) and is_valid_vision_model(cfg['model_type']), cfg['model_type']
import fla  # flash-linear-attention: GatedDeltaNet kernels for the 48 linear-attention layers in HF training
PY
  [ -s "$PROBE_DIR/rows.json" ] || { echo "rows.json missing: run make_rows.py on the laptop and upload it"; exit 1; }
  [ -s /workspace/lab/native/serve.key ] || { echo "serve.key missing"; exit 1; }
  curl -s -o /dev/null -w 'coordinator tunnel: HTTP %{http_code} (401/405 means it is up)\n' -X POST http://127.0.0.1:18975/run || true
}

run() {
  : "${AREAL_ADMIN_KEY:?set AREAL_ADMIN_KEY to a random string}"
  export AREAL_ADMIN_KEY
  cd "$PROBE_DIR"
  export PYTHONPATH="$PROBE_DIR${PYTHONPATH:+:$PYTHONPATH}" ANTISLOP_NATIVE_ROOT=/workspace/lab/native
  export ANTISLOP_PROBE_ROWS="$PROBE_DIR/rows.json" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  mkdir -p /workspace/areal_runs/name_resolve
  # Run with the venv python, not "uv run" (uv run re-syncs the env).
  exec "$AREAL_PY" "$PROBE_DIR/train.py" --config "$PROBE_DIR/config.yaml" "$@"
}

cmd=${1:-run}; shift || true
case "$cmd" in
  install) install ;;
  check) check ;;
  run) run "$@" ;;
  *) echo "usage: $0 install|check|run [key=value ...]"; exit 2 ;;
esac
