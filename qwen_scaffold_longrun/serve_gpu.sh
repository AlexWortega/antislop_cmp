#!/usr/bin/env bash
set -euo pipefail
cd /workspace/longrun/repo
export PATH=/workspace/longrun/vendor/verl/.venv/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ANTISLOP_MODEL_PATH=/workspace/antislop_model
export ANTISLOP_POLICY_ID="$1"
export ANTISLOP_CAPTURE_LOGPROBS=1 ANTISLOP_MAX_NEW_TOKENS=2048
export VLLM_API_KEY="$(cat /workspace/longrun/serve.key)"
export ANTISLOP_CONTEXT=65536 ANTISLOP_MAX_SEQS=16 ANTISLOP_GPU_UTIL=0.85
export ANTISLOP_HOST=127.0.0.1 ANTISLOP_PORT=8000
export ANTISLOP_AUDIT_DIR=/workspace/longrun/audit
if [[ -n "${2:-}" ]]; then
 export ANTISLOP_ADAPTER_PATH="$2" ANTISLOP_ADAPTER_NAME=antislop-scaffold
fi
echo "$$" > /workspace/longrun/server.pid
exec bash serve_vllm.sh --enforce-eager --max-num-batched-tokens 8192
