#!/usr/bin/env bash
set -euo pipefail
cd /workspace/scaffold_training
export PATH=/workspace/pilot/vendor/verl/.venv/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ANTISLOP_MODEL_PATH=/workspace/antislop_model
export ANTISLOP_POLICY_ID=antislop-395c2c-base
export ANTISLOP_CAPTURE_LOGPROBS=1
export VLLM_API_KEY="$(cat /workspace/antislop-pilot-serve.key)"
export ANTISLOP_SERVE_KEY="$VLLM_API_KEY"
export ANTISLOP_CONTEXT=65536 ANTISLOP_MAX_SEQS=2 ANTISLOP_GPU_UTIL=0.7
export ANTISLOP_HOST=127.0.0.1 ANTISLOP_PORT=8000
export ANTISLOP_AUDIT_DIR=/workspace/scaffold_training/private/audit
exec bash serve_vllm.sh --enforce-eager --max-num-batched-tokens 4096
