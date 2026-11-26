#!/usr/bin/env bash
set -euo pipefail
cd /workspace/longrun/repo
export PATH=/workspace/longrun/vendor/verl/.venv/bin:$PATH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/hf
step="$1"
policy="$2"
parent="$3"
lr="$4"
args=(--model-path /workspace/antislop_model
 --rollouts "/workspace/longrun/steps/$step/scored/groups.jsonl"
 --protocol /workspace/longrun/protocol.json --policy-id "$policy"
 --out "/workspace/longrun/checkpoints/$step" --rank 16 --alpha 512
 --min-group-size 32 --max-len 16384 --lr "$lr" --kl-beta 0.01 --clip 0.2)
if [[ -n "$parent" ]]; then args+=(--adapter "$parent"); fi
python scaffold_training/grpo_update.py "${args[@]}"
