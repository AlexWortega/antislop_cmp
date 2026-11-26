#!/usr/bin/env bash
set -euo pipefail
cd /workspace/scaffold_training
export PATH=/workspace/pilot/vendor/verl/.venv/bin:$PATH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HOME=/workspace/hf
python scaffold_training/grpo_update.py \
 --model-path /workspace/antislop_model \
 --rollouts private/scored_train_groups.jsonl \
 --protocol private/dataset_protocol.json \
 --policy-id antislop-395c2c-base \
 --out private/rl-adapter \
 --rank 16 --max-len 16384 --lr 3e-6 --kl-beta 0.01 --clip 0.2 "$@"
