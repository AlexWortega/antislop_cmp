#!/usr/bin/env bash
set -euo pipefail
cd /home/alexw/hemmingway-rft1-20261012
export CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 HMW_FP8_SMALL_PROJECTION_BF16=1
/home/alexw/venvs/vllm/bin/python -u compare_gpu_fixed2.py generate \
    --work /home/alexw/hemmingway-rft1-20261012 --model hemmingway \
    > hemmingway.fixed2.log 2>&1
for model in hemmingway rft1; do
    /home/alexw/venvs/vllm/bin/python -u compare_gpu_fixed2.py generate \
        --work /home/alexw/hemmingway-rft1-20261012 --model "$model" --thinking \
        > "$model.thinking.fixed2.log" 2>&1
done
