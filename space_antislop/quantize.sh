#!/bin/sh
set -eu
TASK_ROOT=/home/alexw/antislop-space-20261012
cd "$TASK_ROOT"
export CUDA_VISIBLE_DEVICES=0
while ! tail -c 700 convert-text.log | rg -q 'Model successfully exported'; do
    if ! ps -p 406022 >/dev/null; then echo 'CONVERSION FAILED'; exit 1; fi
    sleep 10
done
echo 'START IMATRIX'
timeout --kill-after=15 900 /home/alexw/llama.cpp/build/bin/llama-imatrix \
    -m "$TASK_ROOT/Qwen3.8-27B-antislop-rft1-BF16.gguf" -f imatrix.txt \
    -o imatrix.gguf -ngl 42 -c 256 -b 256 -ub 128 -t 8 --chunks -1 --no-ppl \
    > imatrix.log 2>&1
test -s imatrix.gguf
echo 'START QUANTIZE'
timeout --kill-after=15 1800 "$TASK_ROOT/llama.cpp/build-cpu/bin/llama-quantize" \
    --imatrix imatrix.gguf "$TASK_ROOT/Qwen3.8-27B-antislop-rft1-BF16.gguf" \
    "$TASK_ROOT/Qwen3.8-27B-antislop-rft1-Q4_K_M.gguf" Q4_K_M 8 \
    > quantize.log 2>&1
echo 'QUANTIZE DONE'
sha256sum Qwen3.8-27B-antislop-rft1-Q4_K_M.gguf > q4.sha256
timeout --kill-after=15 1800 "$TASK_ROOT/venv/bin/python" -u calibrate.py \
    --model "$TASK_ROOT/Qwen3.8-27B-antislop-rft1-Q4_K_M.gguf" \
    --label q4 --split train > train.log 2>&1
echo 'TRAIN GENERATIONS DONE'
