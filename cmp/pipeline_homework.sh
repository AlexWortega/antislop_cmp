#!/bin/sh
# Full homework run: judge calibration on student essays, generation on GPU 3, judging, reports.
set -eu
cd /mnt/storage/antislop_cmp/code
export EXP=homework TASKS=tasks_homework
M=/mnt/storage/antislop_cmp/models
SERVER=$HOME/build-qwen35-cuda/bin/llama-server

serve() {  # serve MODEL PORT [extra args]; leaves PID in $SRV
  model=$1; port=$2; shift 2
  CUDA_VISIBLE_DEVICES=3 $SERVER -m "$model" -c 32768 -np 4 -ngl 99 --host 127.0.0.1 --port "$port" "$@" \
    > "out/$EXP/srv_$port.log" 2>&1 &
  SRV=$!
  until curl -sf "127.0.0.1:$port/health" > /dev/null; do sleep 5; done
}

mkdir -p out/$EXP
echo "== calib"; python3 calib.py human; python3 calib.py ai
python3 calib.py judge & CALIB=$!
echo "== drafts"; python3 run.py drafts
echo "== humanizer"; serve $M/humanizer-12b-Q8_0.gguf 8091
python3 run.py gen-humanizer http://127.0.0.1:8091; kill $SRV; wait $SRV || true
echo "== qwen"; serve $M/qwen-Q6_K.gguf 8092 --jinja
python3 run.py gen-qwen http://127.0.0.1:8092; kill $SRV; wait $SRV || true
echo "== judge"; python3 run.py judge
wait $CALIB
python3 calib.py report
python3 run.py report
echo PIPELINE_DONE
