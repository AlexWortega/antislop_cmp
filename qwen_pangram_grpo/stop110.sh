#!/bin/bash
# Replaces go.sh's tail: stop at checkpoint-110 (OpenRouter budget), then eval ck110 and ck80.
W=/workspace; exec >> $W/logs/go.log 2>&1
ck=$W/runs/pgemu1/checkpoint-110
until [ -f $ck/adapter_model.safetensors ] && [ -f $ck/trainer_state.json ]; do sleep 20; done
sleep 30; pkill -f go.sh; pkill -f "pod.sh train"; pkill -f 11_grpo_train.py; sleep 15; pkill -f "vllm[ -]serve"
echo "stopped at checkpoint-110 $(date)" >> $W/logs/STOPPED
for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 3000 ] && break; pkill -9 -f "vllm|EngineCore|11_grpo"; sleep 10; done
nvidia-smi --query-gpu=memory.used --format=csv
THINK=0 N=3 GPU=0 bash $W/pod.sh eval ck110 $ck &
sleep 90; THINK=0 N=3 GPU=1 bash $W/pod.sh eval ck80 $W/runs/pgemu1/checkpoint-80 &
wait
[ -f $W/results/heldout_ck80.json ] || THINK=0 N=3 GPU=0 bash $W/pod.sh eval ck80 $W/runs/pgemu1/checkpoint-80
echo "ALL DONE $(date)"; touch $W/logs/ALL_DONE
