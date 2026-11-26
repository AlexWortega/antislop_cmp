#!/bin/bash
# Runs on the box, unattended: wait for setup, patch trl, render (no thinking), train until checkpoint-$STOP, eval two checkpoints.
STOP=${STOP:-140}; MID=${MID:-90}; W=/workspace; L=$W/logs/go.log
exec >> $L 2>&1
echo "go start $(date)"
until [ -f $W/logs/SETUP_DONE ]; do sleep 30; pgrep -f "pod.sh setup" >/dev/null || [ -f $W/logs/SETUP_DONE ] || { echo "setup died"; exit 1; }; done
f=$(python3 -c "import trl.scripts.vllm_serve as m;print(m.__file__)")
grep -q VLLM_MAX_NUM_SEQS "$f" || python3 - "$f" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
old='''    if script_args.trust_remote_code:
        command += ["--trust-remote-code"]'''
assert old in s
s=s.replace(old, old+'''
    command += ["--max-num-seqs", __import__("os").environ.get("VLLM_MAX_NUM_SEQS", "128")]''',1)
open(p,"w").write(s); print("patched")
PY
bash $W/pod.sh render nothink || exit 2
ck=$W/runs/pgemu1/checkpoint-$STOP
( until [ -f $ck/adapter_model.safetensors ] && [ -f $ck/trainer_state.json ]; do sleep 20; done
  sleep 30; pkill -f 11_grpo_train.py; sleep 10; pkill -f "vllm[ -]serve"; echo "stopped at checkpoint-$STOP $(date)" >> $W/logs/STOPPED ) &
RESUME=1 bash $W/pod.sh train
sleep 20; pkill -f "vllm[ -]serve"; sleep 10
echo "train done $(date); checkpoints: $(ls $W/runs/pgemu1 | grep checkpoint | tr '\n' ' ')"
for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 3000 ] && break; pkill -9 -f "vllm|EngineCore"; sleep 10; done
nvidia-smi --query-gpu=memory.used --format=csv
THINK=0 N=3 GPU=0 bash $W/pod.sh eval ck$STOP $ck &
sleep 90; THINK=0 N=3 GPU=1 bash $W/pod.sh eval ck$MID $W/runs/pgemu1/checkpoint-$MID &
wait
[ -f $W/results/heldout_ck$MID.json ] || THINK=0 N=3 GPU=0 bash $W/pod.sh eval ck$MID $W/runs/pgemu1/checkpoint-$MID
echo "ALL DONE $(date)"; touch $W/logs/ALL_DONE
