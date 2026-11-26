#!/bin/bash
# Round 2, unattended on the box: setup, render r2 data (no thinking), train from ck120 to checkpoint-$STOP,
# then eval two checkpoints on the old 20 held-out tasks and the 30 hard held-out drafts.
STOP=${STOP:-100}; MID=${MID:-60}; W=/workspace; C=$W/code; L=$W/logs/go.log
exec >> $L 2>&1; echo "go2 start $(date)"
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
cd $C
python3 render.py --model $W/models/humanlike --src data/train_tasks_r2.jsonl --out data/train.jsonl || exit 2
python3 render.py --model $W/models/humanlike --src data/heldout.jsonl --out data/heldout_rendered.jsonl || exit 2
python3 render.py --model $W/models/humanlike --src data/hard_val.jsonl --out data/hard_val_rendered.jsonl || exit 2
ck=$W/runs/pgemu2/checkpoint-$STOP
( until [ -f $ck/adapter_model.safetensors ] && [ -f $ck/trainer_state.json ]; do sleep 20; done
  sleep 30; pkill -f 11_grpo_train.py; sleep 10; pkill -f "trl vllm-serve"; echo "stopped at checkpoint-$STOP $(date)" >> $W/logs/STOPPED ) &
sed -i 's#configs/pgemu1.yaml#configs/pgemu2.yaml#' $W/pod.sh
bash $W/pod.sh train
sleep 20; pkill -f "trl vllm-serve"
for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 3000 ] && break; pkill -9 -f "EngineCore|11_grpo_train"; sleep 10; done
echo "train done $(date); $(ls $W/runs/pgemu2 | grep checkpoint | tr '\n' ' ')"
ev() {  # $1 label $2 lora $3 gpu $4 data
  CUDA_VISIBLE_DEVICES=$3 python3 eval.py --model $W/models/humanlike --lora $2 --data $4 --n 3 --out $W/results/$1.json > $W/logs/eval_$1.log 2>&1; tail -3 $W/logs/eval_$1.log; }
ev r2ck${STOP}_heldout $ck 0 data/heldout_rendered.jsonl &
sleep 90; ev r2ck${MID}_heldout $W/runs/pgemu2/checkpoint-$MID 1 data/heldout_rendered.jsonl &
wait
ev r2ck${STOP}_hard $ck 0 data/hard_val_rendered.jsonl &
sleep 90; ev r2ck${MID}_hard $W/runs/pgemu2/checkpoint-$MID 1 data/hard_val_rendered.jsonl &
wait
ev ck120_hard /workspace/init/checkpoint-120 0 data/hard_val_rendered.jsonl
echo "ALL DONE $(date)"; touch $W/logs/ALL_DONE
