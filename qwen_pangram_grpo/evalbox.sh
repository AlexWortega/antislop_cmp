#!/bin/bash
# On the eval box: download the base model, render eval data, eval adapters two at a time (one per GPU).
W=/workspace; C=$W/code; L=$W/logs/go.log; exec >> $L 2>&1; echo "evalbox start $(date)"
set -a; . $W/secrets.env; set +a
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1 EMU_RUN_DIR=$W/runs
[ -n "$OPENROUTER_API_KEY" ] || { echo "no OR key"; exit 9; }
pip install -q hf_transfer 2>&1 | tail -1
python3 - <<'PY' || exit 30
from huggingface_hub import snapshot_download
snapshot_download('LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-2.0', revision='0a4d369895bf0a8b217fb08ef738769bac9d881b',
                  local_dir='/workspace/models/humanlike', max_workers=16)
PY
cd $C
python3 render.py --model $W/models/humanlike --src data/heldout.jsonl --out data/heldout_rendered.jsonl || exit 2
python3 render.py --model $W/models/humanlike --src data/hard_val.jsonl --out data/hard_val_rendered.jsonl || exit 2
ev() {  # $1 label $2 adapter $3 gpu $4 data
  [ -f $W/results/$1.json ] && return
  CUDA_VISIBLE_DEVICES=$3 python3 eval.py --model $W/models/humanlike --lora $W/adapters/$2 --data data/$4 --n 3 --out $W/results/$1.json > $W/logs/eval_$1.log 2>&1
  echo "$1 rc=$? $(date)"; tail -14 $W/logs/eval_$1.log | grep -E 'pass_rate|human_rate|task_mean|score_mean|unscored|Error'; }
ev r2ck80_hard ck80 0 hard_val_rendered.jsonl & sleep 60; ev r2ck70_hard ck70 1 hard_val_rendered.jsonl & wait
ev r2ck90_hard ck90 0 hard_val_rendered.jsonl & sleep 60; ev ck120_hard ck120 1 hard_val_rendered.jsonl & wait
ev r2ck80_heldout ck80 0 heldout_rendered.jsonl & sleep 60; ev r2ck70_heldout ck70 1 heldout_rendered.jsonl & wait
ev r2ck90_heldout ck90 0 heldout_rendered.jsonl & wait
ls $W/results; echo "ALL DONE $(date)"; touch $W/logs/ALL_DONE
