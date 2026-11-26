#!/bin/bash
# Long-text check: ck80 vs ck120 on 20 drafts of 500-900 words, one adapter per GPU. Generations are saved before scoring.
W=/workspace; C=$W/code; exec >> $W/logs/go.log 2>&1; echo "evallong start $(date)"
set -a; . $W/secrets.env; set +a
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1 EMU_RUN_DIR=$W/runs
[ -n "$OPENROUTER_API_KEY" ] || { echo "no OR key"; exit 9; }
for a in ck80 ck120; do [ -s $W/adapters/$a/adapter_model.safetensors ] || { echo "adapter $a missing"; exit 8; }; done
pip install -q hf_transfer 2>&1 | tail -1
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-2.0', revision='0a4d369895bf0a8b217fb08ef738769bac9d881b', local_dir='/workspace/models/humanlike', max_workers=16)" || exit 30
cd $C; python3 render.py --model $W/models/humanlike --src data/long_val.jsonl --out data/long_val_rendered.jsonl || exit 2
ev() { CUDA_VISIBLE_DEVICES=$3 python3 eval.py --model $W/models/humanlike --lora $W/adapters/$2 --data data/long_val_rendered.jsonl --n 2 --max-new 4096 --out $W/results/$1.json > $W/logs/eval_$1.log 2>&1; echo "$1 rc=$? $(date)"; }
ev r2ck80_long ck80 0 & sleep 60; ev ck120_long ck120 1 & wait
ls $W/results; echo "ALL DONE $(date)"; touch $W/logs/ALL_DONE
