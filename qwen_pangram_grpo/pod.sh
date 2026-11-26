#!/usr/bin/env bash
# Runs ON the Vast box (2 GPUs). Stages are idempotent; logs in /workspace/logs.
#   bash pod.sh setup | probe [think|nothink] | train | eval <label> [lora_dir]
set -uo pipefail
W=/workspace; C=$W/code
export HF_HOME=$W/hf TMPDIR=$W/tmp PIP_CACHE_DIR=$W/tmp/pip HF_HUB_ENABLE_HF_TRANSFER=1 EMU_RUN_DIR=$W/runs
mkdir -p $W/{logs,models,runs,results,tmp}
set -a; . $W/secrets.env; set +a
PY=python3
cd $C
case "$1" in
setup)
  TORCH_REQ=$($PY -c "import importlib.metadata as m, re
print(next(re.search(r'==([0-9.]+)', r).group(1) for r in m.requires('vllm') if re.match(r'torch\s*==', r)))")
  echo "torch $($PY -c 'import torch;print(torch.__version__)') vllm wants $TORCH_REQ"
  $PY -m pip freeze | grep -iE "^(torch|torchvision|vllm|triton|xformers|flashinfer[^=]*|transformers)==" > $W/tmp/constraints.txt
  $PY -m pip install -q -c $W/tmp/constraints.txt "trl==1.13.0" peft accelerate datasets httpx pyyaml safetensors hf_transfer flash-linear-attention 2>&1 | tail -3
  timeout 1200 $PY -m pip install -q -c $W/tmp/constraints.txt --no-build-isolation causal-conv1d 2>&1 | tail -2 || echo "causal-conv1d not installed"
  $PY -c "import torch, vllm, trl, peft, transformers; print('torch', torch.__version__, 'gpus', torch.cuda.device_count(), 'vllm', vllm.__version__, 'trl', trl.__version__, 'tf', transformers.__version__)" || exit 21
  $PY - <<'PY' || exit 30
from huggingface_hub import snapshot_download
print(snapshot_download('LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-2.0', revision='0a4d369895bf0a8b217fb08ef738769bac9d881b',
                        local_dir='/workspace/models/humanlike', max_workers=8))
PY
  du -sh $W/models/humanlike; touch $W/logs/SETUP_DONE ;;
render)
  T=${2:-think}; F=$([ "$T" = think ] && echo --thinking)
  $PY render.py --model $W/models/humanlike --src data/train_tasks.jsonl --out data/train.jsonl $F
  $PY render.py --model $W/models/humanlike --src data/heldout.jsonl --out data/heldout_rendered.jsonl $F ;;
probe)
  T=${2:-think}; F=$([ "$T" = think ] && echo --thinking)
  bash /workspace/pod.sh render $T
  CUDA_VISIBLE_DEVICES=0 $PY probe.py --model $W/models/humanlike --data data/train.jsonl $F --out $W/results/probe_$T.json ;;
serve)
  pkill -f "vllm[ -]serve" 2>/dev/null; sleep 5
  CUDA_VISIBLE_DEVICES=1 nohup trl vllm-serve --model $W/models/humanlike --port 8000 --gpu-memory-utilization 0.85 \
    --max-model-len 6144 --dtype bfloat16 --trust-remote-code ${2:-} > $W/logs/vllm_rollout.log 2>&1 &
  for _ in $(seq 180); do curl -sf 127.0.0.1:8000/health >/dev/null && echo served && exit 0; sleep 10; done
  exit 1 ;;
train)
  bash /workspace/pod.sh serve || bash /workspace/pod.sh serve --enforce-eager || exit 50
  CUDA_VISIBLE_DEVICES=0 RESUME=${RESUME:-0} $PY -u 11_grpo_train.py --config configs/pgemu1.yaml "${@:2}" >> $W/logs/train.log 2>&1
  echo "train rc=$?" | tee -a $W/logs/train.log
  pkill -f "vllm[ -]serve" ;;
eval)
  L=$2; LORA=${3:-}; F=$([ "${THINK:-1}" = 1 ] && echo --thinking)
  CUDA_VISIBLE_DEVICES=${GPU:-0} $PY eval.py --model $W/models/humanlike ${LORA:+--lora $LORA} --data data/heldout_rendered.jsonl \
    --n ${N:-3} $F --out $W/results/heldout_$L.json > $W/logs/eval_$L.log 2>&1; tail -14 $W/logs/eval_$L.log ;;
esac
