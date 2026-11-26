#!/usr/bin/env bash
# Resume native verl from the last full checkpoint (set in resume_launch.py). Refuses to run with an existing rental.
set -euo pipefail
cd "$(dirname "$0")"
PY=../qwen_scaffold_longrun/.venv/bin/python
[ -f .secrets.env ] || { echo 'Missing .secrets.env'; exit 1; }
set -a; . ./.secrets.env; set +a
: "${VAST_API_KEY:?}"; : "${OPENROUTER_API_KEY:?}"
export ANTISLOP_COORDINATOR_PORT=18976
"$PY" -m unittest test_cap_grade test_coordinator test_guard test_hf_ckpt 2>&1 | tail -n 1 | grep -q OK || { echo 'Tests failed'; exit 1; }
# Restart from the Hub: RESUME_FROM_HF=1 RUN_ARCHIVE=runs/run2 PRIOR_GPU_USD=<spent so far> ./start_run2.sh
ARCHIVE="${RUN_ARCHIVE:-runs/run1}"; export PRIOR_GPU_USD="${PRIOR_GPU_USD:-42.02}"
if [ "${RESUME_FROM_HF:-}" = 1 ]; then
 : "${HF_BACKUP_TOKEN:?}"; : "${RUN_ARCHIVE:?set a new archive dir}"; export RESUME_FROM_HF
 [ "$PRIOR_GPU_USD" != 42.02 ] || { echo 'Set PRIOR_GPU_USD to the real spend so the GPU cap stays correct'; exit 1; }
fi
if [ ! -d "$ARCHIVE" ]; then
 mkdir -p "$ARCHIVE"
 for f in output checkpoints instance.json instance_status.json progress.json heartbeat.json STOP failure.json backup_error.json monitor_status.json monitor_events.jsonl monitor.log native_metrics.jsonl native_runner.log tunnel.log vast_controller.log rental_cleanup_verification.json coordinator.log guard_events.jsonl guard_status.json guard.log restart_accounting.bak; do [ -e "$f" ] && mv "$f" "$ARCHIVE"/; done
 mkdir output
fi
[ ! -e instance.json ] || { echo 'instance.json exists: inspect the rental first'; exit 1; }
rm -f STOP GUARD_STOP REQUEST_STOP launch_failure.json
"$PY" - <<'PY'
import json,os;p='restart_accounting.json';d=json.load(open(p));d['prior_gpu_reserved_usd']=float(os.environ['PRIOR_GPU_USD']);json.dump(d,open(p,'w'),indent=2)
PY
OFFER=$("$PY" pick_offer.py)
nohup "$PY" -u coordinator.py > coordinator.log 2>&1 &
sleep 2
nohup "$PY" -u vast_controller.py --offer "$OFFER" > vast_controller.log 2>&1 &
nohup "$PY" -u resume_launch.py > native_runner.log 2>&1 &
RUN=$!
nohup "$PY" -u monitor.py > monitor.log 2>&1 &
nohup "$PY" -u guard.py > guard.log 2>&1 &
nohup caffeinate -ims -w "$RUN" > /dev/null 2>&1 &
echo "Started offer $OFFER; runner pid $RUN"
