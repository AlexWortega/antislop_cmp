#!/usr/bin/env bash
# Start the run services on the agent host (after setup). vm_start.sh calls this; it can be rerun by hand.
set -uo pipefail
N=/root/antislop_cmp/qwen_verl_native; PY=/root/antislop_cmp/qwen_scaffold_longrun/.venv/bin/python
cd $N
set -a; . $N/.secrets.env; . $N/cloud/run.env; set +a
nohup $PY -u coordinator.py >> coordinator.log 2>&1 < /dev/null &
sleep 3
nohup $PY -u guard.py >> guard.log 2>&1 < /dev/null &
nohup $PY -u monitor.py >> monitor.log 2>&1 < /dev/null &
exec $PY -u cloud/vm_launch.py >> native_runner.log 2>&1 < /dev/null
