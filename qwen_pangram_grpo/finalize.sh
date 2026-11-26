#!/bin/sh
# Wait for go2.sh to finish, pull everything, check the copy, then destroy the box and stop the helpers.
cd "$(dirname "$0")"
until ./box 'test -f /workspace/logs/ALL_DONE' >/dev/null 2>&1; do
  st=$(python3 vast.py mine 2>/dev/null | grep -o '"actual_status": "[a-z]*"')
  case "$st" in *running*|"") ;; *) echo "box status $st; leaving it alone"; exit 1;; esac
  sleep 300
done
./sync.sh; ./sync.sh
ls pulled/logs/ALL_DONE pulled/runs/pgemu2/checkpoint-100/adapter_model.safetensors pulled/runs/pgemu2/checkpoint-60/adapter_model.safetensors || { echo "local copy incomplete; not destroying"; exit 2; }
ls pulled/results/
python3 vast.py destroy 60967816 && echo DESTROYED
pkill -f '[w]atchdog.sh'; pkill -f 'while.*[s]ync.sh'
python3 vast.py mine
