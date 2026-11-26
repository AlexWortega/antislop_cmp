#!/bin/sh
# Wait until all 7 eval results exist on the box, pull them, verify locally, then destroy the eval box.
cd "$(dirname "$0")"; ID=61045902
R="r2ck70_hard r2ck80_hard r2ck90_hard ck120_hard r2ck70_heldout r2ck80_heldout r2ck90_heldout"
T=0
until ./box "cd /workspace/results && ls $(for r in $R; do printf '%s.json ' $r; done)" >/dev/null 2>&1; do
  st=$(python3 vast.py mine 2>/dev/null | grep -o '"actual_status": "[a-z]*"')
  case "$st" in *running*|"") ;; *) echo "box status $st; leaving it alone"; exit 1;; esac
  T=$((T+1)); [ $T -gt 40 ] && { echo "timeout 3h20m; box left running"; exit 3; }
  sleep 300
done
sleep 60; ./sync.sh; ./sync.sh
for r in $R; do [ -s pulled/results/$r.json ] || { echo "missing $r locally; not destroying"; exit 2; }; done
ls -la pulled/results/
python3 vast.py destroy $ID && echo DESTROYED
