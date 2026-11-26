#!/bin/sh
cd "$(dirname "$0")"; ID=61075806; T=0
until ./box 'ls /workspace/results/r2ck80_long.json /workspace/results/ck120_long.json' >/dev/null 2>&1; do
  ./box 'test -f /workspace/logs/ALL_DONE' >/dev/null 2>&1 && { echo "ALL_DONE but results missing"; ./sync.sh; exit 4; }
  st=$(python3 vast.py mine 2>/dev/null | grep -o '"actual_status": "[a-z]*"')
  case "$st" in *running*|"") ;; *) echo "box status $st"; exit 1;; esac
  T=$((T+1)); [ $T -gt 30 ] && { echo timeout; exit 3; }; sleep 120
done
sleep 20; ./sync.sh; ./sync.sh
for r in r2ck80_long ck120_long; do [ -s pulled/results/$r.json ] || { echo "missing $r locally"; exit 2; }; done
python3 vast.py destroy $ID && echo DESTROYED
