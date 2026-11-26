#!/bin/sh
# After 10 h: destroy only a RUNNING instance, after a final sync. A stopped one is left alone (its disk is the only copy).
cd "$(dirname "$0")"; sleep 36000
python3 vast.py mine | grep -q '"actual_status": "running"' && ./sync.sh && python3 vast.py destroy 60967816
