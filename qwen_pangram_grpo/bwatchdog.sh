#!/bin/sh
# Builder: after 8 h destroy only if RUNNING (never a stopped one).
cd "$(dirname "$0")"; sleep 28800
python3 vast.py mine | grep '"id": 60953329' | grep -q '"actual_status": "running"' && python3 vast.py destroy 60953329
