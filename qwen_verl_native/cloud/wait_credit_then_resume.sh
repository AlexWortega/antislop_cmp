#!/usr/bin/env bash
# Wait until the Vast account has credit again, then restart the stopped rentals.
cd "$(dirname "$0")/.."
set -a; . ./.secrets.env; set +a
for i in $(seq 360); do
  c=$(curl -s -m 30 -H "Authorization: Bearer $VAST_API_KEY" https://console.vast.ai/api/v0/users/current/ | python3 -c "import json,sys;print(json.load(sys.stdin).get('credit',0))" 2>/dev/null || echo 0)
  if python3 -c "import sys;sys.exit(0 if float('$c')>=5 else 1)"; then
    echo "$(date -u +%H:%M) credit $c, resuming"
    exec ../qwen_scaffold_longrun/.venv/bin/python -u cloud/resume_stopped.py
  fi
  echo "$(date -u +%H:%M) credit $c"; sleep 120
done
echo "gave up waiting for credit"; exit 1
