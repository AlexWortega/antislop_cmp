#!/usr/bin/env bash
# Build steps on both rentals fetch GitHub release assets. Wait until they download, then launch.
cd "$(dirname "$0")/.."
URLS="https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-unknown-linux-gnu.tar.gz https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64"
end=$(( $(date +%s) + 12*3600 ))
while [ $(date +%s) -lt $end ]; do
  ok=1
  for u in $URLS; do
    c=$(curl -s -m 30 -L -r 0-1023 -o /dev/null -w "%{http_code}" "$u"); case $c in 200|206) ;; *) ok=0;; esac
  done
  if [ $ok = 1 ]; then
    echo "$(date -u +%H:%M) github releases ok, launching"
    exec ../qwen_scaffold_longrun/.venv/bin/python -u cloud/cloud_launch.py --gpu-hours 20 --host-extra-hours 2 --steps 50
  fi
  echo "$(date -u +%H:%M) github releases still failing"
  sleep 120
done
echo "gave up waiting for github"; exit 1
