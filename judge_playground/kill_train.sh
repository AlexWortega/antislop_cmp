#!/bin/bash
# Pause a Vast run without losing the machines: kills the newest verl train job on the GPU box and records its exit
# code, so the runner sees the job end and pauses. Usage: kill_train.sh <run_dir>
set -euo pipefail
run="$1"
read -r host port < <(python3 -c "import json,sys;i=json.load(open(sys.argv[1]+'/state.json'))['instance'];print(i['ssh_host'],i['ssh_port'])" "$run")
key=$(python3 -c "import json,os;print(os.path.expanduser(json.load(open('$(dirname "$0")/judge_config.json'))['vast']['local']['ssh_key']))")
ssh -i "$key" -o BatchMode=yes -o UserKnownHostsFile="$run/known_hosts" -o ConnectTimeout=20 -p "$port" "root@$host" \
  'cd /workspace/lab; f=$(ls -t jobs/train-*.pid | head -1); n=$(basename $f .pid); kill -- -$(cat $f) 2>/dev/null; sleep 3; vendor/verl/.venv/bin/ray stop --force >/dev/null 2>&1; pkill -9 -f "[V]LLM::" || true; [ -s output/$n.exit ] || echo 137 > output/$n.exit; echo killed $n' 2>&1 | tail -1
