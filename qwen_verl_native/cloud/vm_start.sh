#!/usr/bin/env bash
# Entry point on the agent host. The laptop starts this once and may then disconnect.
set -uo pipefail
N=/root/antislop_cmp/qwen_verl_native; PY=/root/antislop_cmp/qwen_scaffold_longrun/.venv/bin/python
cd $N
set -a; . $N/.secrets.env; . $N/cloud/run.env; set +a
# The guard needs only the standard library, so it runs before the long environment setup.
nohup python3 -u cloud/agent_guard.py > cloud/agent_guard.out 2>&1 < /dev/null &
if ! bash cloud/agent_host_setup.sh > cloud/setup.log 2>&1; then
  echo '{"error": "agent host setup failed, see cloud/setup.log"}' > launch_failure.json; touch STOP; exit 1
fi
exec bash cloud/vm_services.sh
