#!/bin/bash
N=/root/antislop_cmp/qwen_verl_native; cd $N
echo "time: $(date -u +%H:%M) UTC"
echo "phase: $(python3 -c "import json;print(json.load(open(\"progress.json\"))[\"phase\"])" 2>/dev/null || (test -f SETUP_DONE && echo setup_done || echo setting_up))"
ssh -o ConnectTimeout=15 -o BatchMode=yes -i /root/.ssh/antislop_gpu -p 2222 root@127.0.0.1 "grep -o \"step:[0-9]* \" /workspace/native/output/continuation.log 2>/dev/null | tail -1 | sed s/^/last_/; cat /workspace/native/output/checkpoints/latest_checkpointed_iteration.txt 2>/dev/null | sed s/^/checkpoint_step:/" 2>/dev/null || echo "gpu: unreachable"
echo "agents running: $(docker ps -q | wc -l)"
echo "infra failures: $(cat infra_failures.jsonl 2>/dev/null | wc -l || echo 0), skipped groups: $(cat skipped_groups.jsonl 2>/dev/null | wc -l || echo 0)"
test -f CIRCUIT_OPEN && echo "CIRCUIT OPEN"
test -f failure.json && echo "failure: $(head -c 200 failure.json)"
test -f STOP && echo "STOP: run finished, machines shutting down"
