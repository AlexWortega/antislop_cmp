#!/usr/bin/env bash
# astra3 (multi-turn coding RL on Terminal-Lego, see cv/sep/antislop_grpo/ASTRA3_CODE_RL.md):
# bundle the code from branch astra3-code-rl, upload it (a few MB) to the private run repo, check budgets,
# then create the 2x H200 pod that runs pod/bootstrap_astra3.sh unattended.
#
#   bash launch_astra3.sh               # dry run: budget checks + local bundle, prints the pod request;
#                                       # uploads nothing, creates nothing
#   SMOKE=1 LAUNCH=1 bash launch_astra3.sh   # 3-step smoke run + evals (~4 h cap)
#   LAUNCH=1 bash launch_astra3.sh      # full run (STEPS=60 by default, 30 h cap)
set -euo pipefail
cd "$(dirname "$0")"
RUN_REPO=${RUN_REPO:-rawmodels/astra3-code-run}
OR_KEY=$(ssh eva01 "cat ${OR_KEY_FILE:-~/.or_key_tb}" < /dev/null)
HF_TOKEN=$(ssh eva01 'cat ~/.cache/huggingface/token' < /dev/null)
RP_KEY=$(grep -E '^RUNPOD_API_KEY=' ~/Desktop/side_projects/rawmodels/.env | cut -d= -f2- | tr -d '"')

left=$(curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OR_KEY" | python3 -c "import json,sys;d=json.load(sys.stdin)['data'];print(int(d['limit_remaining'] if d.get('limit_remaining') is not None else 10**6))")
bal=$(curl -s https://api.runpod.io/graphql -H "Authorization: Bearer $RP_KEY" -H 'Content-Type: application/json' \
  -d '{"query":"{ myself { clientBalance } }"}' | python3 -c "import json,sys;print(int(json.load(sys.stdin)['data']['myself']['clientBalance']))")
need=$([ "${SMOKE:-0}" = 1 ] && echo 60 || echo 330)
echo "openrouter remaining \$$left, runpod balance \$$bal (need >= \$$need)"
[ "$left" -ge 30 ] || { echo "openrouter budget too low"; exit 1; }
[ "$bal" -ge "$need" ] || { echo "runpod balance too low"; exit 1; }

# code bundle: tracked files of the branch + the two prose source files 10_build_prompts.py reads
( cd cv && git diff --quiet HEAD -- sep/antislop_grpo || { echo "uncommitted changes in sep/antislop_grpo"; exit 1; }
  mkdir -p ../bundle3 && { git ls-files sep/antislop_grpo; ls sep/antislop/data/distill.jsonl sep/antislop/data/human_qa.jsonl; } \
    > ../bundle3/list.txt && tar czf ../bundle3/code.tgz -T ../bundle3/list.txt
  echo "bundle: $(wc -l < ../bundle3/list.txt) files, $(du -h ../bundle3/code.tgz | cut -f1), commit $(git rev-parse --short HEAD)" )

body() {  # body <with_cuda_filter 0|1>
python3 - "$1" <<PY
import json, sys
env = {"HF_TOKEN": "$HF_TOKEN", "OPENROUTER_API_KEY": "$OR_KEY", "RUNPOD_API_KEY": "$RP_KEY", "RUN_REPO": "$RUN_REPO"}
if "${SMOKE:-0}" == "1":
    env["SMOKE"] = "1"
if "${STEPS:-}":
    env["STEPS"] = "${STEPS:-}"
b = {
  "name": "astra3-code-rl" + ("-smoke" if "${SMOKE:-0}" == "1" else ""),
  "imageName": "vllm/vllm-openai:v0.30.0-cu129",
  "gpuTypeIds": ["NVIDIA H200", "NVIDIA H200 NVL", "NVIDIA B200"],
  "gpuTypePriority": "custom",
  "gpuCount": 2,
  "cloudType": "SECURE",
  "containerDiskInGb": 500,
  "volumeInGb": 0,
  "dockerEntrypoint": ["bash", "-c"],
  "dockerStartCmd": ["curl -sfL -H \"Authorization: Bearer \$HF_TOKEN\" https://huggingface.co/$RUN_REPO/resolve/main/pod/bootstrap_astra3.sh -o /bootstrap.sh && bash /bootstrap.sh"],
  "env": env,
}
if sys.argv[1] == "1":
    b["allowedCudaVersions"] = ["12.9", "13.0"]
print(json.dumps(b))
PY
}
if [ "${LAUNCH:-0}" != 1 ]; then
  echo "dry run: would POST this pod (secrets elided); set LAUNCH=1 to create it"
  body 1 | python3 -c "import json,sys;b=json.load(sys.stdin);b['env']={k:'***' for k in b['env']};print(json.dumps(b,indent=1))"
  exit 0
fi
HF_TOKEN=$HF_TOKEN RUN_REPO=$RUN_REPO python3 - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["RUN_REPO"]
api.create_repo(repo, private=True, exist_ok=True)
api.upload_file(path_or_fileobj="bundle3/code.tgz", path_in_repo="bundle/code.tgz", repo_id=repo)
api.upload_file(path_or_fileobj="cv/sep/antislop_grpo/pod/bootstrap_astra3.sh", path_in_repo="pod/bootstrap_astra3.sh", repo_id=repo)
print("uploaded bundle + bootstrap to", repo)
PY
for filt in 1 0; do
  resp=$(body $filt | curl -s -w "\n%{http_code}" -X POST https://rest.runpod.io/v1/pods \
    -H "Authorization: Bearer $RP_KEY" -H 'Content-Type: application/json' -d @-)
  code=$(echo "$resp" | tail -1)
  echo "$resp" | sed '$d' | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('non-json response'); sys.exit()
if isinstance(d, dict) and 'id' in d:
    print('pod', d['id'], d.get('machine',{}).get('gpuDisplayName') or d.get('gpu'), 'costPerHr', d.get('costPerHr'))
else: print(str(d)[:400])"
  [ "$code" = 200 ] || [ "$code" = 201 ] && exit 0
  echo "create failed (HTTP $code, cuda filter=$filt)"
done
exit 1
