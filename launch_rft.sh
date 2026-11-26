#!/usr/bin/env bash
# RFT stage: 1x H200 pod running pod/rft_pod.sh (LoRA SFT of astra2 on stock-Qwen TB turns + prose replay,
# merge, prose held-out, push). Auto-deletes when done or after MAX_HOURS (5).
set -euo pipefail
OR_KEY=$(ssh eva01 'cat ~/.or_key' < /dev/null)
HF_TOKEN=$(ssh eva01 'cat ~/.cache/huggingface/token' < /dev/null)
RP_KEY=$(grep -E '^RUNPOD_API_KEY=' ~/Desktop/side_projects/rawmodels/.env | cut -d= -f2- | tr -d '"')

left=$(curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OR_KEY" | python3 -c "import json,sys;print(int(json.load(sys.stdin)['data']['limit_remaining']))")
bal=$(curl -s https://api.runpod.io/graphql -H "Authorization: Bearer $RP_KEY" -H 'Content-Type: application/json' \
  -d '{"query":"{ myself { clientBalance } }"}' | python3 -c "import json,sys;print(int(json.load(sys.stdin)['data']['myself']['clientBalance']))")
echo "openrouter remaining \$$left, runpod balance \$$bal"
[ "$left" -ge 20 ] || { echo "openrouter budget too low"; exit 1; }
[ "$bal" -ge 25 ] || { echo "runpod balance too low"; exit 1; }
curl -sf -o /dev/null -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/rawmodels/antislop-think-opus-run/resolve/main/pod/rft_pod.sh || { echo "bundle missing"; exit 1; }

body() {  # body <with_cuda_filter 0|1>
python3 - "$1" <<PY
import json, sys
b = {
  "name": "rft1-train",
  "imageName": "vllm/vllm-openai:v0.30.0-cu129",
  "gpuTypeIds": ["NVIDIA H200", "NVIDIA H200 NVL", "NVIDIA B200"],
  "gpuTypePriority": "custom",
  "gpuCount": 1,
  "cloudType": "SECURE",
  "containerDiskInGb": 400,
  "volumeInGb": 0,
  "dockerEntrypoint": ["bash", "-c"],
  "dockerStartCmd": ["curl -sfL -H \"Authorization: Bearer \$HF_TOKEN\" https://huggingface.co/rawmodels/antislop-think-opus-run/resolve/main/pod/rft_pod.sh -o /rft_pod.sh && bash /rft_pod.sh"],
  "env": {"HF_TOKEN": "$HF_TOKEN", "OPENROUTER_API_KEY": "$OR_KEY", "RUNPOD_API_KEY": "$RP_KEY"},
}
if sys.argv[1] == "1":
    b["allowedCudaVersions"] = ["12.9", "13.0"]
print(json.dumps(b))
PY
}
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
