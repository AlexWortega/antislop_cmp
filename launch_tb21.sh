#!/usr/bin/env bash
# TB 2.1 model server: 2x H200 pod running pod/tb_serve_pod.sh (astra2 on :30001, FP8 + DFlash2, dp=2),
# exposed as raw TCP and gated by a random TB_KEY (saved to tb21_key.txt). Deletes the demo pod first.
# Auto-deletes after MAX_HOURS (5); delete it yourself once harbor is done.
set -euo pipefail
TB_KEY=${TB_KEY:-$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')}
echo "$TB_KEY" > tb21_key.txt; chmod 600 tb21_key.txt
OR_KEY=$(ssh eva01 'cat ~/.or_key' < /dev/null)
HF_TOKEN=$(ssh eva01 'cat ~/.cache/huggingface/token' < /dev/null)
RP_KEY=$(grep -E '^RUNPOD_API_KEY=' ~/Desktop/side_projects/rawmodels/.env | cut -d= -f2- | tr -d '"')

left=$(curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OR_KEY" | python3 -c "import json,sys;print(int(json.load(sys.stdin)['data']['limit_remaining']))")
bal=$(curl -s https://api.runpod.io/graphql -H "Authorization: Bearer $RP_KEY" -H 'Content-Type: application/json' \
  -d '{"query":"{ myself { clientBalance } }"}' | python3 -c "import json,sys;print(int(json.load(sys.stdin)['data']['myself']['clientBalance']))")
echo "openrouter remaining \$$left, runpod balance \$$bal"
[ "$left" -ge 20 ] || { echo "openrouter budget too low"; exit 1; }
[ "$bal" -ge 15 ] || { echo "runpod balance too low"; exit 1; }
curl -s -o /dev/null -X DELETE https://rest.runpod.io/v1/pods/drlee90wifixod -H "Authorization: Bearer $RP_KEY" && echo "demo pod deleted"
curl -sf -o /dev/null -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/rawmodels/antislop-think-opus-run/resolve/main/pod/tb_serve_pod.sh || { echo "bundle missing"; exit 1; }

body() {  # body <with_cuda_filter 0|1>
python3 - "$1" <<PY
import json, sys
b = {
  "name": "tb21-serve",
  "imageName": "lmsysorg/sglang:v0.5.20",
  "gpuTypeIds": ["NVIDIA H200", "NVIDIA H200 NVL", "NVIDIA B200"],
  "gpuTypePriority": "custom",
  "gpuCount": int("${GPUS:-2}"),
  "cloudType": "SECURE",
  "containerDiskInGb": 400,
  "volumeInGb": 0,
  "dockerEntrypoint": ["bash", "-c"],
  "dockerStartCmd": ["curl -sfL -H \"Authorization: Bearer \$HF_TOKEN\" https://huggingface.co/rawmodels/antislop-think-opus-run/resolve/main/pod/tb_serve_pod.sh -o /tb_serve_pod.sh && bash /tb_serve_pod.sh"],
  "env": {"HF_TOKEN": "$HF_TOKEN", "OPENROUTER_API_KEY": "$OR_KEY", "RUNPOD_API_KEY": "$RP_KEY", "TB_KEY": "$TB_KEY", "DP": "${GPUS:-2}", "MODEL_REPO": "${MODEL_REPO:-rawmodels/Qwen3.8-27B-antislop-think-opus-grpo}", "NAME": "${NAME:-astra2}"},
  "ports": ["30001/tcp"],
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
    print('pod', d['id'], 'costPerHr', d.get('costPerHr')); open('tb21_pod.txt','w').write(d['id'])
else: print(str(d)[:400])"
  [ "$code" = 200 ] || [ "$code" = 201 ] && exit 0
  echo "create failed (HTTP $code, cuda filter=$filt)"
done
exit 1
