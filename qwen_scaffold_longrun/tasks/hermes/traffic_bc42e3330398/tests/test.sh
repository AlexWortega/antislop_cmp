#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier
if [[ -s /app/answer.txt ]] && [[ $(wc -c < /app/answer.txt) -lt 20000 ]]; then
 cp /app/answer.txt /logs/verifier/answer.txt
 printf '1\n' > /logs/verifier/reward.txt
else
 printf '0\n' > /logs/verifier/reward.txt
fi
