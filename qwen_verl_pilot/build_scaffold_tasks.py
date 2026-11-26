"""Freeze a tiny topic-disjoint agent-writing protocol for the authorized pilot."""
import json
from pathlib import Path
from build_data import TRAIN, VAL

ROOT = Path(__file__).resolve().parent
repo = ROOT/'scaffold_training'
tasks = repo/'private'/'writing_tasks'
template = (repo/'examples/hello_file/task.toml').read_text()
rows = []
for split, items in [('train', TRAIN[:2]), ('dev', [VAL[0], VAL[3]])]:
    for topic, question in items:
        slug = 'write_' + topic
        path = tasks/slug
        (path/'environment').mkdir(parents=True, exist_ok=True)
        (path/'tests').mkdir(exist_ok=True)
        (path/'environment/Dockerfile').write_text('FROM python:3.12-slim\nWORKDIR /app\nCMD ["bash"]\n')
        (path/'task.toml').write_text(template.replace('antislop/hello-file', 'antislop/'+slug))
        (path/'instruction.md').write_text(question+'\n\nUse your tools to write the requested text to /app/answer.txt. Put only the requested prose in the file. Do not add invented facts. Your final chat reply may briefly confirm completion.\n')
        (path/'tests/test.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier
if [[ -f /app/answer.txt ]] && [[ -s /app/answer.txt ]] && [[ $(wc -c < /app/answer.txt) -lt 20000 ]]; then
  cp /app/answer.txt /logs/verifier/answer.txt
  printf '1\\n' > /logs/verifier/reward.txt
else
  printf '0\\n' > /logs/verifier/reward.txt
fi
''')
        rows.append({'task':slug,'slug':slug,'split':split,'dataset':'pilot-authored-writing','question':question})
protocol = {'name':'antislop-hermes-pi-writing-pilot-v1',
            'base_revision':'395c2c057ac4d7d6fa333dd9774a652039ebddb6',
            'source_commit':'959e3566727d2fbb6cbfccc47691c15dcdfe5a2c',
            'tasks':rows, 'train_samples_per_group':4, 'dev_samples_per_group':1,
            'reward':'valid file verifier * Opus quality/4 * (0.7 + 0.3 * not_slop)',
            'sampling':{'temperature':1,'top_p':1,'top_k':-1},
            'scope':'one bounded GRPO update from exact native vLLM agent tokens; pilot only'}
(ROOT/'scaffold_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print('Created', len(rows), 'tasks with disjoint train/dev topics')
