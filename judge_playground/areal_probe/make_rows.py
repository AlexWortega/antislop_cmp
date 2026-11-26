"""Write areal_probe/rows.json on the laptop: the same rows vast_core.data_rows makes for verl, sized for the probe.

python3 judge_playground/areal_probe/make_rows.py [steps] [prompts_per_step]   (defaults: 10 steps x 4 prompts)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import vast_core  # noqa: E402  (read-only use of data_rows and the default config)

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
per_step = int(sys.argv[2]) if len(sys.argv) > 2 else 4
train = dict(vast_core.DEFAULT_VAST['train'], steps=steps, prompts_per_step=per_step, eval_tasks=4)
root = HERE.parent.parent
tasks = [json.loads(line) for line in (root / 'tasks_v2/rows.jsonl').read_text().splitlines() if line.strip()]
train_rows, val_rows = vast_core.data_rows(tasks, train)
(HERE / 'rows.json').write_text(json.dumps({'train': train_rows, 'val': val_rows}, ensure_ascii=False, indent=1))
print({'train_rows': len(train_rows), 'val_rows': len(val_rows), 'first_group': train_rows[0]['extra_info']['group']})
