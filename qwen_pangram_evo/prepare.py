"""Fixed English task splits: 25 train tasks for evolution (5 per kind), all 20 held-out tasks for the final check."""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (ROOT.parent / 'tasks_v2/rows.jsonl').read_text().splitlines() if l.strip()]
en = [r for r in rows if r['lang'] == 'en']
by_kind = defaultdict(list)
for r in en:
    if r['split'] == 'train':
        by_kind[r['kind']].append(r)
rng = random.Random(20261114)
train = []
for kind in sorted(by_kind):
    train += rng.sample(by_kind[kind], 5)
heldout = [r for r in en if r['split'] == 'heldout']
(ROOT / 'data').mkdir(exist_ok=True)
for name, part in [('train.jsonl', train), ('heldout.jsonl', heldout)]:
    (ROOT / 'data' / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in part))
    print(name, len(part))
