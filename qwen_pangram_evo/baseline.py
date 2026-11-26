"""Score a system prompt (default: none) on a split. Usage: baseline.py [train|heldout] [candidate.py]"""
import json
import sys
from common import ROOT, load_rows, read_candidate, score_set

split = sys.argv[1] if len(sys.argv) > 1 else 'train'
system = read_candidate(sys.argv[2]) if len(sys.argv) > 2 else ''
summary, results = score_set(system, load_rows(split + '.jsonl'))
tag = 'none' if len(sys.argv) <= 2 else __import__('pathlib').Path(sys.argv[2]).stem
out = ROOT / 'reports' / f'{split}_{tag}.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({'summary': summary, 'results': results}, ensure_ascii=False, indent=1))
print(json.dumps(summary, indent=1))
for r in results:
    print(f"{r['kind']:8} pass={int(r['passed'])} slop={r['slop']} task={r['task']} words={r['words']}/{r['target']}")
