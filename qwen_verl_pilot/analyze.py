"""Summarize actual held-out dumps, matching topic/scaffold by prompt content."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from build_data import VAL, SCAFFOLDS

ROOT = Path(__file__).resolve().parent

def summarize(rows):
    return {'n': len(rows), **{
        key: mean(float(r[key]) for r in rows)
        for key in ('score', 'quality', 'anti_slop', 'invalid')
        if rows and all(key in r for r in rows)
    }}

def main():
    result = {}
    for path in sorted((ROOT/'artifacts'/'validation').glob('*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        groups = defaultdict(list)
        ids = set()
        for row in rows:
            prompt = row['input']
            topics = [topic for topic, question in VAL if question in prompt]
            scaffolds = [name for name, system in SCAFFOLDS.items() if system in prompt]
            assert len(topics) == len(scaffolds) == 1, 'Ambiguous prompt identity'
            pair = (topics[0], scaffolds[0])
            assert pair not in ids, 'Duplicate held-out prompt'
            ids.add(pair)
            groups[scaffolds[0]].append(row)
        assert len(ids) == 16, 'Incomplete validation set'
        result[path.stem] = {'overall': summarize(rows), 'by_scaffold': {
            name: summarize(group) for name, group in groups.items()}}
    if not result:
        raise SystemExit('No completed validation dump: no quality claim is possible.')
    (ROOT/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
