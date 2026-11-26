"""Label a bounded dataset and split by source ID to prevent paired-text leakage."""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from core import ROOT, digest, teacher, teacher_label

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+')
    p.add_argument('--limit', type=int, default=100)
    a = p.parse_args()
    rows, seen = [], set()
    for file in a.inputs:
        for line in Path(file).read_text().splitlines():
            row = json.loads(line)
            text = row['text'].strip()
            h = digest(' '.join(text.split()))
            if h in seen or len(text.split()) < 80:
                continue
            seen.add(h)
            rows.append({'text': text, 'group': str(row.get('group_id', row.get('id', h))), 'hash': h})
    random.Random(42).shuffle(rows)
    rows = rows[:a.limit]
    groups = sorted({r['group'] for r in rows})
    random.Random(43).shuffle(groups)
    test_groups = set(groups[:max(1, len(groups)//4)])
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    splits = {'train': [], 'test': []}
    def label(row):
        return teacher(row['text'])
    pool = ThreadPoolExecutor(max_workers=4)
    for i, (row, result) in enumerate(zip(rows, pool.map(label, rows))):
        row['slop'] = teacher_label(result)
        row['teacher_version'] = result['version']
        splits['test' if row['group'] in test_groups else 'train'].append(row)
        print(f'Labeled {i+1}/{len(rows)}', flush=True)
    pool.shutdown()
    for name, subset in splits.items():
        (data / (name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in subset))
        print(name, len(subset), 'slop', sum(r['slop'] for r in subset))
