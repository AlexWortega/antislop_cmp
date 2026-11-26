"""Run ONCE after freezing the selected prompt. Never feed test errors to evolution."""
import argparse
import json
from pathlib import Path
from core import ROOT, read_prompt, score, digest

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('candidate')
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=1)
    a = p.parse_args()
    rows = [json.loads(line) for line in (ROOT/'data/test.jsonl').read_text().splitlines()]
    report = {}
    for name, path in [('baseline', ROOT/'initial.py'), ('optimized', Path(a.candidate))]:
        metrics, predictions = score(read_prompt(path), rows, batch_size=a.batch_size)
        metrics['prompt_hash'] = digest(read_prompt(path))
        metrics['batch_size'] = a.batch_size
        metrics['predictions'] = [{'hash': r['hash'], 'teacher_slop':r['slop'], 'opus_slop':v} for r,v in zip(rows,predictions)]
        report[name] = metrics
        Path(a.output).write_text(json.dumps(report, indent=2))
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
