"""Freeze a candidate using TRAINING metrics only, before opening holdout results."""
import argparse
import json
import shutil
from pathlib import Path
from core import ROOT, CONTRACT, read_prompt, digest

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run')
    a = p.parse_args()
    records = []
    for metrics_path in (ROOT/'runs'/a.run).glob('gen_*/results/metrics.json'):
        candidate = metrics_path.parent.parent/'main.py'
        correct = json.loads((metrics_path.parent/'correct.json').read_text())
        metrics = json.loads(metrics_path.read_text())
        if not correct['correct']: continue
        prompt = read_prompt(candidate)
        records.append({'candidate':str(candidate.relative_to(ROOT)), 'metrics':metrics['public'],
                        'prompt_hash':digest(prompt),'length':len(prompt)})
    records.sort(key=lambda x: (-x['metrics']['balanced_accuracy'], x['length']))
    best = records[0]
    prompt = read_prompt(ROOT/best['candidate'])
    shutil.copyfile(ROOT/best['candidate'], ROOT/'best_candidate.py')
    (ROOT/'best_prompt.txt').write_text(prompt)
    (ROOT/'inference_system.txt').write_text(prompt+'\n\n'+CONTRACT)
    (ROOT/'selection.json').write_text(json.dumps({'criterion':'maximum training balanced accuracy; shortest prompt tie-break',
        'selected':best,'candidates':records},indent=2))
    print(json.dumps(best,indent=2))
