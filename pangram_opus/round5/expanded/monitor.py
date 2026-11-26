import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from core import ROOT,digest,read_prompt
p=ROOT/'round5/expanded';rows=[json.loads(l) for l in (p/'train.jsonl').read_text().splitlines()];mode='default'
for d in sorted((ROOT/'runs/round5_expanded').glob('gen_*')):
 m=d/'results/metrics.json'
 if m.exists():
  r=json.loads(m.read_text());print(d.name,r['combined_score'],r.get('public',{}).get('overall',r.get('text_feedback')))
 elif (d/'main.py').exists():
  prompt=read_prompt(d/'main.py');print('Scoring',d.name,sum((ROOT/'cache'/('opus_runtime_'+digest(['strict-runtime-v1',mode,prompt,r['text'],None])+'.json')).exists() for r in rows),'/',len(rows))
