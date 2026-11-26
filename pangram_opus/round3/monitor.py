import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,MODEL,digest,read_prompt
OUT=ROOT/'round3'
if (OUT/'pipeline_status.json').exists():print('Pipeline',json.loads((OUT/'pipeline_status.json').read_text()))
records=[];rows=[json.loads(l) for l in (OUT/'search.jsonl').read_text().splitlines()]
for d in sorted((ROOT/'runs/single_pairwise2').glob('gen_*'),key=lambda p:int(p.name[4:])):
 metrics=d/'results/metrics.json'
 if metrics.exists():
  r=json.loads(metrics.read_text());records.append({'gen':d.name,'objective':round(r['combined_score'],4),'overall':r.get('public',{}).get('overall')})
 elif (d/'main.py').exists():
  try:prompt=read_prompt(d/'main.py')
  except ValueError:continue
  n=sum((ROOT/'cache'/('opus_batch_'+digest([MODEL,'strict-schema-batch-v2',prompt,[r['text']]])+'.json')).exists() for r in rows)
  print('Scoring',d.name,n,'/',len(rows))
if records:
 def brief(r):
  m=r['overall'] or {};return {k:v for k,v in {'gen':r['gen'],'objective':r['objective'],'fp':m.get('fp'),'fn':m.get('fn'),'unavailable':m.get('unavailable')}.items()}
 print('Completed',len(records),'latest',brief(records[-1]),'best',brief(max(records,key=lambda r:r['objective'])))
b=json.loads((ROOT/'cache/budget.json').read_text());print('Conservative budget',{k:round(b['spent'][k]+sum(v['amount'] for v in b['pending'].values() if v['provider']==k),2) for k in b['limits']})
