"""Show run progress and session-wide conservative costs, without credentials."""
import argparse
import json
from core import ROOT

p=argparse.ArgumentParser();p.add_argument('--run',default='single_pairwise2');p.add_argument('--round',default='round3');a=p.parse_args()
status=ROOT/a.round/'pipeline_status.json'
if status.exists():print('Pipeline:',json.loads(status.read_text()))
records=[]
for path in (ROOT/'runs'/a.run).glob('gen_*/results/metrics.json'):
 r=json.loads(path.read_text());records.append({'generation':int(path.parent.parent.name.split('_')[1]),'objective':r['combined_score'],'metrics':r.get('public',{})})
print('Evaluated candidates:',len(records))
if records:print('Best training:',max(records,key=lambda r:r['objective']))
validation=ROOT/a.round/'validation.json'
if validation.exists():
 for r in json.loads(validation.read_text()):print('Dev:',r['candidate'],r['report']['overall'])
r=json.loads((ROOT/'cache/budget.json').read_text())
print('Conservative costs incl reservations:',{k:round(r['spent'][k]+sum(x['amount'] for x in r['pending'].values() if x['provider']==k),3) for k in r['limits']})
