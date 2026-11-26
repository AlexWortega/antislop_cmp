"""Repair a sparse calibration stratum before any dev scoring or final outcomes."""
import collections,json,math,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,teacher,teacher_label
from round3 import fresh as f
OUT=ROOT/'round5';f.OUT=OUT/'dev_extension_work'
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def save(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
if __name__=='__main__':
 f.OUT.mkdir(parents=True,exist_ok=True)
 assert not (OUT/'validation.json').exists(),'Declare and build extension before dev scoring'
 ledger=json.loads((ROOT/'cache/budget.json').read_text());remaining=100-ledger['spent']['pangram']-sum(v['amount'] for v in ledger['pending'].values() if v['provider']=='pangram');fresh=read(OUT/'fresh_unlabeled.jsonl');reserved_estimate=.055*1.15*sum(max(1,math.ceil(len(r['text'].split())/100)) for r in fresh)
 assert remaining>=reserved_estimate+.055*1.15*12,'Preserve full fresh-test budget'
 topics=[('technical','Briefly explain why bread dough rises while yeast is active. Use a single paragraph of plain accurate prose.'),('business','Write a brief factual notice for residents saying the shared laundry room will be unavailable for a morning inspection and asking them to remove clothes beforehand.')]
 def one(job):
  model,genre,task=job
  for attempt in range(3):
   instruction=task+' Write approximately '+(['32–40','28–36','34–42'][attempt])+' English tokens. Return only the text.'
   r=f.generate((model,'direct_'+genre,'round5-devext:'+genre,'25-49',instruction,None))
   if 25<=r['tokens']<50 and len(r['text'].split())<80:
    t=teacher(r['text']);r.update(split_group=r['group'],slop=teacher_label(t),teacher_version=t['version'],teacher_status='ok');return r
  raise RuntimeError('Could not obtain required calibration length')
 if (OUT/'dev_extension.jsonl').exists():rows=read(OUT/'dev_extension.jsonl')
 else:
  with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(one,[(model,genre,task) for model in f.MODELS for genre,task in topics]))
  save(OUT/'dev_extension.jsonl',rows)
 prior=read(OUT/'prior.jsonl');existing={r['hash'] for r in prior};assert all(r['hash'] not in {x['hash'] for x in fresh} for r in rows);prior += [r for r in rows if r['hash'] not in existing];save(OUT/'prior.jsonl',prior)
 dev=read(OUT/'dev.jsonl');existing={r['hash'] for r in dev};dev += [r for r in rows if r['hash'] not in existing];save(OUT/'dev.jsonl',dev)
 protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(dev_n=len(dev),prior_rows=len(prior),dev_extension={'n':len(rows),'positive':sum(r['slop'] for r in rows),'reason':'Original calibration dev had only one positive in 25–49 tokens. Twelve fixed generation jobs across six models and two new topics were specified before any dev scoring; both returned classes are retained.','final_test_unseen':True,'teacher_cost_estimate':.055*len(rows),'fresh_budget_preserved_estimate':reserved_estimate});(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 print('Expanded dev',len(dev),'new class counts',dict(collections.Counter(r['slop'] for r in rows)),flush=True)
