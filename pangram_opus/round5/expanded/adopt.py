"""Adopt only a calibration-improving expanded candidate, before fresh labels."""
import json,shutil,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from core import ROOT,read_prompt,digest
OUT=ROOT/'round5/expanded';ROUND=ROOT/'round5'
def read(path):return [json.loads(l) for l in path.read_text().splitlines()]
if __name__=='__main__':
 assert not (ROUND/'test.jsonl').exists(),'Do not reselect after final labels'
 selection=json.loads((OUT/'selection.json').read_text());assert selection['fresh_label_gate'],'No qualifying calibration improvement'
 prompt=read_prompt(OUT/'best_candidate.py');assert digest(prompt)==selection['selected']['prompt_hash']
 budget=json.loads((ROOT/'cache/budget.json').read_text());available=budget['limits']['openrouter']-budget['spent']['openrouter']-sum(v['amount'] for v in budget['pending'].values() if v['provider']=='openrouter')
 def estimate(p,rows,n):
  values=[]
  for r in rows:
   f=ROOT/'cache'/('opus_runtime_'+digest(['strict-runtime-v1','default',p,r['text'],None])+'.json')
   if f.exists():
    cost=json.loads(f.read_text()).get('usage',{}).get('cost')
    if cost is not None:values.append(cost)
  assert values,'Need observed costs before fresh evaluation'
  values.sort();return n*1.1*max(statistics.mean(values)*1.25,values[int(.95*(len(values)-1))]),len(values)
 n=len(read(ROUND/'fresh_unlabeled.jsonl'));new_cost,new_n=estimate(prompt,read(OUT/'train.jsonl')+read(OUT/'dev.jsonl'),n+16);old_cost,old_n=estimate(read_prompt(ROUND/'previous_best_candidate.py'),read(OUT/'train.jsonl')+read(OUT/'dev.jsonl'),n)
 check={'openrouter_remaining':available,'estimated_new_with_margin':new_cost,'estimated_old_with_margin':old_cost,'observed_new_calls':new_n,'observed_old_calls':old_n,'n_final':n};(ROUND/'final_budget_check.json').write_text(json.dumps(check,indent=2));assert available>=new_cost+old_cost,'Preserve budget for complete final comparison'
 for name in ['best_candidate.py','best_prompt.txt','inference_system.txt','runtime_selection.json']:shutil.copyfile(OUT/name,ROUND/name)
 selection['calibration_stage']='expanded';selection['calibration_train']='expanded/train.jsonl';selection['calibration_dev']='expanded/dev.jsonl';(ROUND/'selection.json').write_text(json.dumps(selection,indent=2));print('Adopted expanded candidate; final budget estimate',check,flush=True)
