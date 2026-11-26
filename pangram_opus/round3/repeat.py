"""Predetermined length-stratified repeat check, never used for prompt selection."""
import json
from concurrent.futures import ThreadPoolExecutor
from round3.evaluate import OUT,read,read_prompt
from core import PredictionUnavailable
from runtime import predict_mode

def check():
 target=OUT/'repeatability.json'
 if target.exists():return json.loads(target.read_text())
 rows=read(OUT/'test.jsonl');selected=[]
 for bucket in ['lt25','25-49','50-99','100-199']:
  for label in [False,True]:
   choices=sorted((r for r in rows if r['token_bucket']==bucket and r['slop']==label),key=lambda r:r['hash'])
   if choices:selected.append(choices[0])
 mode=json.loads((OUT/'runtime_selection.json').read_text())['mode'];prompt=read_prompt(OUT/'best_candidate.py');base={r['hash']:r['prediction'] for r in json.loads((OUT/'test_results.json').read_text())['selected']['predictions']}
 def one(job):
  r,rep=job
  try:v=predict_mode(prompt,r['text'],mode,cache_tag='round3-repeat-v1:'+str(rep))
  except PredictionUnavailable:v=None
  return r['hash'],rep,v
 values={r['hash']:[base[r['hash']],None,None] for r in selected}
 with ThreadPoolExecutor(max_workers=6) as pool:
  for h,rep,v in pool.map(one,[(r,rep) for r in selected for rep in [1,2]]):values[h][rep]=v
 cases=[{'hash':r['hash'],'bucket':r['token_bucket'],'teacher':r['slop'],'predictions':values[r['hash']]} for r in selected]
 result={'n':len(cases),'calls_per_text':3,'stable_cases':sum(len(set(r['predictions']))==1 for r in cases),'unavailable_calls':sum(x is None for r in cases for x in r['predictions']),'cases':cases,'selection':'First hash per available class in each <200-token bucket; not based on correctness. Diagnostic only.'}
 target.write_text(json.dumps(result,indent=2));print('Repeatability',result['stable_cases'],'/',result['n'],flush=True);return result
