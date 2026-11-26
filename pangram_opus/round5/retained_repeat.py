"""Repeat the unchanged working prompt on predetermined fresh-test cases."""
import json,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,read_prompt,digest,PredictionUnavailable
from round3.runtime import predict_mode
OUT=ROOT/'round5'
if __name__=='__main__':
 rows=[json.loads(l) for l in (OUT/'test.jsonl').read_text().splitlines()]
 prompt=read_prompt(ROOT/'best_candidate.py');report=json.loads((OUT/'test_results.json').read_text())['previous'];assert digest(prompt)==report['prompt_hash']
 target=OUT/'retained_repeatability.json'
 if target.exists():
  result=json.loads(target.read_text());assert result['prompt_hash']==digest(prompt);print(result);raise SystemExit
 selected=[]
 for bucket in ['lt25','25-49','50-99','100-199']:
  for label in [False,True]:
   choices=sorted((r for r in rows if r['token_bucket']==bucket and r['slop']==label),key=lambda r:r['hash'])
   if choices:selected.append(choices[0])
 base={r['hash']:r['prediction'] for r in report['predictions']};values={r['hash']:[base[r['hash']],None,None] for r in selected}
 def one(job):
  row,rep=job
  try:value=predict_mode(prompt,row['text'],'default',cache_tag='round3-repeat-v1:'+str(rep))
  except PredictionUnavailable:value=None
  return row['hash'],rep,value
 with ThreadPoolExecutor(max_workers=6) as pool:
  for h,rep,value in pool.map(one,[(r,k) for r in selected for k in [1,2]]):values[h][rep]=value
 cases=[{'hash':r['hash'],'bucket':r['token_bucket'],'teacher':r['slop'],'predictions':values[r['hash']]} for r in selected]
 result={'prompt_hash':digest(prompt),'n':len(cases),'calls_per_text':3,'stable_cases':sum(len(set(r['predictions']))==1 for r in cases),'unavailable_calls':sum(v is None for r in cases for v in r['predictions']),'cases':cases,'selection':'First hash per teacher class and short-length bin, independent of correctness. Diagnostic only.'}
 target.write_text(json.dumps(result,indent=2));print(result)
