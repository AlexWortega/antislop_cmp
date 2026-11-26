import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,read_prompt,predict_batch,PredictionUnavailable
OUT=ROOT/'round2'

def length_group(row):
 return '200plus' if row['tokens']>=200 else row['token_bucket']

def metrics(rows,predictions):
 tp=sum(r['slop'] and p is True for r,p in zip(rows,predictions));tn=sum(not r['slop'] and p is False for r,p in zip(rows,predictions))
 fp=sum(not r['slop'] and p is True for r,p in zip(rows,predictions));fn=sum(r['slop'] and p is False for r,p in zip(rows,predictions))
 positive=sum(r['slop'] for r in rows);negative=len(rows)-positive
 refused_positive=sum(r['slop'] and p is None for r,p in zip(rows,predictions));refused_negative=sum(not r['slop'] and p is None for r,p in zip(rows,predictions))
 return {'n':len(rows),'agreement':(tp+tn)/len(rows),'human_recall':tn/negative if negative else None,'slop_recall':tp/positive if positive else None,'balanced_accuracy':(tn/negative+tp/positive)/2 if negative and positive else None,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'unavailable':refused_positive+refused_negative,'unavailable_positive':refused_positive,'unavailable_negative':refused_negative}

def classify_batch(prompt,batch):
 try:return predict_batch(prompt,[r['text'] for r in batch])
 except PredictionUnavailable:
  if len(batch)!=1:raise
  return [None]

def assess(prompt,rows,batch_size=8,progress=False):
 batches=[rows[i:i+batch_size] for i in range(0,len(rows),batch_size)];predictions=[]
 with ThreadPoolExecutor(max_workers=6) as pool:
  for i,values in enumerate(pool.map(lambda b:classify_batch(prompt,b),batches)):
   predictions.extend(values)
   if progress and (i+1)%16==0:print('Scored batches',i+1,'/',len(batches),flush=True)
 overall=metrics(rows,predictions);by_bucket={}
 for key in sorted({length_group(r) for r in rows}):
  subset=[(r,p) for r,p in zip(rows,predictions) if length_group(r)==key]
  by_bucket[key]=metrics([r for r,p in subset],[p for r,p in subset])
 balanced=[m['balanced_accuracy'] for m in by_bucket.values() if m['balanced_accuracy'] is not None]
 objective=.5*overall['balanced_accuracy']+.5*sum(balanced)/len(balanced)
 return {'overall':overall,'by_bucket':by_bucket,'objective':objective},predictions

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--program_path',required=True);p.add_argument('--results_dir',required=True);a=p.parse_args()
 out=Path(a.results_dir);out.mkdir(parents=True,exist_ok=True)
 try:
  rows=[json.loads(l) for l in (OUT/'search.jsonl').read_text().splitlines()]
  report,predictions=assess(read_prompt(a.program_path),rows)
  # Balanced error feedback across length buckets and classes.
  strata=collections.defaultdict(list);anchors=[]
  for r,pred in zip(rows,predictions):
   if r['slop']!=pred:strata[(r['slop'],length_group(r))].append({'text':r['text'],'expected_slop':r['slop'],'predicted_slop':pred,'tokens_o200k':r['tokens']})
  errors=[r for key in sorted(strata) for r in strata[key][:3]]
  for label in [False,True]:
   for b in sorted({length_group(r) for r in rows}):
    anchor=next(({'text':r['text'],'expected_slop':label} for r,p in zip(rows,predictions) if r['slop']==p==label and length_group(r)==b),None)
    if anchor:anchors.append(anchor)
  result={'combined_score':report['objective'],'public':report,'text_feedback':json.dumps({'training_errors':errors,'correct_training_examples_to_preserve':anchors},ensure_ascii=False)}
  correct={'correct':True}
 except Exception as e:
  result={'combined_score':0,'text_feedback':type(e).__name__+': '+str(e)};correct={'correct':False,'error':result['text_feedback']}
 (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False));(out/'correct.json').write_text(json.dumps(correct))
