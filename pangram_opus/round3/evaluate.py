import collections,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'round2'))
from core import ROOT,read_prompt
from evaluate_round import assess,length_group
OUT=ROOT/'round3'
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--program_path',required=True);p.add_argument('--results_dir',required=True);a=p.parse_args();out=Path(a.results_dir);out.mkdir(exist_ok=True,parents=True)
 try:
  rows=read(OUT/'search.jsonl');prompt=read_prompt(a.program_path)
  normalized=' '.join(prompt.lower().split())
  for r in rows:
   sentence=' '.join(r['text'].lower().split())
   if len(sentence.split())>=9 and sentence in normalized:raise ValueError('Prompt memorizes a full training example')
  report,predictions=assess(prompt,rows,batch_size=1,progress=True)
  strata=collections.defaultdict(list);anchors=[]
  for r,pred in zip(rows,predictions):
   if pred!=r['slop']:strata[(r['slop'],length_group(r))].append({'text':r['text'],'expected_slop':r['slop'],'prediction':pred,'tokens':r['tokens']})
  for label in [False,True]:
   for b in sorted({length_group(r) for r in rows}):
    anchor=next(({'text':r['text'],'expected_slop':label,'tokens':r['tokens']} for r,pred in zip(rows,predictions) if pred==r['slop']==label and length_group(r)==b),None)
    if anchor:anchors.append(anchor)
  feedback={'errors':[r for key in sorted(strata) for r in strata[key][:3]],'preserve_correct':anchors,'warning':'All scores are SINGLE-TEXT, not batch. Do not copy exact examples into the prompt. Preserve specificity AND short positive sensitivity.'}
  result={'combined_score':report['objective'],'public':report,'text_feedback':json.dumps(feedback,ensure_ascii=False)};correct={'correct':True}
  (out/'predictions.json').write_text(json.dumps([{'hash':r['hash'],'teacher':r['slop'],'prediction':p} for r,p in zip(rows,predictions)]))
 except Exception as e:result={'combined_score':0,'text_feedback':type(e).__name__+': '+str(e)};correct={'correct':False,'error':result['text_feedback']}
 (out/'metrics.json').write_text(json.dumps(result));(out/'correct.json').write_text(json.dumps(correct))
