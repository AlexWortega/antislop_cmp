import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,read_prompt
from round3.runtime import assess_mode
OUT=ROOT/'round5'
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--program_path',required=True);p.add_argument('--results_dir',required=True);a=p.parse_args();out=Path(a.results_dir);out.mkdir(parents=True,exist_ok=True)
 try:
  prompt=read_prompt(a.program_path);suffix=(OUT/'reference_suffix.txt').read_text()
  if suffix not in prompt:raise ValueError('Do not change labeled reference examples')
  initial=read_prompt(OUT/'initial.py')
  rulebook=initial.split('<reference_rulebook>',1)[1].split('</reference_rulebook>',1)[0]
  if '<reference_rulebook>'+rulebook+'</reference_rulebook>' not in prompt:raise ValueError('Keep the reference rulebook intact')
  rows=[json.loads(l) for l in (OUT/'train.jsonl').read_text().splitlines()];mode='default';report,pred=assess_mode(prompt,rows,mode,True)
  errors=[{'text':r['text'],'expected_slop':r['slop'],'prediction':v,'tokens':r['tokens']} for r,v in zip(rows,pred) if v is not None and v!=r['slop']]
  anchors=[]
  for label in [False,True]:
   for b in ['lt25','25-49','50-99','100-199','200-349']:
    x=next(({'text':r['text'],'expected_slop':label} for r,v in zip(rows,pred) if r['slop']==v==label and r['token_bucket']==b),None)
    if x:anchors.append(x)
  # Do not optimize refusal behavior or embed entire calibration examples.
  normalized=' '.join(prompt.lower().split())
  original=' '.join(read_prompt(OUT/'initial.py').lower().split())
  for r in rows:
   text=' '.join(r['text'].lower().split())
   if len(text.split())>=9 and text in normalized and text not in original:raise ValueError('Mutation copied a calibration text verbatim')
  result={'combined_score':report['objective'],'public':report,'text_feedback':json.dumps({'errors':errors[:24],'preserve_correct':anchors,'note':'This is training, including the former round3 test. Fresh test remains unseen. Fix general mechanisms, not exact sentences/names/topics; do not change safety/refusal behavior.'})};correct={'correct':True}
  (out/'predictions.json').write_text(json.dumps([{'hash':r['hash'],'teacher':r['slop'],'prediction':v} for r,v in zip(rows,pred)]))
 except Exception as e:result={'combined_score':0,'text_feedback':type(e).__name__+': '+str(e)};correct={'correct':False,'error':result['text_feedback']}
 (out/'metrics.json').write_text(json.dumps(result));(out/'correct.json').write_text(json.dumps(correct))
