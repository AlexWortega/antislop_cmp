"""Dev-only selection, then one frozen comparison on the new grouped test set."""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,CONTRACT,digest,read_prompt
from evaluate_round import assess,metrics
OUT=ROOT/'round2'

def read(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]

def validate(run):
 records=[]
 for p in (ROOT/'runs'/run).glob('gen_*/results/metrics.json'):
  if not json.loads((p.parent/'correct.json').read_text())['correct']:continue
  r=json.loads(p.read_text());candidate=p.parent.parent/'main.py';prompt=read_prompt(candidate)
  records.append({'candidate':str(candidate),'search_objective':r['combined_score'],'prompt_hash':digest(prompt),'length':len(prompt)})
 records.sort(key=lambda r:(-r['search_objective'],r['length']))
 shortlist=[{'candidate':str(OUT/'previous_best.py'),'prompt_hash':digest(read_prompt(OUT/'previous_best.py')),'role':'previous'}]
 for r in records:
  if r['prompt_hash'] not in {x['prompt_hash'] for x in shortlist}:shortlist.append(r)
  if len(shortlist)==3:break
 (OUT/'shortlist.json').write_text(json.dumps(shortlist,indent=2))
 rows=read(OUT/'dev.jsonl');validated=[]
 for candidate in shortlist:
  print('Validating',candidate['candidate'],flush=True)
  report,pred=assess(read_prompt(candidate['candidate']),rows,batch_size=1,progress=True)
  record={**candidate,'report':report,'predictions':[{'hash':r['hash'],'slop':r['slop'],'prediction':p} for r,p in zip(rows,pred)]}
  validated.append(record);(OUT/'validation.json').write_text(json.dumps(validated,indent=2))
 chosen=max(validated,key=lambda r:(r['report']['objective'],r['report']['overall']['human_recall'],-len(read_prompt(r['candidate']))))
 shutil.copyfile(chosen['candidate'],OUT/'best_candidate.py');prompt=read_prompt(OUT/'best_candidate.py')
 (OUT/'best_prompt.txt').write_text(prompt);(OUT/'inference_system.txt').write_text(prompt+'\n\n'+CONTRACT)
 (OUT/'selection.json').write_text(json.dumps({'criterion':'max dev objective under single-text inference; human recall then prompt length as tie breaks','selected':{k:v for k,v in chosen.items() if k!='predictions'},'search_candidates':records},indent=2))
 print('Frozen selection',chosen['candidate'],chosen['report']['overall'],flush=True)

def test():
 rows=read(OUT/'test.jsonl');reports={}
 for name,path in [('previous',OUT/'previous_best.py'),('expanded',OUT/'best_candidate.py')]:
  print('Testing',name,flush=True)
  report,pred=assess(read_prompt(path),rows,batch_size=1,progress=True)
  report['prompt_hash']=digest(read_prompt(path))
  report['predictions']=[{'hash':r['hash'],'teacher_slop':r['slop'],'opus_slop':p} for r,p in zip(rows,pred)]
  for field in ['origin','source','scenario','model']:
   report['by_'+field]={}
   for key in sorted({r.get(field) or 'not_applicable' for r in rows}):
    subset=[(r,p) for r,p in zip(rows,pred) if (r.get(field) or 'not_applicable')==key]
    report['by_'+field][key]=metrics([r for r,p in subset],[p for r,p in subset])
  report['by_recommended_word_length']={}
  for short_input in [False,True]:
   subset=[(r,p) for r,p in zip(rows,pred) if (r['word_count']<50)==short_input]
   if subset:report['by_recommended_word_length']['below50' if short_input else 'atleast50']=metrics([r for r,p in subset],[p for r,p in subset])
  short=[(r,p) for r,p in zip(rows,pred) if r['tokens']<200]
  report['under_200']=metrics([r for r,p in short],[p for r,p in short])
  human=[(r,p) for r,p in zip(rows,pred) if r['origin']=='human' and not r['slop']]
  report['human_origin_teacher_negative']=metrics([r for r,p in human],[p for r,p in human])
  reports[name]=report;(OUT/'test_results.json').write_text(json.dumps(reports,indent=2));print(name,report['overall'],flush=True)
 pairs=list(zip(reports['previous']['predictions'],reports['expanded']['predictions']))
 better=sum(a['opus_slop']!=a['teacher_slop'] and b['opus_slop']==b['teacher_slop'] for a,b in pairs)
 worse=sum(a['opus_slop']==a['teacher_slop'] and b['opus_slop']!=b['teacher_slop'] for a,b in pairs)
 n=better+worse;pvalue=min(1,2*sum(math.comb(n,k) for k in range(min(better,worse)+1))/2**n) if n else 1
 comparison={'improved':better,'regressed':worse,'exact_mcnemar_p':pvalue,'n':len(rows)}
 (OUT/'comparison.json').write_text(json.dumps(comparison,indent=2));print(comparison,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','test']);p.add_argument('--run',default='expanded1');a=p.parse_args()
 if a.action=='validate':validate(a.run)
 else:test()
