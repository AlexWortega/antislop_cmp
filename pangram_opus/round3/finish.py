import json,math,random,collections,shutil,sys
from pathlib import Path
from evaluate import OUT,ROOT,read,assess,read_prompt
from core import CONTRACT,digest
from evaluate_round import metrics
from runtime import assess_mode

def validate():
 records=[]
 for path in (ROOT/'runs/single_pairwise2').glob('gen_*/results/metrics.json'):
  if not json.loads((path.parent/'correct.json').read_text())['correct']:continue
  data=json.loads(path.read_text());candidate=path.parent.parent/'main.py';records.append({'candidate':str(candidate),'search_objective':data['combined_score'],'prompt_hash':digest(read_prompt(candidate))})
 records.sort(key=lambda r:-r['search_objective']);shortlist=[{'candidate':str(OUT/'previous_best_candidate.py'),'prompt_hash':digest(read_prompt(OUT/'previous_best_candidate.py'))}]
 for r in records:
  if r['prompt_hash']!=shortlist[0]['prompt_hash']:shortlist.append(r);break
 (OUT/'shortlist.json').write_text(json.dumps(shortlist,indent=2));rows=read(OUT/'dev.jsonl');results=[]
 for r in shortlist:
  print('Validate',r['candidate'],flush=True);report,pred=assess(read_prompt(r['candidate']),rows,batch_size=1,progress=True)
  results.append({**r,'report':report,'predictions':pred});(OUT/'validation.json').write_text(json.dumps(results,indent=2));print(report['overall'],flush=True)
 chosen=max(results,key=lambda r:(r['report']['objective'],r['report']['overall']['human_recall'],-len(read_prompt(r['candidate']))))
 shutil.copyfile(chosen['candidate'],OUT/'best_candidate.py');prompt=read_prompt(OUT/'best_candidate.py');(OUT/'best_prompt.txt').write_text(prompt);(OUT/'inference_system.txt').write_text(prompt+'\n\n'+CONTRACT)
 (OUT/'selection.json').write_text(json.dumps({'criterion':'predeclared reused-dev length-balanced objective; no fresh test information','selected':{k:v for k,v in chosen.items() if k!='predictions'},'search_candidates':records},indent=2));print('Frozen',chosen['candidate'],flush=True)

def test():
 rows=read(OUT/'test.jsonl');reports={}
 for name,path in [('previous',OUT/'previous_best_candidate.py'),('selected',OUT/'best_candidate.py')]:
  print('Test',name,flush=True);mode='minimal' if name=='previous' else json.loads((OUT/'runtime_selection.json').read_text())['mode'];report,pred=assess_mode(read_prompt(path),rows,mode,progress=True);report['inference_mode']=mode
  report['prompt_hash']=digest(read_prompt(path));report['predictions']=[{'hash':r['hash'],'teacher':r['slop'],'prediction':p} for r,p in zip(rows,pred)]
  for field in ['origin','source','scenario','model']:
   report['by_'+field]={}
   for key in sorted({r.get(field) or 'not_applicable' for r in rows}):
    sub=[(r,p) for r,p in zip(rows,pred) if (r.get(field) or 'not_applicable')==key];report['by_'+field][key]=metrics([r for r,p in sub],[p for r,p in sub])
  for key,condition in [('under200',lambda r:r['tokens']<200),('below50words',lambda r:r['word_count']<50)]:
   sub=[(r,p) for r,p in zip(rows,pred) if condition(r)];report[key]=metrics([r for r,p in sub],[p for r,p in sub])
  reports[name]=report;(OUT/'test_results.json').write_text(json.dumps(reports,indent=2));print(report['overall'],flush=True)
 pairs=zip(reports['previous']['predictions'],reports['selected']['predictions']);better=worse=0
 for a,b in pairs:
  better+=a['prediction']!=a['teacher'] and b['prediction']==b['teacher'];worse+=a['prediction']==a['teacher'] and b['prediction']!=b['teacher']
 n=better+worse;comparison={'improved':better,'regressed':worse,'mcnemar_casewise_exploratory_p':min(1,2*sum(math.comb(n,k) for k in range(min(better,worse)+1))/2**n) if n else 1}
 groups=collections.defaultdict(list)
 for i,r in enumerate(rows):groups[r['split_group']].append(i)
 rng=random.Random(3199);deltas=[];keys=sorted(groups)
 for _ in range(3000):
  indices=[i for k in rng.choices(keys,k=len(keys)) for i in groups[k]];sample=[rows[i] for i in indices]
  v={name:metrics(sample,[report['predictions'][i]['prediction'] for i in indices])['balanced_accuracy'] for name,report in reports.items()}
  if all(x is not None for x in v.values()):deltas.append(v['selected']-v['previous'])
 deltas.sort();comparison['group_bootstrap_delta95']=[deltas[int(.025*len(deltas))],deltas[int(.975*len(deltas))]];comparison['groups']=len(groups)
 (OUT/'comparison.json').write_text(json.dumps(comparison,indent=2));print(comparison,flush=True)
if __name__=='__main__':{'validate':validate,'test':test}[sys.argv[1]]()
