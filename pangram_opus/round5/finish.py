import collections,json,math,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,read_prompt,digest
from round3.runtime import assess_mode
from round2.evaluate_round import metrics
from round3.goal_check import check
OUT=ROOT/'round5'
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
if __name__=='__main__':
 rows=read(OUT/'test.jsonl');reports={}
 for name,path in [('previous',OUT/'previous_best_candidate.py'),('selected',OUT/'best_candidate.py')]:
  print('Fresh test',name,flush=True);prompt=read_prompt(path);mode='default' if name=='previous' else json.loads((OUT/'runtime_selection.json').read_text())['mode'];report,pred=assess_mode(prompt,rows,mode,True)
  report.update(prompt_hash=digest(prompt),inference_mode=mode,predictions=[{'hash':r['hash'],'teacher':r['slop'],'prediction':v} for r,v in zip(rows,pred)])
  for field in ['origin','source','scenario','model']:
   report['by_'+field]={}
   for key in sorted({r.get(field) or 'not_applicable' for r in rows}):
    pairs=[(r,p) for r,p in zip(rows,pred) if (r.get(field) or 'not_applicable')==key];report['by_'+field][key]=metrics([r for r,p in pairs],[p for r,p in pairs])
  for key,condition in [('under200',lambda r:r['tokens']<200),('below50words',lambda r:r['word_count']<50)]:
   pairs=[(r,p) for r,p in zip(rows,pred) if condition(r)];report[key]=metrics([r for r,p in pairs],[p for r,p in pairs])
  reports[name]=report;(OUT/'test_results.json').write_text(json.dumps(reports,indent=2));print(name,report['overall'],flush=True)
 better=worse=0
 for a,b in zip(reports['previous']['predictions'],reports['selected']['predictions']):
  better+=a['prediction']!=a['teacher'] and b['prediction']==b['teacher'];worse+=a['prediction']==a['teacher'] and b['prediction']!=b['teacher']
 n=better+worse;comparison={'improved':better,'regressed':worse,'mcnemar_casewise_exploratory_p':min(1,2*sum(math.comb(n,k) for k in range(min(better,worse)+1))/2**n) if n else 1}
 groups=collections.defaultdict(list)
 for i,r in enumerate(rows):groups[r['split_group']].append(i)
 rng=random.Random(4177);deltas=[];absolute_ba=[];absolute_agreement=[];keys=sorted(groups)
 for _ in range(3000):
  indices=[i for k in rng.choices(keys,k=len(keys)) for i in groups[k]];sample=[rows[i] for i in indices];vals={name:metrics(sample,[report['predictions'][i]['prediction'] for i in indices])['balanced_accuracy'] for name,report in reports.items()}
  selected_metrics=metrics(sample,[reports['selected']['predictions'][i]['prediction'] for i in indices])
  absolute_agreement.append(selected_metrics['agreement'])
  if vals['selected'] is not None:absolute_ba.append(vals['selected'])
  if all(v is not None for v in vals.values()):deltas.append(vals['selected']-vals['previous'])
 deltas.sort();comparison['group_bootstrap_delta95']=[deltas[int(.025*len(deltas))],deltas[int(.975*len(deltas))]];comparison['groups']=len(groups);(OUT/'comparison.json').write_text(json.dumps(comparison,indent=2))
 def interval(values):
  values=sorted(values);return [values[int(.025*len(values))],values[int(.975*len(values))]]
 uncertainty={'method':'Exploratory percentile bootstrap over source/topic groups, 3000 resamples; not a guarantee on future sources or models.','groups':len(groups),'agreement95':interval(absolute_agreement),'balanced_accuracy95':interval(absolute_ba)}
 (OUT/'uncertainty.json').write_text(json.dumps(uncertainty,indent=2))
 assessment=check(reports['selected'],json.loads((OUT/'goal_criteria.json').read_text()));(OUT/'goal_assessment.json').write_text(json.dumps(assessment,indent=2));print('Goal criteria',assessment,flush=True)
