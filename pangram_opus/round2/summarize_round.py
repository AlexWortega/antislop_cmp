"""Summarize frozen test outcomes and uncertainty without additional API calls."""
import collections
import json
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,digest,read_prompt
from evaluate_round import metrics
OUT=ROOT/'round2'

def main():
 rows=[json.loads(l) for l in (OUT/'test.jsonl').read_text().splitlines()]
 reports=json.loads((OUT/'test_results.json').read_text())
 assert set(reports)=={'previous','expanded'}
 for name,filename in [('previous','previous_best.py'),('expanded','best_candidate.py')]:
  report=reports[name]
  assert report['prompt_hash']==digest(read_prompt(OUT/filename))
  assert [r['hash'] for r in rows]==[r['hash'] for r in report['predictions']]
 groups=collections.defaultdict(list)
 for i,r in enumerate(rows):groups[r['split_group']].append(i)
 keys=sorted(groups);rng=random.Random(30291);deltas=[]
 for _ in range(4000):
  indices=[i for k in rng.choices(keys,k=len(keys)) for i in groups[k]]
  sampled=[rows[i] for i in indices]
  values={name:metrics(sampled,[report['predictions'][i]['opus_slop'] for i in indices])['balanced_accuracy'] for name,report in reports.items()}
  if all(v is not None for v in values.values()):deltas.append(values['expanded']-values['previous'])
 deltas.sort()
 budget=json.loads((ROOT/'cache/budget.json').read_text())
 costs={provider:{'settled_with_margin':budget['spent'][provider], 'pending_reservations':sum(r['amount'] for r in budget['pending'].values() if r['provider']==provider),'limit':limit} for provider,limit in budget['limits'].items()}
 for c in costs.values():c['conservative_total']=c['settled_with_margin']+c['pending_reservations'];assert c['conservative_total']<=c['limit']
 compact={name:{k:v for k,v in report.items() if k!='predictions'} for name,report in reports.items()}
 search=[]
 for path in sorted((ROOT/'runs/expanded1').glob('gen_*/results/metrics.json')):
  record=json.loads(path.read_text())
  search.append({'generation':path.parts[-3],'correct':json.loads((path.parent/'correct.json').read_text())['correct'],'objective':record['combined_score'],'overall':record.get('public',{}).get('overall')})
 result={'protocol':'New group-disjoint test, frozen dev-selected prompt, strict single-text inference','selection':json.loads((OUT/'selection.json').read_text())['selected'],'test':compact,'paired':json.loads((OUT/'comparison.json').read_text()),'group_bootstrap_balanced_accuracy_delta':{'groups':len(groups),'resamples':len(deltas),'percentile95':[deltas[int(.025*len(deltas))],deltas[int(.975*len(deltas))]],'note':'Exploratory interval; sampling units are source/topic groups, not independent texts.'},'budget_session':costs,'split':json.loads((OUT/'split_manifest.json').read_text()),'labels':json.loads((OUT/'label_summary.json').read_text())}
 result['search_results']=search
 result['validation']=[{k:v for k,v in r.items() if k!='predictions'} for r in json.loads((OUT/'validation.json').read_text())]
 (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'metrics':{k:v['overall'] for k,v in compact.items()},'bootstrap':result['group_bootstrap_balanced_accuracy_delta'],'costs':costs},indent=2))

if __name__=='__main__':main()
