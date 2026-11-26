import argparse
import collections
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,digest,predict_batch,read_prompt
from data_build import OUT,tokens,bucket
from expand import read,save

class Union:
 def __init__(self,keys):self.parents={k:k for k in keys}
 def root(self,k):
  if self.parents[k]!=k:self.parents[k]=self.root(self.parents[k])
  return self.parents[k]
 def join(self,a,b):self.parents[self.root(a)]=self.root(b)

def split():
 labeled=read(OUT/'human_labeled.jsonl')+read(OUT/'extras_labeled.jsonl')+read(OUT/'direct_labeled.jsonl')
 failures=[r for r in labeled if r['teacher_status']!='ok']
 save(OUT/'teacher_failures.jsonl',failures)
 if failures:raise RuntimeError(f'{len(failures)} teacher failures need review/retry before splitting')
 u=Union({r['group'] for r in labeled});seen={};normalized_groups={}
 for r in labeled:
  r['hash']=digest(r['text'])
  normalized=' '.join(r['text'].split())
  if normalized in normalized_groups:u.join(r['group'],normalized_groups[normalized])
  else:normalized_groups[normalized]=r['group']
  if r['hash'] in seen:u.join(r['group'],seen[r['hash']]['group'])
  else:seen[r['hash']]=r
 rows=list(seen.values())
 # Group near-duplicates, including boilerplate shared by different emails.
 grams=[]
 for r in rows:
  words=re.findall(r'\w+',r['text'].lower());grams.append({tuple(words[i:i+5]) for i in range(len(words)-4)})
 for i,a in enumerate(rows):
  if len(grams[i])<8:continue
  for j in range(i):
   if len(grams[j])>=8 and len(grams[i]&grams[j])/min(len(grams[i]),len(grams[j]))>=.8:u.join(a['group'],rows[j]['group'])
 for r in rows:r['split_group']=u.root(r['group'])
 groups=collections.defaultdict(list)
 for r in rows:groups[r['split_group']].append(r)
 keys=sorted(groups);random.Random(48017).shuffle(keys)
 test_keys=set(keys[:max(1,len(keys)//5)]);dev_keys=set(keys[len(test_keys):len(test_keys)+max(1,len(keys)//8)])
 # Hold out complete direct-generation topics, across all writer models.
 for b in ['lt25','25-49','50-99','100-199']:
  direct_groups=sorted({r['split_group'] for r in rows if r['origin']=='ai_direct' and r['target_bucket']==b})
  random.Random(48017+len(b)).shuffle(direct_groups)
  test_group,dev_group=direct_groups[:2]
  dev_keys.discard(test_group);test_keys.add(test_group)
  test_keys.discard(dev_group);dev_keys.add(dev_group)
 # Label-stratified group adjustment, never based on classifier predictions.
 for name,chosen in [('test',test_keys),('dev',dev_keys)]:
  minimum=20 if name=='test' else 12
  for label in [False,True]:
   while sum(r['slop']==label for k in chosen for r in groups[k])<minimum:
    candidates=[k for k in keys if k not in test_keys|dev_keys and any(r['slop']==label for r in groups[k])]
    if not candidates:raise ValueError('Cannot populate both heldout classes')
    chosen.add(candidates[0])
 datasets={'test':[r for k in keys if k in test_keys for r in groups[k]],'dev':[r for k in keys if k in dev_keys for r in groups[k]],
           'new_train':[r for k in keys if k not in test_keys|dev_keys for r in groups[k]]}
 old=[]
 for path,name in [(ROOT/'data/train.jsonl','round1_training'),(ROOT/'robustness/labeled.jsonl','round1_diagnostic_reused_for_training')]:
  for r in read(path):
   old.append({**r,'hash':digest(r['text']),'id':r.get('id',r['hash']),'group':name+':'+str(r.get('group',r.get('topic',r['hash']))),
               'source':r.get('source',name),'scenario':r.get('scaffold',name),'origin':r.get('origin','legacy_unknown_origin'),
               'tokens':tokens(r['text']),'token_bucket':bucket(r['text']),'word_count':len(r['text'].split()),'split_group':name+':'+str(r.get('group',r.get('topic',r['hash'])))})
 # Retain the old 31-case holdout for regression only; it is never training data.
 old_holdout_hashes={digest(r['text']) for r in read(ROOT/'data/test.jsonl')}
 old_hashes={r['hash'] for r in old}
 assert not old_hashes & old_holdout_hashes
 assert not any(r['hash'] in old_hashes|old_holdout_hashes for split in datasets.values() for r in split)
 datasets['train_pool']=datasets['new_train']+old
 for name,subset in datasets.items():save(OUT/(name+'.jsonl'),subset)
 summary={name:{'n':len(v),'teacher_slop':sum(r['slop'] for r in v),'groups':len({r['split_group'] for r in v}),'buckets':dict(collections.Counter(r['token_bucket'] for r in v))} for name,v in datasets.items()}
 (OUT/'split_manifest.json').write_text(json.dumps({'seed':48017,'raw_cases':len(labeled),'deduplicated_new':len(rows),'datasets':summary,'old_stress_status':'Reused as training, no longer an independent evaluation set for round 2'},indent=2));print(json.dumps(summary,indent=2))

def mine():
 rows=read(OUT/'train_pool.jsonl');prompt=read_prompt(OUT/'previous_best.py')
 batches=[rows[i:i+8] for i in range(0,len(rows),8)]
 predictions=[]
 with ThreadPoolExecutor(max_workers=6) as pool:
  for i,values in enumerate(pool.map(lambda b:predict_batch(prompt,[r['text'] for r in b]),batches)):
   predictions.extend(values)
   if (i+1)%8==0:print('Mining batches',i+1,'/',len(batches),flush=True)
 enriched=[{**r,'previous_prediction':p,'previous_error':r['slop']!=p} for r,p in zip(rows,predictions)]
 save(OUT/'train_mined.jsonl',enriched)
 # Fixed, stratified hard-example curriculum: same examples for every candidate.
 strata=collections.defaultdict(list)
 for r in enriched:strata[(r['slop'],r['token_bucket'])].append(r)
 rng=random.Random(751);selected=[];used=set()
 for label in [False,True]:
  available=[s for s in strata if s[0]==label];quota=128//len(available)
  for key in sorted(available):
   candidates=strata[key];rng.shuffle(candidates);candidates.sort(key=lambda r:not r['previous_error'])
   for r in candidates[:quota]:selected.append(r);used.add(r['hash'])
  left=[r for r in enriched if r['slop']==label and r['hash'] not in used];rng.shuffle(left);left.sort(key=lambda r:not r['previous_error'])
  missing=128-sum(r['slop']==label for r in selected)
  selected.extend(left[:missing]);used.update(r['hash'] for r in left[:missing])
 rng.shuffle(selected);save(OUT/'search.jsonl',selected)
 summary={'pool_n':len(enriched),'search_n':len(selected),'search_previous_errors':sum(r['previous_error'] for r in selected),'pool_previous_errors':sum(r['previous_error'] for r in enriched),'search_teacher_slop':sum(r['slop'] for r in selected),'search_buckets':dict(collections.Counter(r['token_bucket'] for r in selected)),'hard_human_negatives':sum(r['origin']=='human' and not r['slop'] and r['previous_error'] for r in selected),'selection':'Fixed 256-case class/length-stratified training curriculum, old-prompt errors first. Dev/test never consulted.'}
 (OUT/'mining.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['split','mine']);a=p.parse_args();{'split':split,'mine':mine}[a.action]()
