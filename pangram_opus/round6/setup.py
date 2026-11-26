"""Prepare the next calibration split offline; prior test is now learning data."""
import collections,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,digest,teacher_label,read_prompt
OUT=ROOT/'round6'
def read(p):
 rows=[json.loads(l) for l in p.read_text().splitlines()]
 for r in rows:
  actual=digest(r['text'])
  if r['hash']!=actual:r['legacy_hash']=r['hash'];r['hash']=actual
 return rows
def write(name,rows):(OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
if __name__=='__main__':
 assert not (OUT/'test.jsonl').exists(),'Do not recreate calibration after new outcomes'
 old=ROOT/'round5';train=read(old/'expanded/train.jsonl')+read(old/'test.jsonl');dev=read(old/'expanded/dev.jsonl')
 assert len(train)==622 and len({r['hash'] for r in train})==622
 assert not {r['split_group'] for r in train}&{r['split_group'] for r in dev}
 assert not {r['hash'] for r in train}&{r['hash'] for r in dev}
 refs=[r for name in ['support.jsonl','extended_support.jsonl'] for r in read(ROOT/'round3'/name)]
 assert not {r['split_group'] for r in refs}&{r['split_group'] for r in train+dev}
 for r in train+dev:
  assert r['hash']==digest(r['text'])
  assert r['slop']==teacher_label(json.loads((ROOT/'cache'/('pangram4_'+r['hash']+'.json')).read_text()))
 prior=list({r['hash']:r for r in read(old/'prior.jsonl')+read(old/'test.jsonl')}.values())
 write('train.jsonl',train);write('dev.jsonl',dev);write('prior.jsonl',prior)
 shutil.copyfile(ROOT/'best_candidate.py',OUT/'initial.py');shutil.copyfile(old/'goal_criteria.json',OUT/'goal_criteria.json')
 report={'status':'prepared_offline_not_running','train_n':len(train),'dev_n':len(dev),'prior_n':len(prior),'train_groups':len({r['split_group'] for r in train}),'dev_groups':len({r['split_group'] for r in dev}),'canonicalized_legacy_hashes':sum('legacy_hash' in r for r in train+dev),'teacher_label_mismatches':0,'group_overlap':0,'reference_group_overlap':0,'initial_prompt_hash':digest(read_prompt(OUT/'initial.py')),'data_policy':'Round5 final142 is disclosed learning data here; never call it independent for a round6 candidate. Reused dev112 is adaptive calibration only. Next final test needs new source/topic groups outside prior1264.','train_lengths':dict(collections.Counter(r['token_bucket'] for r in train)),'train_classes':dict(collections.Counter(str(r['slop']) for r in train)),'goal_thresholds_unchanged':True,'api_calls':0,'spending_policy':'Existing budgets remain $100 per provider. Requested additional Pangram allowance is not authorization; no increased spending until user reply.'}
 (OUT/'protocol.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
