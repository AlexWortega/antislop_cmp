import json,random,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,read_prompt,digest
OUT=ROOT/'round4';PRE=ROOT/'round3'
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def save(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
if __name__=='__main__':
 assert json.loads((PRE/'pipeline_status.json').read_text())['status']=='complete'
 for name in ['best_candidate.py','best_prompt.txt','request_template.json','inference_config.json','README.md','results.json','manifest.json']:
  target=OUT/('previous_'+name)
  if not target.exists():shutil.copyfile(ROOT/name,target)
 shutil.copyfile(PRE/'goal_criteria.json',OUT/'goal_criteria.json')
 old=read(PRE/'test.jsonl');pool=read(PRE/'fine_train.jsonl');rng=random.Random(40119);rng.shuffle(pool)
 old_hashes={r['hash'] for r in old};controls=[]
 for label in [False,True]:
  for b in ['lt25','25-49','50-99','100-199']:
   controls.extend([r for r in pool if r['slop']==label and r['token_bucket']==b and r['hash'] not in old_hashes][:3])
 rows=old+controls;save(OUT/'train.jsonl',rows)
 prompt=read_prompt(OUT/'previous_best_candidate.py');(OUT/'initial.py').write_text('# EVOLVE-BLOCK-START\nPROMPT = """'+prompt.replace('\\','\\\\').replace('"""','\\"\\"\\"')+'"""\n# EVOLVE-BLOCK-END\n')
 suffix=(PRE/'fine_reference_suffix.txt').read_text();assert suffix in prompt;(OUT/'reference_suffix.txt').write_text(suffix)
 allprior=read(PRE/'train_pool.jsonl')+read(PRE/'dev.jsonl')+read(ROOT/'data/test.jsonl')+old;save(OUT/'prior.jsonl',allprior)
 (OUT/'protocol.json').write_text(json.dumps({'round':4,'train_n':len(rows),'promoted_round3_test_n':len(old),'preservation_controls':len(controls),'mode':'default','generations':3,'selection':'length-balanced objective on calibration train only; final test source/topic-disjoint and unseen','warning':'Round3 test is now training, not an independent score for round4. No added examples or exact sample wording permitted in mutations.','fresh_outcomes_used':False,'initial_hash':digest(prompt),'goal_criteria':'unchanged from round3','budget':'same cumulative $100 per provider; no reset'},indent=2))
 print('Round4 train',len(rows),'prior',len(allprior))
