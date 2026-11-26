import collections,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from core import ROOT,read_prompt,digest
OUT=ROOT/'round5/expanded';ROUND=ROOT/'round5'
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def save(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def bucket(r):return '200plus' if r['tokens']>=200 else r['token_bucket']
if __name__=='__main__':
 assert not (ROUND/'test.jsonl').exists()
 old_train=read(ROUND/'train.jsonl');old_dev=read(ROUND/'dev.jsonl');old=old_train+old_dev;oldgroups={r['split_group'] for r in old_dev};refs=read(ROOT/'round3/support.jsonl')+read(ROOT/'round3/extended_support.jsonl');refgroups={r['split_group'] for r in refs};refhash={r['hash'] for r in refs};unique={r['hash']:r for r in read(ROUND/'prior.jsonl')};pool=[r for r in unique.values() if r['split_group'] not in refgroups and r['hash'] not in refhash]
 assert all(r['split_group'] not in refgroups for r in old)
 groups=collections.defaultdict(list)
 for r in pool:
  if r['split_group'] not in oldgroups:groups[r['split_group']].append(r)
 rng=random.Random(58217);keys=sorted(groups);rng.shuffle(keys);chosen=set();strata=[(b,label) for b in ['lt25','25-49','50-99','100-199','200plus'] for label in [True,False]]
 for b,label in strata:
  target=8 if label or b=='200plus' else 16
  while sum(bucket(r)==b and r['slop']==label for g in chosen for r in groups[g])<target:
   candidates=[g for g in keys if g not in chosen and any(bucket(r)==b and r['slop']==label for r in groups[g])]
   if not candidates:break
   chosen.add(candidates[0])
 devpool=[r for g in sorted(chosen) for r in groups[g]];rng.shuffle(devpool);dev=[]
 for b,label in strata:dev += [r for r in devpool if bucket(r)==b and r['slop']==label][:(8 if label or b=='200plus' else 16)]
 train=list({r['hash']:r for r in old if r['split_group'] not in chosen}.values());existing={r['hash'] for r in train};remaining=[r for r in pool if r['split_group'] not in chosen and r['hash'] not in existing];rng.shuffle(remaining)
 while remaining and len(train)<480:
  counts=collections.Counter((bucket(r),r['slop']) for r in train);available=collections.defaultdict(list)
  for i,r in enumerate(remaining):available[(bucket(r),r['slop'])].append(i)
  key=min(available,key=lambda k:(counts[k],str(k)));train.append(remaining.pop(available[key][0]))
 rng.shuffle(train);rng.shuffle(dev)
 assert not ({r['split_group'] for r in train}&{r['split_group'] for r in dev});assert not ({r['hash'] for r in train}&{r['hash'] for r in dev})
 save(OUT/'train.jsonl',train);save(OUT/'dev.jsonl',dev)
 prompt=read_prompt(ROUND/'previous_best_candidate.py');(OUT/'initial.py').write_text('# EVOLVE-BLOCK-START\nPROMPT = """'+prompt.replace('\\','\\\\').replace('"""','\\"\\"\\"')+'"""\n# EVOLVE-BLOCK-END\n');(OUT/'reference_suffix.txt').write_text((ROOT/'round4/reference_suffix.txt').read_text())
 (OUT/'protocol.json').write_text(json.dumps({'train_n':len(train),'dev_n':len(dev),'promoted_old_calibration_n':len(old_dev),'train_groups':len({r['split_group'] for r in train}),'dev_groups':len(chosen),'rule':'All former 86 calibration-dev cases are now learning data. Historical train cases may be reallocated by entire source group to ensure adequate positive coverage in the new calibration dev. Fixed reference groups excluded from both.','warning':'These are historical calibration data; only the untouched 142 final texts are independent. Do not compare percentages across changed calibration splits.','generations':2,'mode':'default','selection':'new calibration dev length-balanced objective; require gain >=0.005, and no regression in agreement/BA versus previous working prompt','fresh_outcomes_used':False,'initial_hash':digest(prompt),'teacher_calls':0},indent=2));print('Expanded train',len(train),'new calibration dev',len(dev),flush=True)
