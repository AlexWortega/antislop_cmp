import collections,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,digest
OUT=ROOT/'round3'
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def save(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
if __name__=='__main__':
 rows=read(ROOT/'round2/train_pool.jsonl')+read(ROOT/'round2/test.jsonl')
 rows=list({r['hash']:r for r in rows}.values());dev=read(ROOT/'round2/dev.jsonl')
 assert not {r['hash'] for r in rows}&{r['hash'] for r in dev}
 assert not {r['split_group'] for r in rows}&{r['split_group'] for r in dev}
 save(OUT/'train_pool.jsonl',rows);save(OUT/'dev.jsonl',dev)
 rng=random.Random(3018);selected=[];used=set()
 groups=collections.defaultdict(list)
 for r in rows:groups[r['split_group']].append(r)
 support=[];support_groups=set()
 for b in ['lt25','25-49','50-99','100-199','200-349']:
  options=[]
  for group,items in groups.items():
   if group in support_groups:continue
   neg=[r for r in items if r['token_bucket']==b and not r['slop']];pos=[r for r in items if r['token_bucket']==b and r['slop']]
   for a in neg:
    for z in pos:options.append((abs(a['tokens']-z['tokens'])+0.1*(a['tokens']+z['tokens']),group,a,z))
  _,group,a,z=min(options,key=lambda x:(x[0],x[1]));support.extend([a,z]);support_groups.add(group)
 save(OUT/'support.jsonl',support)
 eligible=[r for r in rows if r['split_group'] not in support_groups]
 # Fixed 128 cases, equal teacher classes, length strata; all available tiny positives.
 quotas={'lt25':16,'25-49':12,'50-99':12,'100-199':12,'200plus':12}
 def bucket(r):return '200plus' if r['tokens']>=200 else r['token_bucket']
 for label in [False,True]:
  for b,n in quotas.items():
   choices=[r for r in eligible if r['slop']==label and bucket(r)==b];rng.shuffle(choices)
   # prioritize legacy failure scenarios rather than sample-specific prior predictions
   choices.sort(key=lambda r: r['origin'] not in ['human_format_only','mixed','ai_rewrite','ai_direct'])
   anchors=[r for r in choices if r['origin']=='human'][:4] if not label else []
   anchor_hashes={r['hash'] for r in anchors};chosen=anchors+[r for r in choices if r['hash'] not in anchor_hashes][:n-len(anchors)]
   selected.extend(chosen);used.update(r['hash'] for r in chosen)
  remaining=[r for r in eligible if r['slop']==label and r['hash'] not in used];rng.shuffle(remaining)
  more=remaining[:64-sum(r['slop']==label for r in selected)];selected.extend(more);used.update(r['hash'] for r in more)
 rng.shuffle(selected);save(OUT/'search.jsonl',selected)
 screen=[]
 for label in [False,True]:
  for b in quotas:
   choices=[r for r in selected if r['slop']==label and bucket(r)==b]
   screen.extend(choices[:4 if b=='lt25' else 3])
 rng.shuffle(screen);save(OUT/'screen.jsonl',screen)
 protocol={'train_n':len(rows),'search_n':len(selected),'screen_n':len(screen),'dev_n':len(dev),'support_n':len(support),'support_groups':sorted(support_groups),'support_rule':'Five matched positive/negative Pangram pairs appended as in-context references to advisor seeds. Whole support groups excluded from search and screening; support never comes from dev or fresh test.','seed':3018,'search_class_length':dict(collections.Counter(str(r['slop'])+':'+bucket(r) for r in selected)), 'inference':'strict single text, same core.predict settings for every stage','objective':'0.5 overall balanced accuracy +0.5 mean per-length balanced accuracy on strata with both classes; provider refusal counts as error','selection':'Screen three advisor proposals and previous prompt on fixed training screen; best starts Shinka. After eight attempted generations compare previous prompt and best training candidate on reused dev. Freeze winner before fresh test. Old test is now training, no longer held out.','session_limits':{'pangram':100,'openrouter':100},'reference':'reference.json'}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print(protocol)
