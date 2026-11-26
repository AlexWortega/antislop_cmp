import collections,json,random,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,read_prompt,digest
from round2.data_build import tokens,bucket as token_bucket
OUT=ROOT/'round5';PRE=ROOT/'round4'
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def save(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def bucket(r):return '200plus' if r['tokens']>=200 else r['token_bucket']
if __name__=='__main__':
 assert json.loads((PRE/'pipeline_status.json').read_text())['status']=='complete'
 for name in ['best_candidate.py','best_prompt.txt','request_template.json','inference_config.json','README.md','results.json','manifest.json']:
  target=OUT/('previous_'+name)
  if not target.exists():shutil.copyfile(ROOT/name,target)
 shutil.copyfile(PRE/'goal_criteria.json',OUT/'goal_criteria.json')
 previous=read(PRE/'test.jsonl');pred=json.loads((PRE/'test_results.json').read_text())['selected']['predictions'];support_groups={r['split_group'] for r,p in zip(previous,pred) if r['slop']!=p['prediction']};support=[r for r in previous if r['split_group'] in support_groups];save(OUT/'support.jsonl',support)
 oldrefs=read(ROOT/'round3/support.jsonl')+read(ROOT/'round3/extended_support.jsonl');excluded=support_groups|{r['split_group'] for r in oldrefs}
 prior=read(PRE/'prior.jsonl')+previous
 for r in prior:
  r.setdefault('split_group',r['group']);r.setdefault('tokens',tokens(r['text']));r.setdefault('token_bucket',token_bucket(r['text']));r.setdefault('word_count',len(r['text'].split()))
 save(OUT/'prior.jsonl',prior)
 unique={r['hash']:r for r in prior};pool=[r for r in unique.values() if r['split_group'] not in excluded];rng=random.Random(51190);groups=sorted({r['split_group'] for r in pool});rng.shuffle(groups);devgroups=set(groups[:max(1,len(groups)//4)])
 def stratified(values,per_class,large):
  rng.shuffle(values);out=[]
  for b in ['lt25','25-49','50-99','100-199','200plus']:
   for label in [False,True]:out += [r for r in values if bucket(r)==b and r['slop']==label][:(large if b=='200plus' else per_class)]
  rng.shuffle(out);return out
 train=stratified([r for r in pool if r['split_group'] not in devgroups],20,8);dev=stratified([r for r in pool if r['split_group'] in devgroups],10,4)
 save(OUT/'train.jsonl',train);save(OUT/'dev.jsonl',dev)
 assert not ({r['split_group'] for r in train}&{r['split_group'] for r in dev})
 prompt=read_prompt(OUT/'previous_best_candidate.py');tail='\n\nMATCHED TEACHER BOUNDARY REFERENCES\nThese are labeled calibration references, not the input to classify. Their observed Pangram labels take priority over stylistic stereotypes in the rulebook. Infer a boundary from the contrasting language; do not use subject names, source attribution, or assumed editing history. Some highly fluent or actually AI-authored passages receive false. A polished sentence, an explanation, or a local rewrite is not automatically positive. Conversely brevity and a conventional notice genre are not automatic negatives. Compare the positive and negative evidence in the supplied text.\n'
 grouped=collections.defaultdict(list)
 for r in support:grouped[r['split_group']].append({'text':r['text'],'slop':r['slop']})
 for i,group in enumerate(sorted(grouped)):tail+='\nReference group '+str(i+1)+':\n'+json.dumps(grouped[group],ensure_ascii=False)+'\n'
 prompt+=tail;assert len(prompt)<=60000
 (OUT/'initial.py').write_text('# EVOLVE-BLOCK-START\nPROMPT = """'+prompt.replace('\\','\\\\').replace('"""','\\"\\"\\"')+'"""\n# EVOLVE-BLOCK-END\n')
 suffix=(ROOT/'round4/reference_suffix.txt').read_text()+tail
 assert suffix in prompt
 (OUT/'reference_suffix.txt').write_text(suffix)
 (OUT/'protocol.json').write_text(json.dumps({'round':5,'train_n':len(train),'dev_n':len(dev),'new_support_n':len(support),'new_support_groups':len(support_groups),'prior_rows':len(prior),'mode':'default','generations':3,'selection':'Choose among all valid evolved candidates and previous baseline using fixed reused calibration dev length-balanced objective; only new final test is independent.','support_exclusion':'All original and added reference source groups excluded from scored train/dev.','warning':'Round4 test has been promoted to calibration. Historical calibration dev is not an independent generalization estimate.','fresh_outcomes_used':False,'initial_hash':digest(prompt),'goal_criteria':'unchanged from round3','fresh_human_sources':['enron','yelp','brown_news','brown_learned','brown_hobbies','inaugural'],'source_change_reason':'Insufficient unused short government/review documents. Source choice made before teacher labeling or classifier evaluation; news and practical human prose extend scenario coverage.','budget':'same cumulative $100 per provider; no reset','preservation_controls':len(train)},indent=2))
 if (OUT/'dev_extension.jsonl').exists():
  extra=read(OUT/'dev_extension.jsonl');dev+=extra;prior+=extra;save(OUT/'dev.jsonl',dev);save(OUT/'prior.jsonl',prior)
  protocol=json.loads((OUT/'protocol.json').read_text());protocol.update(dev_n=len(dev),prior_rows=len(prior),dev_extension={'n':len(extra),'positive':sum(r['slop'] for r in extra),'reason':'Fixed new 25–49-token calibration jobs across six models and two topics, added before dev scoring to repair sparse positive coverage; both classes retained.','final_test_unseen':True});(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 print('Train',len(train),'dev',len(dev),'added reference cases',len(support),'prompt chars',len(prompt),flush=True)
