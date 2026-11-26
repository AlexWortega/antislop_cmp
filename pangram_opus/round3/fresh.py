"""Fresh source/topic-disjoint final test. No labels used by prompt selection."""
import collections,json,os,random,re,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'round2'))
from core import ROOT,MODEL,digest,request,teacher,teacher_label
from data_build import sources,chunks,tokens,bucket,sloplike,nltk
OUT=ROOT/'round3';MODELS=['openai/gpt-5.5','anthropic/claude-sonnet-5.5','google/gemini-3.5-flash','deepseek/deepseek-v4.1-flash','qwen/qwen3.6-plus',MODEL]
BINS=['lt25','25-49','50-99','100-199']
PREFIX='round3'
WANTED=['enron','yelp','brown_government','brown_learned','brown_editorial','inaugural']
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def save(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def grams(text):
 words=re.findall(r'\w+',text.lower());return {tuple(words[i:i+5]) for i in range(len(words)-4)}
def prior():return read(OUT/'train_pool.jsonl')+read(OUT/'dev.jsonl')+read(ROOT/'data/test.jsonl')
def humans():
 path=OUT/'fresh_human.jsonl'
 if path.exists():return read(path)
 old=prior();blocked_groups={r.get('group') for r in old};norms={' '.join(r['text'].lower().split()) for r in old};oldgrams=[grams(r['text']) for r in old]
 pools=collections.defaultdict(list);wanted=WANTED
 for source,group,sentences,meta in sources():
  if source not in wanted or group in blocked_groups:continue
  for text,start in chunks(sentences):
   b=bucket(text)
   if b not in BINS or ' '.join(text.lower().split()) in norms:continue
   pools[(source,b)].append({'text':text,'source':source,'group':group,'source_metadata':meta,'sentence_start':start})
 rng=random.Random(30135);rows=[];used=set();hashes=set()
 for b in BINS:
  for source in wanted:
   candidates=pools[(source,b)];rng.shuffle(candidates);candidates.sort(key=lambda r:-sloplike(r['text']));picked=[]
   for r in candidates:
    if r['group'] in used or digest(r['text']) in hashes:continue
    g=grams(r['text'])
    if len(g)>=8 and any(len(h)>=8 and len(g&h)/min(len(g),len(h))>=.8 for h in oldgrams):continue
    r={**r,'id':digest(['round3-human',r['group'],r['sentence_start']]),'hash':digest(r['text']),'origin':'human','scenario':'authentic_formal_or_formulaic','tokens':tokens(r['text']),'token_bucket':b,'target_bucket':b,'word_count':len(r['text'].split())}
    picked.append(r);used.add(r['group']);hashes.add(r['hash'])
    if len(picked)==2:break
   if len(picked)!=2:raise ValueError('Insufficient fresh humans '+source+' '+b)
   rows.extend(picked)
 save(path,rows);print('Fresh humans',len(rows),flush=True);return rows

TOPICS={
'lt25':[
 ('business','Write one short notice saying the fictional museum coat-check desk closes fifteen minutes before the galleries.'),
 ('review','Write a one-sentence mixed review of a fictional silicone ice tray: easy to release cubes, awkward to carry when full.')],
'25-49':[
 ('technical','Explain in a short self-contained paragraph why a bicycle tyre feels softer on a cold morning, without numbers.'),
 ('academic','Write a short academic-style statement about the difference between a calendar anniversary and a collective memory.')],
'50-99':[
 ('business','Write a practical notice for a fictional seed library: borrowers may take two packets, record the variety, and return seeds after harvest only if they can verify the variety.'),
 ('review','Write a mixed first-person review of a fictional desk lamp with a sturdy clamp, a surprisingly bright lowest setting, and a power button that is hard to find in the dark.')],
'100-199':[
 ('technical','Explain how a float valve keeps a livestock water trough filled, what happens as the water level rises, and two ordinary causes of overflow. Use plain accurate prose.'),
 ('academic','Discuss in a short academic paragraph how preserving an obsolete train timetable can help historians study everyday mobility; mention one limitation of that source without inventing citations.')]
}
RANGES={'lt25':'12–22','25-49':'28–44','50-99':'60–85','100-199':'120–170'}
STYLES={'plain_rewrite':'Rewrite in clear polished English while preserving the facts and approximate length.', 'light_edit':'Make only minimal grammar or punctuation corrections; preserve wording and meaning.', 'antislop_rewrite':'Rewrite in direct concrete English, removing canned transitions and inflated language. Preserve facts and approximate length.'}
def generate(job):
 model,scenario,group,target,task,parent=job;key=digest([model,scenario,group,target,task]);path=OUT/'fresh_generation'/f'{key}.json';path.parent.mkdir(exist_ok=True)
 if path.exists():return json.loads(path.read_text())['row']
 reasoning={'enabled':False} if model.startswith(('deepseek/','qwen/')) else {'effort':'minimal','exclude':True}
 body={'model':model,'max_tokens':2048,'reasoning':reasoning,'messages':[{'role':'system','content':'Return only the requested English text. Do not add a preface or word/token count.'},{'role':'user','content':task}]}
 response=request('https://openrouter.ai/api/v1/chat/completions',os.environ['OPENROUTER_API_KEY'],body);choice=response['choices'][0]
 text=(choice['message'].get('content') or '').strip()
 if choice['finish_reason']!='stop' or not text:raise ValueError('Incomplete fresh generation '+str(choice['finish_reason']))
 row={'id':key,'hash':digest(text),'text':text,'group':group,'source':parent['source'] if parent else 'fresh_direct_generated','origin':'ai_rewrite' if parent else 'ai_direct','scenario':scenario,'model':model,'tokens':tokens(text),'token_bucket':bucket(text),'target_bucket':target,'word_count':len(text.split()),'parent_id':parent['id'] if parent else None,'task':task if parent is None else STYLES[scenario]}
 path.write_text(json.dumps({'request':body,'response':response,'row':row},ensure_ascii=False));print('Fresh generation',model,scenario,row['token_bucket'],flush=True);return row

def build():
 human=humans();jobs=[]
 for model in MODELS:
  for b in BINS:
   for genre,task in TOPICS[b]:jobs.append((model,'direct_'+genre,PREFIX+'-direct:'+b+':'+genre,b,task+' Aim for approximately '+RANGES[b]+' English tokens.',None))
 for i,model in enumerate(MODELS):
  for b in BINS:
   parent=[r for r in human if r['token_bucket']==b][i];style=list(STYLES)[i%3]
   jobs.append((model,style,parent['group'],b,STYLES[style]+'\n\n'+parent['text'],parent))
 with ThreadPoolExecutor(max_workers=6) as pool:generated=list(pool.map(generate,jobs))
 formats=[]
 for b in BINS:
  for parent in [r for r in human if r['token_bucket']==b][-3:]:
   text='## Notes\n\n- '+parent['text']
   formats.append({**parent,'id':digest(['format',parent['id']]),'hash':digest(text),'text':text,'origin':'human_format_only','scenario':'markdown_only','tokens':tokens(text),'token_bucket':bucket(text),'parent_id':parent['id'],'word_count':len(text.split())})
 mixes=[];by_id={r['id']:r for r in human}
 for b in BINS:
  rewrites=[r for r in generated if r['origin']=='ai_rewrite' and r['target_bucket']==b][:3]
  for i,r in enumerate(rewrites):
   parent=by_id[r['parent_id']];original=nltk.sent_tokenize(parent['text']);replacement=nltk.sent_tokenize(r['text'])[0]
   text=' '.join(original[:-1]+[replacement]) if i%2 and len(original)>1 else parent['text']+'\n'+replacement
   mixes.append({**parent,'id':digest(['mixed',parent['id'],r['id']]),'hash':digest(text),'text':text,'origin':'mixed','scenario':'human_plus_ai_sentence','model':r['model'],'parent_id':parent['id'],'ai_parent_id':r['id'],'tokens':tokens(text),'token_bucket':bucket(text),'word_count':len(text.split())})
 rows=human+generated+formats+mixes;old=prior();oldnorm={' '.join(r['text'].lower().split()) for r in old};oldgrams=[grams(r['text']) for r in old]
 for r in rows:
  assert ' '.join(r['text'].lower().split()) not in oldnorm
  g=grams(r['text'])
  assert not (len(g)>=8 and any(len(h)>=8 and len(g&h)/min(len(g),len(h))>=.8 for h in oldgrams)),r['id']
  r['split_group']=r['group']
 dedup={}
 for r in rows:
  if r['hash'] in dedup:dedup[r['hash']].setdefault('identical_variant_ids',[]).append(r['id'])
  else:dedup[r['hash']]=r
 rows=list(dedup.values());save(OUT/'fresh_unlabeled.jsonl',rows)
 (OUT/'fresh_audit.json').write_text(json.dumps({'n':len(rows),'raw_n':len(human)+len(generated)+len(formats)+len(mixes),'groups':len({r['group'] for r in rows}),'old_rows':len(old),'normalized_and_near_duplicates':0,'origin_counts':dict(collections.Counter(r['origin'] for r in rows)),'length_counts':dict(collections.Counter(r['token_bucket'] for r in rows))},indent=2));print('Fresh test ready, UNLABELED',len(rows),flush=True)

def label():
 assert (OUT/'selection.json').exists(),'Freeze prompt before teacher labeling fresh test'
 rows=read(OUT/'fresh_unlabeled.jsonl');labeled=[]
 def one(r):
  t=teacher(r['text']);return {**r,'slop':teacher_label(t),'teacher_version':t['version'],'teacher_status':'ok'}
 with ThreadPoolExecutor(max_workers=6) as pool:
  for r in pool.map(one,rows):labeled.append(r);print('Labeled fresh',len(labeled),flush=True)
 save(OUT/'test.jsonl',labeled)
if __name__=='__main__':{'build':build,'label':label}[sys.argv[1]]()
