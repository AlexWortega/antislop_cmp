"""Pre-LLM human excerpts selected before detector labels; token counts are o200k_base."""
import collections
import json
import random
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,digest
import nltk
import tiktoken
from nltk.tokenize.treebank import TreebankWordDetokenizer

OUT=ROOT/'round2'; nltk.data.path.insert(0,str(OUT/'nltk_data'))
ENC=tiktoken.get_encoding('o200k_base'); DETOK=TreebankWordDetokenizer()
PHRASES=['furthermore','moreover','therefore','in conclusion','in summary','it is important','significant','essential','crucial','comprehensive','understanding','development','effective','ensure','various','not only','in addition','however','role','consider','in order to','important','benefits','challenge']

def tokens(text):return len(ENC.encode(text))
def bucket(text):
 n=tokens(text)
 return 'lt25' if n<25 else '25-49' if n<50 else '50-99' if n<100 else '100-199' if n<200 else '200-349' if n<350 else '350plus'
def sloplike(text):
 low=text.lower();words=text.split()
 return sum(low.count(x) for x in PHRASES)*2+sum(len(w)>10 for w in words)/max(1,len(words))*15

def hf_rows(dataset,config,offset):
 path=OUT/'source_cache'/(digest([dataset,config,offset])+'.json');path.parent.mkdir(exist_ok=True)
 if path.exists():r=json.loads(path.read_text())
 else:
  url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode({'dataset':dataset,'config':config,'split':'train','offset':offset,'length':100})
  with urllib.request.urlopen(url,timeout=120) as response:r=json.load(response)
  path.write_text(json.dumps(r))
 return [(x['row_idx'],x['row']) for x in r['rows']]

def chunks(sentences):
 for start in range(0,len(sentences),max(1,len(sentences)//30)):
  for lower,upper in [(8,24),(25,49),(50,99),(100,199),(200,349)]:
   acc=[]
   for sentence in sentences[start:start+16]:
    acc.append(sentence);text=' '.join(acc).strip();n=tokens(text)
    if n>upper:break
    if n>=lower and len(text.split())>=3:
     yield text,start;break

def sources():
 for category in ['government','learned','editorial','reviews','hobbies','news']:
  for file in nltk.corpus.brown.fileids(categories=category):
   sentences=[DETOK.detokenize(s).replace('``','"').replace("''",'"') for s in nltk.corpus.brown.sents(file)]
   yield 'brown_'+category,'brown:'+file,sentences,{'corpus':'Brown 1961','license':'non-commercial research','url':'https://www.nltk.org/book/ch02.html'}
 for file in nltk.corpus.inaugural.fileids():
  if int(file[:4])>2009:continue
  sentences=nltk.sent_tokenize(nltk.corpus.inaugural.raw(file))
  yield 'inaugural','inaugural:'+file,sentences,{'corpus':'US inaugural addresses through 2009','license':'public domain','url':'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/inaugural.xml'}
 for source,dataset,config,offsets in [('enron','corbt/enron-emails','default',[21000,41000,71000,111000,151000,191000]),('yelp','Yelp/yelp_review_full','yelp_review_full',[12000,22000,42000])]:
  for offset in offsets:
   for idx,row in hf_rows(dataset,config,offset):
    text=row['body'] if source=='enron' else row['text'].replace('\\n','\n').replace('\\"','"')
    if source=='enron' and any(x.lower() in text.lower() for x in ['Original Message','Forwarded','-----']):continue
    if not 5<=len(text.split())<=1600:continue
    text=re.sub(r'\s+',' ',text).strip()
    yield source,f'{source}:{idx}',nltk.sent_tokenize(text),{'corpus':dataset,'url':'https://huggingface.co/datasets/'+dataset,'row_index':idx,'provenance':'pre-ChatGPT corpus, human-origin reference'}

def main():
 OUT.mkdir(exist_ok=True)
 blocked=[]
 for path in [ROOT/'data/train.jsonl',ROOT/'data/test.jsonl',ROOT/'robustness/labeled.jsonl']:
  blocked.extend(' '.join(json.loads(l)['text'].split()).lower() for l in path.read_text().splitlines())
 by_source=collections.defaultdict(list);seen=set()
 for source,group,sents,meta in sources():
  for text,start in chunks(sents):
   norm=' '.join(text.split()).lower();h=digest(norm)
   if h in seen or any(norm in old or old in norm for old in blocked):continue
   if len(re.findall(r'[A-Za-z]',text))/max(1,len(text))<0.6:continue
   seen.add(h)
   by_source[source].append({'id':h,'hash':digest(' '.join(text.split())),'text':text,'group':group,'source':source,'origin':'human','scenario':'human_sloplike','tokens':tokens(text),'token_bucket':bucket(text),'word_count':len(text.split()),'sloplike_score':sloplike(text),'sentence_start':start,'provenance':meta})
 quotas={'enron':96,'yelp':64,'brown_government':24,'brown_learned':32,'brown_editorial':16,'brown_reviews':16,'brown_hobbies':16,'brown_news':16,'inaugural':40}
 chosen=[];rng=random.Random(20261101)
 for source,n in quotas.items():
  candidates=by_source[source];rng.shuffle(candidates);candidates.sort(key=lambda r:-r['sloplike_score'])
  selected=[];group_count=collections.Counter();used=set()
  cap=2 if source.startswith(('brown','inaugural')) else 1
  for label,target in [('lt25',n*15//100),('25-49',n*15//100),('50-99',n//4),('100-199',n*35//100),('200-349',n-(n*15//100)*2-n//4-n*35//100)]:
   bucket_count=0
   for r in candidates:
    if r['token_bucket']!=label or r['id'] in used or group_count[r['group']]>=cap:continue
    selected.append(r);used.add(r['id']);group_count[r['group']]+=1;bucket_count+=1
    if bucket_count>=target:break
  for r in candidates:
   if len(selected)>=n:break
   if r['id'] not in used and group_count[r['group']]<cap:
    selected.append(r);used.add(r['id']);group_count[r['group']]+=1
  if len(selected)<n:raise RuntimeError(f'Not enough {source}: {len(selected)} / {n}')
  chosen.extend(selected[:n]);print(source,len(selected[:n]),flush=True)
 (OUT/'human.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen))
 protocol={'tokenizer':'o200k_base (explicit proxy, not proprietary Opus tokenizer)','human_n':len(chosen),'selection':'Source quotas, corpus-authentic contiguous sentence excerpts, lexical/formality enrichment before any detector labels; no LLM-authored human controls','buckets':dict(collections.Counter(r['token_bucket'] for r in chosen)),'seed':20261101,'pangram_recommended_min_words':50,'short_input_handling':'API probe accepts 26 words; shorter accepted outputs included as out-of-scope teacher-imitation cases, rejections recorded without labels','sources':quotas}
 (OUT/'data_protocol.json').write_text(json.dumps(protocol,indent=2));print(protocol,flush=True)

if __name__=='__main__':main()
