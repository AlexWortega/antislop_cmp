import argparse
import collections
import json
import os
import random
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,request,digest,teacher,teacher_label
from data_build import OUT,tokens,bucket,nltk

MODELS=['openai/gpt-5.5','anthropic/claude-sonnet-5.5','google/gemini-3.5-flash','deepseek/deepseek-v4.1-flash','qwen/qwen3.6-plus','anthropic/claude-opus-5.5']
STYLES={'plain_rewrite':'Rewrite in clear, polished standard English. Preserve all facts and meaning.',
        'conversational_rewrite':'Rewrite in natural conversational English, with a particular human voice and varied rhythm. Preserve facts. Do not add invented memories, errors or new facts.',
        'antislop_rewrite':'Rewrite in direct, concrete English. Remove generic transitions, inflated adjectives and canned conclusions. Preserve the facts and avoid unnecessary polishing.',
        'light_edit':'Make only small grammar or punctuation corrections. Preserve wording and structure as much as possible. Do not add content.'}
BINS=['lt25','25-49','50-99','100-199']

def read(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]
def save(path,rows):Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def generation(job):
 model,style,parent=job
 path=OUT/'generated'/(digest([model,style,parent['id']])+'.json');path.parent.mkdir(exist_ok=True)
 if path.exists():return json.loads(path.read_text())
 reasoning={'enabled':False} if model.startswith(('qwen/','deepseek/')) else {'effort':'minimal','exclude':True}
 body={'model':model,'max_tokens':2048,'reasoning':reasoning,
       'messages':[{'role':'system','content':STYLES[style]+' Keep approximately the same length as the source. Return only the rewritten text, without an introduction or quotation wrapper.'},
                   {'role':'user','content':parent['text']}]}
 rp=OUT/'generation_responses'/(digest(body)+'.json');rp.parent.mkdir(exist_ok=True)
 if rp.exists():result=json.loads(rp.read_text())
 else:
  result=request('https://openrouter.ai/api/v1/chat/completions',os.environ['OPENROUTER_API_KEY'],body)
  rp.write_text(json.dumps({'request':body,'response':result},ensure_ascii=False))
 if 'response' in result:result=result['response']
 choice=result['choices'][0];text=(choice['message'].get('content') or '').strip()
 if choice.get('finish_reason')!='stop' or not text:raise ValueError('Incomplete generation '+model+' '+style)
 row={'id':digest([model,style,parent['id']]),'hash':digest(' '.join(text.split())),'text':text,'group':parent['group'],'parent_id':parent['id'],'source':parent['source'],'origin':'ai_rewrite','scenario':style,'model':model,'target_bucket':parent['token_bucket'],'tokens':tokens(text),'token_bucket':bucket(text),'word_count':len(text.split())}
 path.write_text(json.dumps(row,ensure_ascii=False,indent=2));print('Generated',model,style,row['token_bucket'],flush=True)
 return row

def build():
 humans=read(OUT/'human.jsonl');rng=random.Random(117)
 by_bucket={b:[r for r in humans if r['token_bucket']==b] for b in BINS}
 for rows in by_bucket.values():rng.shuffle(rows)
 used=set();jobs=[]
 for model in MODELS:
  for style in STYLES:
   for b in BINS:
    parent=next(r for r in by_bucket[b] if r['id'] not in used)
    used.add(parent['id']);jobs.append((model,style,parent))
 rng.shuffle(jobs)
 with ThreadPoolExecutor(max_workers=12) as pool:generated=list(pool.map(generation,jobs))
 parents={r['id']:r for r in humans};mixed=[]
 for r in generated:
  parent=parents[r['parent_id']]
  a=nltk.sent_tokenize(parent['text']);b=nltk.sent_tokenize(r['text'])
  if len(a)<2 or len(b)<2:continue
  text=' '.join(a[:max(1,len(a)//2)]+b[max(1,len(b)//2):])
  if text==parent['text'] or text==r['text']:continue
  mixed.append({**r,'id':digest(['mixed',r['id']]),'hash':digest(' '.join(text.split())),'text':text,'origin':'mixed','scenario':'human_ai_sentence_mix','tokens':tokens(text),'token_bucket':bucket(text),'word_count':len(text.split())})
  if len(mixed)==32:break
 formatted=[]
 for r in sorted(humans,key=lambda r:r['id']):
  if r['token_bucket'] not in ['50-99','100-199']:continue
  text='## Notes\n\n'+'\n'.join('- '+s for s in nltk.sent_tokenize(r['text']))
  formatted.append({**r,'id':digest(['formatted',r['id']]),'hash':digest(' '.join(text.split())),'text':text,'origin':'human_format_only','scenario':'markdown_format_only','tokens':tokens(text),'token_bucket':bucket(text),'word_count':len(text.split())})
  if len(formatted)==32:break
 all_rows=humans+generated+mixed+formatted
 save(OUT/'unlabeled.jsonl',all_rows)
 protocol=json.loads((OUT/'data_protocol.json').read_text());protocol.update(generators=MODELS,styles=STYLES,raw_cases=len(all_rows),origin_counts=dict(collections.Counter(r['origin'] for r in all_rows)),short_probes='Pangram API accepted 1, 7, 18 and 26 word probes, despite 50-word intended-use recommendation')
 (OUT/'data_protocol.json').write_text(json.dumps(protocol,indent=2));print(protocol,flush=True)

def label(path):
 rows=read(path)
 def work(row):
  try:
   result=teacher(row['text'])
   return {**row,'slop':teacher_label(result),'teacher_version':result['version'],'teacher_prediction_short':result['prediction_short'],'below_recommended_words':row['word_count']<50,'teacher_status':'ok'}
  except urllib.error.HTTPError as e:return {**row,'teacher_status':'rejected','http_status':e.code,'detail':e.read().decode()[:500]}
  except Exception as e:return {**row,'teacher_status':'failed','detail':str(e)[:500]}
 with ThreadPoolExecutor(max_workers=int(os.environ.get('TEACHER_WORKERS','4'))) as pool:
  results=[]
  for i,row in enumerate(pool.map(work,rows)):
   results.append(row)
   if (i+1)%20==0:print('Labeled',i+1,'/',len(rows),flush=True)
 destination = 'labeled.jsonl' if Path(path).name=='unlabeled.jsonl' else Path(path).stem+'_labeled.jsonl'
 save(OUT/destination,results)
 print('Status',dict(collections.Counter(r['teacher_status'] for r in results)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','label','label-human']);a=p.parse_args()
 if a.action=='build':build()
 else:label(OUT/('human.jsonl' if a.action=='label-human' else 'unlabeled.jsonl'))
