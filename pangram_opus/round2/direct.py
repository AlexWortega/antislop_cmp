import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from expand import OUT,MODELS,read,save
from data_build import tokens,bucket
from core import request,digest

TASKS={
 'lt25':[
  ('business','Write a brief, courteous reminder to a teammate to update the shared task list before tomorrow morning.'),
  ('technical','Explain why a bicycle tire feels softer on a cold morning.'),
  ('review','Write a fictional one-sentence review of an umbrella that is light but difficult to fold.'),
  ('academic','State a concise conclusion for a fictional classroom experiment about paper airplane wing width.')],
 '25-49':[
  ('business','Write a short status update: the catalog photos are ready, two product descriptions still need approval, and launch is planned for Tuesday.'),
  ('technical','Explain what a browser cache does using a simple everyday comparison.'),
  ('review','Write a fictional customer review of a small desk lamp that has a useful dimmer but a short power cable.'),
  ('academic','Write a concise abstract for a fictional student project comparing how quickly two types of paper absorb water.')],
 '50-99':[
  ('business','Write a notice explaining that a fictional neighborhood repair cafe will move its Saturday session from the library to the community hall. Volunteers should arrive at 9:30 and visitors at 10.'),
  ('technical','Explain the difference between compressing a file and encrypting it, including one everyday use of each.'),
  ('review','Write a fictional first-person review of a backpack after a weekend trip: comfortable straps, awkward side pocket, and a zipper that caught once.'),
  ('academic','Write an abstract for a fictional small study comparing plant growth under two colors of light. Mention the small sample and avoid invented numerical results.')],
 '100-199':[
  ('business','Write a professional email explaining a fictional workshop schedule change: the practical session moves to the afternoon, registration remains at 8:45, lunch is included, and participants should bring laptops.'),
  ('technical','Explain why a room can feel stuffy even when the temperature is comfortable. Discuss humidity and airflow without giving medical advice.'),
  ('review','Write a fictional personal review of a used coffee grinder: uneven coarse settings, good fine settings, a noisy motor, and a removable cup that makes cleaning easier.'),
  ('academic','Write a clear abstract for a fictional classroom project comparing the stability of folded cardboard bridges. Describe the method, the limited scope, and a cautious conclusion without claiming real scientific findings.')]}
WORDS={'lt25':'8–16','25-49':'20–32','50-99':'40–65','100-199':'85–120'}

def work(job):
 model,b,genre,task=job;ident=digest([model,b,genre,task]);path=OUT/'direct_generated'/(ident+'.json');path.parent.mkdir(exist_ok=True)
 if path.exists():return json.loads(path.read_text())
 body={'model':model,'max_tokens':2048,'reasoning':({'enabled':False} if model.startswith(('qwen/','deepseek/')) else {'effort':'minimal','exclude':True}),
       'messages':[{'role':'system','content':'Write the requested English text in a normal helpful style. Return only the text.'},{'role':'user','content':task+' Use '+WORDS[b]+' words.'}]}
 cache=OUT/'generation_responses'/(digest(body)+'.json')
 if cache.exists():result=json.loads(cache.read_text())['response']
 else:
  result=request('https://openrouter.ai/api/v1/chat/completions',os.environ['OPENROUTER_API_KEY'],body);cache.write_text(json.dumps({'request':body,'response':result},ensure_ascii=False))
 choice=result['choices'][0];text=(choice['message'].get('content') or '').strip()
 if not text or choice['finish_reason']!='stop':raise ValueError('Incomplete direct generation')
 row={'id':ident,'hash':digest(' '.join(text.split())),'text':text,'group':'direct:'+b+':'+genre,'source':'direct_generated','origin':'ai_direct','scenario':'direct_'+genre,'model':model,'target_bucket':b,'tokens':tokens(text),'token_bucket':bucket(text),'word_count':len(text.split()),'task':task}
 path.write_text(json.dumps(row,ensure_ascii=False));print(model,b,genre,flush=True);return row

if __name__=='__main__':
 jobs=[(m,b,genre,task) for m in MODELS for b,values in TASKS.items() for genre,task in values];random.Random(214).shuffle(jobs)
 with ThreadPoolExecutor(max_workers=12) as pool:rows=list(pool.map(work,jobs))
 save(OUT/'direct.jsonl',rows);print('Direct generated:',len(rows),flush=True)
