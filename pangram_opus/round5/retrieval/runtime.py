import collections,json,math,os,re,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import tiktoken
from core import ROOT,CONTRACT,digest,request,PredictionUnavailable
from round3.runtime import body_for
from round5.score_runtime import report
ENC=tiktoken.get_encoding('o200k_base')
def features(text):
 words=re.findall(r'[a-z]+|<num>',re.sub(r'\d+','<num>',text.lower()));return collections.Counter(['u:'+w for w in words]+['b:'+a+' '+b for a,b in zip(words,words[1:])])
class Index:
 def __init__(self,path):
  self.rows=json.loads(Path(path).read_text());self.hash=digest(self.rows);counts=[features(r['text']) for r in self.rows];df=collections.Counter(k for c in counts for k in c);self.idf={k:1+math.log((1+len(counts))/(1+n)) for k,n in df.items()};self.vectors=[self.vector(c) for c in counts]
 def vector(self,c):
  v={k:(1+math.log(n))*self.idf[k] for k,n in c.items() if k in self.idf};norm=math.sqrt(sum(x*x for x in v.values())) or 1;return {k:x/norm for k,x in v.items()}
 def select(self,text):
  q=self.vector(features(text));tokens=max(1,len(ENC.encode(text)));ranked=[]
  for r,v in zip(self.rows,self.vectors):
   sim=sum(w*v.get(k,0) for k,w in q.items());sim*=math.exp(-abs(math.log(max(1,r['tokens'])/tokens))/2);ranked.append((sim,r['hash'],r))
  ranked.sort(key=lambda x:(-x[0],x[1]));by_label={False:[],True:[]}
  for label in [False,True]:
   seen=set()
   for sim,h,r in ranked:
    if r['slop']!=label or r['split_group'] in seen:continue
    by_label[label].append(r);seen.add(r['split_group'])
    if len(by_label[label])==3:break
  return [by_label[label][i] for i in range(3) for label in [False,True] if len(by_label[label])>i]
def request_body(prompt,text,neighbors):
 body=body_for(prompt,text,'default');body['messages'][0]['content']=[{'type':'text','text':prompt,'cache_control':{'type':'ephemeral'}},{'type':'text','text':'CALIBRATION REFERENCES (not the input):\n'+json.dumps([{'text':r['text'],'pangram_slop':r['slop']} for r in neighbors],ensure_ascii=False)+'\n\n'+CONTRACT}];return body
def predict(prompt,text,index,cache_tag=None):
 neighbors=index.select(text);key=digest(['retrieval-v1',prompt,index.hash,text,cache_tag]);path=ROOT/'cache'/('opus_retrieval_'+key+'.json')
 if path.exists():response=json.loads(path.read_text())
 else:
  response=request('https://openrouter.ai/api/v1/chat/completions',os.environ['OPENROUTER_API_KEY'],request_body(prompt,text,neighbors));path.write_text(json.dumps(response,ensure_ascii=False))
 choice=response['choices'][0]
 if choice['finish_reason'] in ['length','content_filter']:raise PredictionUnavailable('No retrieval classification: '+choice['finish_reason'])
 values=json.loads(choice['message']['content'])['results']
 if len(values)!=1 or values[0].get('id')!=0 or type(values[0].get('slop')) is not bool:raise ValueError('Invalid retrieval classification schema')
 return values[0]['slop']
def assess(prompt,rows,index):
 def one(r):
  try:return predict(prompt,r['text'],index)
  except PredictionUnavailable:return None
 pred=[]
 with ThreadPoolExecutor(max_workers=6) as pool:
  for i,v in enumerate(pool.map(one,rows)):
   pred.append(v)
   if (i+1)%16==0:print('Retrieved classification',i+1,'/',len(rows),flush=True)
 return report(rows,pred),pred
