"""Explicit, versioned inference settings; frozen on dev before fresh-test inspection."""
import json,os,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,MODEL,CONTRACT,digest,predict,request,PredictionUnavailable
from round2.evaluate_round import metrics,length_group

def body_for(prompt,text,mode):
 # The exported example may describe another inference contract (e.g. a score).
 # Keep binary runtime settings anchored to the immutable binary template.
 body=json.loads((ROOT/'round3/previous_request_template.json').read_text())
 body['model']=MODEL;body['max_tokens']=4000
 body['messages']=[{'role':'system','content':[{'type':'text','text':prompt+'\n\n'+CONTRACT,'cache_control':{'type':'ephemeral'}}]},{'role':'user','content':json.dumps([{'id':0,'text':text}],ensure_ascii=False)}]
 body.pop('reasoning',None)
 if mode in ['medium','high']:body['reasoning']={'effort':mode,'exclude':True}
 elif mode!='default':raise ValueError('Unknown runtime mode')
 return body

def predict_mode(prompt,text,mode,cache_tag=None):
 if mode=='minimal':return predict(prompt,text,cache_tag=cache_tag)
 cache=ROOT/'cache'/('opus_runtime_'+digest(['strict-runtime-v1',mode,prompt,text,cache_tag])+'.json')
 if cache.exists():response=json.loads(cache.read_text())
 else:
  response=request('https://openrouter.ai/api/v1/chat/completions',os.environ['OPENROUTER_API_KEY'],body_for(prompt,text,mode));cache.write_text(json.dumps(response,ensure_ascii=False))
 choice=response['choices'][0]
 if choice['finish_reason'] in ['content_filter','length']:raise PredictionUnavailable('No classification: '+choice['finish_reason'])
 values=json.loads(choice['message']['content'])['results']
 if len(values)!=1 or values[0].get('id')!=0 or type(values[0].get('slop')) is not bool:raise ValueError('Invalid classifier schema')
 return values[0]['slop']

def assess_mode(prompt,rows,mode,progress=False):
 def one(r):
  try:return predict_mode(prompt,r['text'],mode)
  except PredictionUnavailable:return None
 pred=[]
 with ThreadPoolExecutor(max_workers=6) as pool:
  for i,v in enumerate(pool.map(one,rows)):
   pred.append(v)
   if progress and (i+1)%16==0:print('Runtime scored',mode,i+1,'/',len(rows),flush=True)
 overall=metrics(rows,pred);buckets={}
 for b in sorted({length_group(r) for r in rows}):
  sub=[(r,p) for r,p in zip(rows,pred) if length_group(r)==b];buckets[b]=metrics([r for r,p in sub],[p for r,p in sub])
 balanced=[m['balanced_accuracy'] for m in buckets.values() if m['balanced_accuracy'] is not None]
 return {'overall':overall,'by_bucket':buckets,'objective':.5*overall['balanced_accuracy']+.5*sum(balanced)/len(balanced)},pred

if __name__=='__main__':
 from core import read_prompt
 out=ROOT/'round3';rows=[json.loads(l) for l in (out/'dev.jsonl').read_text().splitlines()];prompt=read_prompt(out/'best_candidate.py');results=[]
 for mode in ['minimal','default','medium']:
  print('Runtime dev',mode,flush=True);report,pred=assess_mode(prompt,rows,mode,True)
  results.append({'mode':mode,'report':report,'predictions':pred});(out/'runtime_validation.json').write_text(json.dumps(results,indent=2));print(mode,report['overall'],flush=True)
 chosen=max(results,key=lambda r:(r['report']['objective'],-['minimal','default','medium'].index(r['mode'])))
 (out/'runtime_selection.json').write_text(json.dumps({'mode':chosen['mode'],'prompt_hash':digest(prompt),'report':chosen['report'],'criterion':'dev-only length-balanced objective; ties prefer minimal, default, medium in that order; no fresh outcomes inspected'},indent=2));print('Frozen runtime',chosen['mode'],flush=True)
