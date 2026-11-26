"""Loopback-only playground for the exact current Opus anti-slop judge."""
import copy,hashlib,json,os,sys,time,threading,webbrowser
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parent
JUDGE=ROOT.parent/'qwen_scaffold_longrun'
sys.path.insert(0,str(JUDGE))
import judge_budget as budget
from score_groups import indexed_results,validate_labels
TEMPLATE=json.loads((JUDGE/'opus55_antislop_request.json').read_text())
SYSTEM=TEMPLATE['messages'][0]['content']
if not isinstance(SYSTEM,str):SYSTEM='\n'.join(x.get('text','') for x in SYSTEM)
PROMPT_HASH=hashlib.sha256(SYSTEM.encode()).hexdigest()
BUSY=threading.Lock()

def classify(text):
 body=copy.deepcopy(TEMPLATE)
 body['messages'][-1]['content']=json.dumps([{'id':0,'text':text}],ensure_ascii=False)
 cost=0.;attempts=[];started=time.monotonic()
 for attempt in range(3):
  payload=copy.deepcopy(body)
  if attempt:
   payload['messages'].append({'role':'user','content':f'Formatting retry {attempt}. Return exactly one result for integer ID 0. No sentinel IDs, missing IDs, duplicate IDs or extra IDs. Follow the JSON schema.'})
  key=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  cached=(budget.CACHE/(key+'.json')).exists()
  result=budget.call(payload)
  usage=result.get('usage',{})
  charged=0. if cached else float(usage.get('cost',0.))
  cost+=charged;attempts.append({'cached':cached,'cost_usd':charged,'usage':usage})
  try:
   labels=indexed_results(result,[0]);validate_labels(labels)
   return {'slop':labels[0]['slop'],'result':{'results':list(labels.values())},'cost_usd':cost,'cached':all(x['cached'] for x in attempts),'seconds':round(time.monotonic()-started,2),'attempts':attempts,'model':TEMPLATE['model'],'prompt_sha256':PROMPT_HASH}
  except (ValueError,KeyError,TypeError,RuntimeError) as exc:
   if attempt==2:raise ValueError('Judge returned an invalid response after 3 attempts; no label assigned') from exc
 raise AssertionError('Unreachable')

def status():
 b=json.loads(budget.LEDGER.read_text())
 remaining=b['limit']-b['spent_with_margin']-sum(b['pending'].values())
 return {'model':TEMPLATE['model'],'connected':bool(os.environ.get('OPENROUTER_API_KEY')),'prompt_sha256':PROMPT_HASH,'prompt_characters':len(SYSTEM),'remaining_usd':round(remaining,2),'budget_shared_with_training':True}

class Handler(BaseHTTPRequestHandler):
 def json(self,code,data):
  raw=json.dumps(data,ensure_ascii=False).encode();self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def do_GET(self):
  if self.path=='/api/status':return self.json(200,status())
  if self.path=='/api/prompt':return self.json(200,{'system':SYSTEM,'sha256':PROMPT_HASH})
  if self.path not in ('/','/index.html'):return self.json(404,{'error':'Not found'})
  raw=(ROOT/'index.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
 def do_POST(self):
  if self.path!='/api/classify':return self.json(404,{'error':'Not found'})
  # Disallow cross-origin spending on the local API.
  origin=self.headers.get('Origin')
  if origin and origin!=f'http://{self.headers.get("Host")}':return self.json(403,{'error':'Invalid origin'})
  if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.json(415,{'error':'JSON required'})
  try:
   length=int(self.headers.get('Content-Length','0'))
   if not 0<length<=250000:raise ValueError('Input too large (maximum 250 KB)')
   data=json.loads(self.rfile.read(length));text=data.get('text')
   if not isinstance(text,str) or not text.strip():raise ValueError('Enter a text to classify')
  except (ValueError,TypeError,AttributeError):return self.json(400,{'error':'Введите непустой текст, максимум 250 КБ.'})
  if not os.environ.get('OPENROUTER_API_KEY'):return self.json(503,{'error':'OPENROUTER_API_KEY is not configured on the server'})
  if not BUSY.acquire(blocking=False):return self.json(409,{'error':'Дождитесь завершения текущей проверки.'})
  try:self.json(200,classify(text))
  except Exception as exc:
   # Never return provider payloads, request headers or environment variables.
   message=str(exc) if isinstance(exc,(ValueError,RuntimeError)) else 'API request failed; try again'
   self.json(502,{'error':message})
  finally:BUSY.release()
 def log_message(self,format,*args):print(format%args,flush=True)

if __name__=='__main__':
 port=int(os.environ.get('JUDGE_PORT','8765'));server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
 print(f'Judge playground: http://127.0.0.1:{port}',flush=True);server.serve_forever()
