"""Local Harbor + Opus service; API billing credentials never reach GPU or agents."""
import collections,concurrent.futures,json,os,subprocess,sys,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'qwen_scaffold_longrun';sys.path.insert(0,str(OLD))
from score_groups import score
from cap_grade import confirmed_cap_grade,confirmed_timeout_grade,hermes_startup_failed,confirmed_rate_limit_label_grade,docker_command_timed_out
KEY=(ROOT/'serve.key').read_text().strip();GROUPS={};LOCK=threading.Lock();SLOTS=threading.Semaphore(int(os.environ.get('ANTISLOP_TRIAL_SLOTS','12')));SCORE_LOCK=threading.Lock();STARTUP_BACKOFF=(30,60,120)
GROUP_SIZE=int(os.environ.get('ANTISLOP_GROUP_SIZE','4'))
# Budget and stop errors end the run; anything else from one trial or one judge call is an
# infrastructure failure that gets zero advantage instead of crashing the whole training step.
FATAL=('Judge HTTP 402','Opus budget exhausted','Budget stop','Circuit open')
RECENT=collections.deque(maxlen=64);GROUPS_RECENT=collections.deque(maxlen=16);CIRCUIT=ROOT/'CIRCUIT_OPEN'

def fatal(exc):return any(s in str(exc) for s in FATAL)

def note(kind,failed,reason=''):
 """Track recent trial health; most recent trials failing means a real fault, so stop the run."""
 with LOCK:
  RECENT.append(bool(failed))
  if failed:
   with (ROOT/'infra_failures.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'kind':kind,'reason':reason[:800]})+'\n')
  if len(RECENT)>=32 and sum(RECENT)*2>len(RECENT):CIRCUIT.touch()

def note_group(skipped):
 """Trial health alone misses a judge that fails every call, so also watch whole groups."""
 with LOCK:
  GROUPS_RECENT.append(bool(skipped))
  if len(GROUPS_RECENT)>=8 and sum(GROUPS_RECENT)*2>len(GROUPS_RECENT):CIRCUIT.touch()

def neutral(rows,valid_rewards,reason):
 """Failed trials get the mean valid reward (zero GRPO advantage) and a flag that masks their tokens."""
 mean=sum(valid_rewards)/len(valid_rewards) if valid_rewards else 0.
 return {u:{'reward':mean,'metrics':{'infra_failed':1,'reason':reason[:200]},'infra_failed':True} for u in rows}

def run_trial(body):
 uid=body['uid'];task=body['task'];agent=body['agent'];split=body['split'];out=ROOT/'rollouts'/body['group']/uid;out.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,str(OLD/'repo/scaffold_training/hermes_collect.py'),'--agent',agent,'--agent-version','1.4.0' if agent=='pi' else 'v0.22.5','--preinstalled','--terminal-only','--protocol',str(OLD/'protocol.json'),'--task-root',str(OLD/'tasks'/agent),'--tasks',task,'--split',split,'--api-base','http://host.docker.internal:18974/session/'+uid+'/v1','--policy-id',body['policy'],'--model','antislop-scaffold','--samples','1','--max-turns','4','--timeout','480','--setup-timeout','300','--out',str(out)]
 allowed={'PATH','HOME','LANG','LC_ALL','TMPDIR','DOCKER_HOST','DOCKER_CONFIG','ANTISLOP_SERVE_KEY','SSL_CERT_FILE','ANTISLOP_EXTRA_COMPOSE','HF_HOME'};env={k:v for k,v in os.environ.items() if k in allowed}
 env.update(ANTISLOP_SERVE_KEY=KEY,DOCKER_HOST=os.environ.get('ANTISLOP_DOCKER_HOST','unix:///Users/alexwortega/.colima/antislop-pilot/docker.sock'),DOCKER_CONFIG='/tmp/antislop-longrun-docker-config')
 for attempt in range(len(STARTUP_BACKOFF)+1):
  out.mkdir(parents=True,exist_ok=True)
  with SLOTS,(out/'collector.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1000)
  rows=[json.loads(x) for x in (out/'outcomes.jsonl').read_text().splitlines()];assert len(rows)==1
  row=rows[0]
  if row['grade_outcome'].get('valid'):return row,out
  raw=json.loads(next(out.glob('*-result.json')).read_text())
  if docker_command_timed_out(raw):
   # A loaded host stalled Docker. No verifier ran, so the trial is repeated rather than graded.
   failed=ROOT/'rollouts_startup_failed'/body['group'];failed.mkdir(parents=True,exist_ok=True);out.rename(failed/f'{uid}-{attempt}')
   if attempt==len(STARTUP_BACKOFF):raise ValueError('Docker timed out on every attempt')
   time.sleep(STARTUP_BACKOFF[attempt]);continue
  grade=None
  if agent=='hermes':
   sessions=list(out.glob('trials/*/agent/hermes-session.jsonl'))
   text=sessions[0].read_text() if sessions else ''
   if text.strip():grade=confirmed_cap_grade(raw,json.loads(text.splitlines()[0]))
   # A gateway that dies early may leave no session file, only its message in hermes.txt.
   agent_log=next(iter(out.glob('trials/*/agent/hermes.txt')),None)
   if grade is None and agent_log and hermes_startup_failed(raw,text,agent_log.read_text(errors='replace')):
    # No model message was sent, so a retry discards no sample. The collector
    # skips finished trial names and scoring globs by name: move evidence away.
    failed=ROOT/'rollouts_startup_failed'/body['group'];failed.mkdir(parents=True,exist_ok=True);out.rename(failed/f'{uid}-{attempt}')
    if attempt==len(STARTUP_BACKOFF):raise ValueError('Hermes startup failed on every attempt')
    time.sleep(STARTUP_BACKOFF[attempt]);continue
  if grade is None:grade=confirmed_timeout_grade(raw)
  if grade is None:grade=confirmed_rate_limit_label_grade(raw)
  if grade is None:raise ValueError('Invalid trial grade: '+str(row['grade_outcome'].get('reason')))
  (out/'turn_cap_classification.json').write_text(json.dumps({'original_grade':row['grade_outcome'],'actual_verifier_grade':grade},indent=2));row['grade_outcome']=grade
  return row,out

def process(body):
 ledger=json.loads((OLD/'opus_budget.json').read_text())
 remaining=ledger['limit']-ledger['spent_with_margin']-sum(ledger['pending'].values())
 rental=json.loads((ROOT/'instance.json').read_text())
 if remaining<1. or time.time()>rental['deadline']-int(os.environ.get('ANTISLOP_STOP_BUFFER','3600')) or (ROOT/'REQUEST_STOP').exists():
  raise RuntimeError('Budget stop: preserve latest native checkpoint')
 if CIRCUIT.exists():raise RuntimeError('Circuit open: most recent trials failed')
 group=body['group'];uid=body['uid']
 size=body.get('size') if isinstance(body.get('size'),int) and body['size']>0 else GROUP_SIZE
 with LOCK:
  state=GROUPS.setdefault(group,{'rows':{},'future':concurrent.futures.Future()})
 try:row,out=run_trial(body);note('trial',False)
 except Exception as exc:
  if fatal(exc):
   with LOCK:
    if not state['future'].done():state['future'].set_exception(exc)
   raise
  note('trial',True,f'{group}/{uid}: {type(exc).__name__}: {exc}');row,out=None,None
 with LOCK:
  state['rows'][uid]=(row,out);ready=len(state['rows'])==size
 if ready:
  try:
   valid={u:(r,o) for u,(r,o) in state['rows'].items() if r is not None}
   failed=[u for u,(r,_) in state['rows'].items() if r is None]
   # One hung command is the model's outcome; a whole group timing out is a stalled server.
   if valid and len(valid)==size and all(r['grade_outcome'].get('agent_timeout') for r,_ in valid.values()):raise ValueError('Every trial in group timed out')
   result=None
   if len(valid)>=2:
    parent=next(iter(valid.values()))[1].parent;prepared=parent/'groups.jsonl'
    g={'task':body['task'],'agent':body['agent'],'split':body['split'],'trajectories':[{'trial_id':r['trial_id'],'grade_outcome':r['grade_outcome']} for r,_ in valid.values()]}
    prepared.write_text(json.dumps(g)+'\n')
    try:
     with SCORE_LOCK:score(prepared,parent,parent/'scored')
     metrics=[json.loads(x) for x in (parent/'scored/metrics.jsonl').read_text().splitlines()]
     bytrial={r['trial_id']:r for r in metrics}
     result={u:{'reward':bytrial[r['trial_id']]['reward'],'metrics':bytrial[r['trial_id']]} for u,(r,_) in valid.items()}
     result.update(neutral(failed,[x['reward'] for x in result.values()],'trial failed'))
    except Exception as exc:
     if fatal(exc):raise
     note('judge',True,f'{group}: {type(exc).__name__}: {exc}')
   if result is None:
    # Too few usable trials, or the judge failed: the group carries no learning signal.
    result=neutral(list(state['rows']),[],'group skipped')
    with (ROOT/'skipped_groups.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'group':group,'valid':len(valid)})+'\n')
   note_group(all(r.get('infra_failed') for r in result.values()))
   state['future'].set_result(result)
  except Exception as exc:state['future'].set_exception(exc)
 return state['future'].result(timeout=1500)[uid]

class Handler(BaseHTTPRequestHandler):
 def do_POST(self):
  if self.path!='/run' or self.headers.get('Authorization')!='Bearer '+KEY:self.send_error(401);return
  try:
   body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   if not all(isinstance(body.get(k),str) and body[k].replace('-','').replace('_','').isalnum() for k in ('uid','group','task','agent','split')):raise ValueError('Invalid identifiers')
   result=process(body);payload=json.dumps(result).encode();self.send_response(200)
  except Exception as exc:
   payload=json.dumps({'error':type(exc).__name__+': '+str(exc)}).encode();self.send_response(500)
   print(payload.decode(),flush=True)
  self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)

if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',int(os.environ.get('ANTISLOP_COORDINATOR_PORT','18975'))),Handler).serve_forever()
