"""Run genuine verl proof, verify weights, resume native trainer within existing caps."""
import json,os,sys,threading,time,subprocess,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'qwen_scaffold_longrun'
sys.path.insert(0,str(OLD));import run_loop as t
t.ROOT=ROOT
EVENT=threading.Event();PHASE='native_verl_retry'
def save(name,data):
 p=ROOT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def phase(value,**kw):
 global PHASE
 PHASE=value;save('progress.json',{'phase':value,'updated_at':time.time(),**kw});print(value,flush=True)
def heartbeat():
 while not EVENT.is_set():save('heartbeat.json',{'phase':PHASE,'time':time.time(),'pid':os.getpid()});EVENT.wait(20)
def command(text,timeout=120,tries=10):
 """Retry ssh-level failures (exit 255, hung connection): the tunnel to the GPU can drop briefly."""
 for i in range(tries):
  try:return t.remote(text,timeout=timeout)
  except (RuntimeError,subprocess.TimeoutExpired) as exc:
   transient=isinstance(exc,subprocess.TimeoutExpired) or '(255)' in str(exc)
   if not transient or i==tries-1:raise
   print(json.dumps({'ssh_retry':i+1,'error':str(exc)[:200]}),flush=True);time.sleep(30)
def train(args):
 """Run verl detached on the GPU so a dropped ssh connection cannot kill training; poll for its exit code."""
 run='bash /workspace/native/run_verl.sh'+args+' > /workspace/native/output/continuation.log 2>&1; echo $? > /workspace/native/output/train_exit'
 command('rm -f /workspace/native/output/train_exit; (setsid nohup bash -c '+shlex.quote(run)+' > /dev/null 2>&1 < /dev/null &)')
 while True:
  time.sleep(60)
  # Thirty tries 30 s apart ride out a 15-minute tunnel outage before giving up on this rental.
  status=command('cat /workspace/native/output/train_exit 2>/dev/null || echo running',timeout=60,tries=30).strip()
  if status!='running':
   if status!='0':raise RuntimeError('Training exited with '+status)
   return
threading.Thread(target=heartbeat,daemon=True).start()
try:
 if not (ROOT/'native_step_verification.json').exists() and not os.environ.get('ANTISLOP_SKIP_PROOF'):
  phase('native_verl_first_step')
  command('bash /workspace/native/run_verl.sh > /workspace/native/output/train.log 2>&1',timeout=7200)
  phase('native_verl_verifying')
  report=command('/workspace/longrun/vendor/verl/.venv/bin/python /workspace/native/verify_native.py',timeout=180)
  save('native_step_verification.json',json.loads(report.strip().splitlines()[-1]))
  # Keep immutable proof weights via hard links while the trainer rotates files.
  command('cp -al /workspace/native/output/checkpoints/global_step_1 /workspace/native/proof_step_1')
 phase('native_verl_resume')
 command('/workspace/longrun/vendor/verl/.venv/bin/ray stop --force > /workspace/native/output/ray-stop-resume.log 2>&1; pkill -9 -f "[V]LLM::"; true',timeout=120)
 # Standard verl checkpoint resume restores optimizer, sampler and global step.
 phase('native_verl_training',target_native_steps=int(os.environ.get('ANTISLOP_TOTAL_STEPS','200')))
 read_latest=lambda:int(command('cat /workspace/native/output/checkpoints/latest_checkpointed_iteration.txt 2>/dev/null || echo 0').strip())
 TOTAL=os.environ.get('ANTISLOP_TOTAL_STEPS','200')
 start=read_latest();seen=start;failures=0;attempt=0
 while True:
  latest=read_latest()
  if latest>seen:seen=latest;failures=0
  # The resume point has no completed validation until the first new step exists.
  validate=' trainer.val_before_train=True' if latest==start and not os.environ.get('ANTISLOP_SKIP_RESUME_VAL') else ''
  try:
   resume=' trainer.resume_mode=resume_path trainer.resume_from_path=/workspace/native/output/checkpoints/global_step_'+str(latest) if latest>0 else ' trainer.resume_mode=disable'
   train(' trainer.total_training_steps='+TOTAL+' trainer.save_freq=1 trainer.test_freq=10 trainer.max_actor_ckpt_to_keep=2'+resume+validate)
   break
  except RuntimeError:
   command('cp /workspace/native/output/continuation.log /workspace/native/output/continuation-attempt'+str(attempt)+'.log')
   attempt+=1;failures+=1
   # Four failures with no new checkpoint means a real fault, not a transient one.
   if failures>=4 or (ROOT/'GUARD_STOP').exists():raise
   command('/workspace/longrun/vendor/verl/.venv/bin/ray stop --force > /workspace/native/output/ray-stop-retry.log 2>&1; pkill -9 -f "[V]LLM::"; true',timeout=120)
   # Agents from the dead attempt would only call sessions that no longer exist.
   subprocess.run(['pkill','-f','[h]ermes_collect.py'],check=False)
 phase('native_verl_completed')
except BaseException as exc:
 save('failure.json',{'error':type(exc).__name__+': '+str(exc),'phase':PHASE,'time':time.time()});phase('failed',error=str(exc))
 # Bounded recovery window for a compatibility fix on this same rental.
 for _ in range(60):
  if (ROOT/'STOP').exists() or (ROOT/'GUARD_STOP').exists():break
  time.sleep(10)
 raise
finally:
 phase('backup')
 try:
  for name in ('output','audit'):
   (ROOT/name).mkdir(exist_ok=True)
   t.rsync('/workspace/native/'+name+'/',str(ROOT/name)+'/',exclude=('checkpoints',))
  state=json.loads((ROOT/'instance_status.json').read_text());shell=shlex.join(t.ssh_args()[:-1]);name='global_step_'+str(read_latest())
  (ROOT/'checkpoints'/name).mkdir(parents=True,exist_ok=True)
  # Monitor keeps every tenth step; a one-hour cap stops a slow copy billing the idle GPU.
  subprocess.run(['rsync','-a','--partial','--timeout=600','-e',shell,'root@'+state['ssh_host']+':/workspace/native/output/checkpoints/'+name+'/',str(ROOT/'checkpoints'/name)+'/'],check=True,timeout=3600)
 except BaseException as exc:save('backup_error.json',{'error':str(exc)})
 finally:
  (ROOT/'STOP').touch();EVENT.set();phase('finished')
  subprocess.run(['chown','-R','alexwortega:staff',str(ROOT)],check=False)
