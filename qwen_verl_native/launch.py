"""Bounded native verl bootstrap and proof run; existing spending is retained."""
import json,os,subprocess,sys,threading,time,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'qwen_scaffold_longrun'
sys.path.insert(0,str(OLD));import run_loop as transport
transport.ROOT=ROOT
remote=transport.remote;rsync=transport.rsync
EVENT=threading.Event();PHASE='setup';tunnel=None

def save(name,data):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)

def phase(value,**kw):
 global PHASE
 PHASE=value;save('progress.json',{'phase':value,'updated_at':time.time(),**kw});print(value,flush=True)

def heartbeat():
 while not EVENT.is_set():
  save('heartbeat.json',{'time':time.time(),'phase':PHASE,'pid':os.getpid()});EVENT.wait(20)

def cleanup():
 phase('cleanup')
 try:
  (ROOT/'gpu_output').mkdir(exist_ok=True)
  rsync('/workspace/native/output/',str(ROOT/'gpu_output')+'/',exclude=('checkpoints','rollouts','validation'))
  rsync('/workspace/native/audit/',str(ROOT/'audit')+'/')
 finally:
  (ROOT/'STOP').touch()
  if tunnel:tunnel.terminate()
  EVENT.set()

threading.Thread(target=heartbeat,daemon=True).start()
try:
 for _ in range(90):
  try:
   if remote('echo ready',timeout=25).strip()=='ready':break
  except Exception:pass
  time.sleep(10)
 else:raise RuntimeError('SSH setup timed out')
 remote('mkdir -p /workspace/native/output /workspace/native/audit /workspace/longrun/output /workspace/longrun/vendor/verl')
 rsync('/workspace/longrun/vendor/verl/',str(ROOT.parent/'qwen_verl_pilot/vendor/verl')+'/',upload=True,exclude=('.git','.venv','__pycache__'))
 rsync('/workspace/longrun/bootstrap_gpu.sh',OLD/'bootstrap_gpu.sh',upload=True)
 for name in ('native_agent.py','agent.yaml','run_verl.sh','reward_fn.py','serve.key','train.parquet','val.parquet'):
  rsync('/workspace/native/'+name,ROOT/name,upload=True)
 remote('chmod 600 /workspace/native/serve.key; nohup bash /workspace/longrun/bootstrap_gpu.sh > /workspace/longrun/output/bootstrap.log 2>&1 < /dev/null &')
 phase('uploading_adapter')
 rsync('/workspace/native/adapter-step0010/',str(OLD/'checkpoints/step0010')+'/',upload=True,exclude=('optimizer.pt','.optimizer*','turn_metrics.jsonl','README.md','transfer_verification.json','update.json','adapter_config.server.json'))
 expected=json.loads((OLD/'checkpoints/step0010/transfer_verification.json').read_text())['adapter_model.safetensors']
 actual=remote('sha256sum /workspace/native/adapter-step0010/adapter_model.safetensors').split()[0]
 if actual!=expected:raise RuntimeError('Uploaded adapter SHA256 mismatch')
 phase('waiting_bootstrap')
 for _ in range(240):
  if remote('test -f /workspace/longrun/BOOTSTRAP_DONE && echo ready || true').strip()=='ready':break
  time.sleep(10)
 else:raise RuntimeError('GPU bootstrap deadline')
 # A local bridge exposes only inference; the reverse port exposes authenticated
 # Harbor execution to the one owned GPU. API billing keys stay local.
 tunnel=subprocess.Popen(transport.ssh_args()[:-1]+['-N','-o','ExitOnForwardFailure=yes','-L','0.0.0.0:18974:127.0.0.1:9001','-R','127.0.0.1:18975:127.0.0.1:18975',transport.ssh_args()[-1]],stdout=(ROOT/'tunnel.log').open('w'),stderr=subprocess.STDOUT)
 time.sleep(2)
 if tunnel.poll() is not None:raise RuntimeError('Bridge SSH tunnel failed')
 phase('native_verl_preflight')
 remote('bash /workspace/native/run_verl.sh --cfg job > /workspace/native/output/resolved.yaml',timeout=180)
 phase('native_verl_first_step')
 remote('bash /workspace/native/run_verl.sh > /workspace/native/output/train.log 2>&1',timeout=7200)
 phase('native_verl_step_finished')
 # Preserve actual native checkpoint before releasing the rental. One GPU means
 # every FSDP shard is local to rank zero; no checkpoint is synthesized here.
 (ROOT/'checkpoints').mkdir(exist_ok=True)
 for attempt in range(3):
  try:
   # Longer deadline, no compression for optimizer blobs; retries retain partials.
   state=json.loads((ROOT/'instance_status.json').read_text());shell=__import__('shlex').join(transport.ssh_args()[:-1])
   subprocess.run(['rsync','-a','--partial','--timeout=600','-e',shell,'root@'+state['ssh_host']+':/workspace/native/output/checkpoints/',str(ROOT/'checkpoints')+'/'],check=True,timeout=10800)
   break
  except (subprocess.TimeoutExpired,subprocess.CalledProcessError):
   if attempt==2:raise
 phase('native_verl_proof_saved')
except BaseException as exc:
 phase('failed',error=type(exc).__name__+': '+str(exc));raise
finally:
 try:cleanup()
 except BaseException as exc:
  (ROOT/'STOP').touch();print('Cleanup error '+str(exc),flush=True)
