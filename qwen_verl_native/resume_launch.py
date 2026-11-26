"""Prepare a fresh rental, restore the last full checkpoint with the patched LR, supervise tunnel and runner."""
import json,os,shlex,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'qwen_scaffold_longrun'
sys.path.insert(0,str(OLD));import run_loop as t
t.ROOT=ROOT;remote=t.remote;rsync=t.rsync
EVENT=threading.Event();PHASE='setup';STEP=40;OLD_LR=1e-6;NEW_LR=5e-7;PORT=os.environ['ANTISLOP_COORDINATOR_PORT']
PYTHON='/workspace/longrun/vendor/verl/.venv/bin/python'
FROM_HF=os.environ.get('RESUME_FROM_HF')=='1'
CHECKPOINT=ROOT/'milestone_backups'/f'global_step_{STEP}';DEST=f'/workspace/native/output/checkpoints/global_step_{STEP}'
def phase(value):
 global PHASE
 PHASE=value;p=ROOT/'progress.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps({'phase':value,'updated_at':time.time()}));temp.replace(p);print(value,flush=True)
def heartbeat():
 while not EVENT.is_set():
  p=ROOT/'heartbeat.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps({'phase':PHASE,'time':time.time(),'pid':os.getpid()}));temp.replace(p);EVENT.wait(20)
def wait_rental():
 for _ in range(180):
  path=ROOT/'instance_status.json'
  if path.exists() and json.loads(path.read_text()).get('ssh_host'):
   try:
    if remote('echo ready',timeout=25).strip()=='ready':return
   except Exception:pass
  time.sleep(10)
 raise RuntimeError('SSH setup timed out')
def upload_checkpoint():
 state=json.loads((ROOT/'instance_status.json').read_text());shell=shlex.join(t.ssh_args()[:-1])
 remote('mkdir -p '+DEST)
 for attempt in range(4):
  if subprocess.run(['rsync','-a','--partial','--timeout=600','--exclude','transfer_verification.json','-e',shell,str(CHECKPOINT)+'/','root@'+state['ssh_host']+':'+DEST+'/'],timeout=7200).returncode==0:break
 else:raise RuntimeError('Checkpoint upload failed')
 expected=json.loads((CHECKPOINT/'transfer_verification.json').read_text())
 for name,digest in expected.items():
  if remote('sha256sum '+shlex.quote(DEST+'/'+name),timeout=300).split()[0]!=digest:raise RuntimeError('Checkpoint upload hash mismatch: '+name)
 print(json.dumps({'checkpoint_uploaded_files':len(expected)}),flush=True)
def tunnel_loop(stop):
 args=t.ssh_args()
 with (ROOT/'tunnel.log').open('a') as log:
  while not stop.is_set():
   child=subprocess.Popen(args[:-1]+['-N','-o','ExitOnForwardFailure=yes','-L','0.0.0.0:18974:127.0.0.1:9001','-R','127.0.0.1:18975:127.0.0.1:'+PORT,args[-1]],stdout=log,stderr=subprocess.STDOUT)
   while child.poll() is None and not stop.is_set():stop.wait(5)
   if child.poll() is None:child.terminate()
   elif not stop.is_set():print(json.dumps({'tunnel_restart':child.returncode,'time':time.time()}),flush=True);stop.wait(5)
threading.Thread(target=heartbeat,daemon=True).start();STOP_TUNNEL=threading.Event()
code=1
try:
 phase('waiting_rental');wait_rental()
 remote('mkdir -p /workspace/native/output/checkpoints /workspace/native/audit /workspace/longrun/output /workspace/longrun/vendor/verl')
 phase('uploading_code')
 rsync('/workspace/longrun/vendor/verl/',str(ROOT.parent/'qwen_verl_pilot/vendor/verl')+'/',upload=True,exclude=('.git','.venv','__pycache__'))
 rsync('/workspace/longrun/bootstrap_gpu.sh',OLD/'bootstrap_gpu.sh',upload=True)
 for name in ('native_agent.py','native_compat.py','agent.yaml','run_verl.sh','reward_fn.py','serve.key','train.parquet','val.parquet','patch_lr.py','hf_ckpt.py'):
  rsync('/workspace/native/'+name,ROOT/name,upload=True)
 remote('chmod 600 /workspace/native/serve.key; nohup bash /workspace/longrun/bootstrap_gpu.sh > /workspace/longrun/output/bootstrap.log 2>&1 < /dev/null &')
 if FROM_HF:
  from hf_backup_start import send_token
  send_token(os.environ['HF_BACKUP_TOKEN'])
 else:
  phase('uploading_adapter')
  rsync('/workspace/native/adapter-step0010/',str(OLD/'checkpoints/step0010')+'/',upload=True,exclude=('optimizer.pt','.optimizer*','turn_metrics.jsonl','README.md','transfer_verification.json','update.json','adapter_config.server.json'))
  expected=json.loads((OLD/'checkpoints/step0010/transfer_verification.json').read_text())['adapter_model.safetensors']
  if remote('sha256sum /workspace/native/adapter-step0010/adapter_model.safetensors').split()[0]!=expected:raise RuntimeError('Uploaded adapter SHA256 mismatch')
  phase('uploading_checkpoint');upload_checkpoint()
  remote(f'echo {STEP} > /workspace/native/output/checkpoints/latest_checkpointed_iteration.txt')
 phase('waiting_bootstrap')
 for _ in range(240):
  if remote('test -f /workspace/longrun/BOOTSTRAP_DONE && echo ready || true').strip()=='ready':break
  time.sleep(10)
 else:raise RuntimeError('GPU bootstrap deadline')
 if FROM_HF:
  # Hub checkpoints were saved by the patched trainer, so they already carry the new LR.
  phase('restoring_from_hf')
  print(remote(f'{PYTHON} /workspace/native/hf_ckpt.py restore',timeout=3600),flush=True)
 else:
  phase('patching_lr')
  print(remote(f'{PYTHON} /workspace/native/patch_lr.py {DEST} {OLD_LR} {NEW_LR}',timeout=900),flush=True)
 threading.Thread(target=tunnel_loop,args=(STOP_TUNNEL,),daemon=True).start();time.sleep(5)
 phase('native_verl_preflight')
 remote('bash /workspace/native/run_verl.sh --cfg job > /workspace/native/output/resolved.yaml',timeout=300)
 if 'lr: 5.0e-07' not in remote('grep -E "^ +lr: " /workspace/native/output/resolved.yaml'):raise RuntimeError('Resolved config lacks the new LR')
 EVENT.set();code=subprocess.run([sys.executable,'-u',str(ROOT/'native_runner.py')],env=os.environ).returncode
except BaseException as exc:
 phase('failed');(ROOT/'launch_failure.json').write_text(json.dumps({'error':type(exc).__name__+': '+str(exc),'phase':PHASE,'time':time.time()}));print(exc,flush=True)
finally:
 STOP_TUNNEL.set();(ROOT/'STOP').touch();EVENT.set()
sys.exit(code)
