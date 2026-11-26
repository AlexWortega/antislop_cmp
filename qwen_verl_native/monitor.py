"""Observe real native metrics and back up immutable checkpoint snapshots."""
import concurrent.futures,hashlib,json,math,os,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'qwen_scaffold_longrun'))
import run_loop as transport
transport.ROOT=ROOT
POOL=concurrent.futures.ThreadPoolExecutor(max_workers=1);JOB=None
LAST_BUCKET=max([int(x.name.split('_')[-1])//10 for x in (ROOT/'milestone_backups').glob('global_step_*')],default=0)

def save(name,data):
 p=ROOT/name;temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(p)

def event(data):
 with (ROOT/'monitor_events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**data})+'\n')
 print(json.dumps(data),flush=True)

SCRIPT="""import json,pathlib
r=pathlib.Path('/workspace/native/output')
p=r/'metrics.jsonl';rows=[]
if p.exists():
 for line in p.read_text().splitlines():
  try:rows.append(json.loads(line))
  except ValueError:pass
p=r/'checkpoints/latest_checkpointed_iteration.txt'
latest=int(p.read_text()) if p.exists() else 0
p=r/'continuation.log';tail=p.read_text(errors='replace')[-2500:] if p.exists() else ''
print(json.dumps({'metrics':rows,'latest_checkpoint':latest,'tail':tail}))
"""

def backup(step):
 name=f'global_step_{step}';source='/workspace/native/output/checkpoints/'+name;dest='/workspace/native/milestones/'+name
 transport.remote('mkdir -p /workspace/native/milestones; test -d '+shlex.quote(dest)+' || cp -al '+shlex.quote(source)+' '+shlex.quote(dest),timeout=120)
 code="import pathlib,hashlib,json; r=pathlib.Path("+repr(dest)+"); print(json.dumps({str(p.relative_to(r)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in r.rglob('*') if p.is_file()}))"
 hashes=json.loads(transport.remote('/workspace/longrun/vendor/verl/.venv/bin/python -c '+shlex.quote(code),timeout=180))
 target=ROOT/'milestone_backups'/name;target.mkdir(parents=True,exist_ok=True)
 state=json.loads((ROOT/'instance_status.json').read_text());shell=shlex.join(transport.ssh_args()[:-1])
 for attempt in range(3):
  result=subprocess.run(['rsync','-a','--partial','--timeout=600','-e',shell,'root@'+state['ssh_host']+':'+dest+'/',str(target)+'/'],stdout=subprocess.DEVNULL,stderr=(ROOT/'backup.log').open('a'),timeout=10800)
  if result.returncode==0:break
 else:raise RuntimeError('Checkpoint transfer failed')
 for name,digest in hashes.items():
  with (target/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
  if actual!=digest:raise RuntimeError('Checkpoint hash mismatch: '+name)
 (target/'transfer_verification.json').write_text(json.dumps(hashes,indent=2))
 subprocess.run(['chown','-R','alexwortega:staff',str(target)],check=False)
 event({'checkpoint_backed_up':step,'verified_files':len(hashes)})
 return step

previous=None;failures=0
while True:
 try:
  state=json.loads((ROOT/'instance.json').read_text())
  if state.get('status') in ('destroyed','absent'):event({'monitor_stopped':'rental_absent'});break
  data=json.loads(transport.remote('/workspace/longrun/vendor/verl/.venv/bin/python - <<\'PY\'\n'+SCRIPT+'PY',timeout=90));failures=0
  rows=data.pop('metrics');(ROOT/'native_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
  latest=rows[-1] if rows else {};metrics=latest.get('data',{});step=max(latest.get('step',0),data['latest_checkpoint'])
  alerts=[]
  norm=metrics.get('actor/grad_norm')
  if norm is not None and (not math.isfinite(norm) or norm<=0):alerts.append('Invalid actor gradient norm')
  validations=[{'step':r['step'],'metrics':{k:v for k,v in r['data'].items() if k.startswith(('val-core/','val-aux/'))}} for r in rows if any(k.startswith('val-core/') for k in r['data'])]
  ledger=json.loads((ROOT.parent/'qwen_scaffold_longrun/opus_budget.json').read_text())
  report={'checked_at':time.time(),'completed_native_steps':step,'checkpoint_step':data['latest_checkpoint'],'latest_metrics':latest,'latest_validation':validations[-1] if validations else None,'alerts':alerts,'opus_spent_with_margin':ledger['spent_with_margin'],'recent_log':data['tail'],'backup_running':JOB is not None and not JOB.done()}
  if JOB is not None and JOB.done():
   try:report['last_verified_local_backup']=JOB.result()
   except Exception as exc:alerts.append('Backup failed: '+str(exc));event({'backup_error':str(exc)})
   JOB=None
  if step>=10 and step//10>LAST_BUCKET and JOB is None:
   LAST_BUCKET=step//10;JOB=POOL.submit(backup,data['latest_checkpoint']);event({'checkpoint_backup_started':data['latest_checkpoint']})
  save('monitor_status.json',report)
  if previous!=step or alerts:event({'native_step':step,'alerts':alerts,'validation':report['latest_validation']});previous=step
 except Exception as exc:
  failures+=1;event({'monitor_error':type(exc).__name__+': '+str(exc),'consecutive_failures':failures})
  if failures>=3:save('monitor_alert.json',{'error':str(exc),'checked_at':time.time(),'consecutive_failures':failures})
 time.sleep(60)
POOL.shutdown(wait=True)
