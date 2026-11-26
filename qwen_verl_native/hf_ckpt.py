"""Back up verl checkpoints to a private HF repo and restore them on a new rental.

Runs on the server beside training. A failure here is logged and retried; it never touches the trainer.
"""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
REPO='AlexWortega/qwen-verl-antislop-checkpoints'
TOKEN_FILE='/root/.hf_backup_token';CHECKPOINTS=Path('/workspace/native/output/checkpoints')
SNAPSHOTS=Path('/workspace/native/hf_snapshots');ADAPTER=Path('/workspace/native/adapter-step0010')
LOG=Path('/workspace/native/output/hf_backup.jsonl');STEP_FILE='latest_checkpointed_iteration.txt';EVERY=10
def due(latest,last):return last is None or latest//EVERY>last//EVERY
def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1<<22),b''):h.update(chunk)
 return h.hexdigest()
def manifest(folder):
 folder=Path(folder);files=sorted(p for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json' and '.cache' not in p.parts)
 return {str(p.relative_to(folder)):{'size':p.stat().st_size,'sha256':sha256(p)} for p in files}
def remote_mismatches(expected,remote):
 """remote maps relative path -> (size, lfs sha256 or None); small non-LFS files are checked by size."""
 bad=[]
 for name,want in expected.items():
  got=remote.get(name)
  if got is None or got[0]!=want['size'] or (got[1] is not None and got[1]!=want['sha256']):bad.append(name)
 return bad
def local_mismatches(folder):
 expected=json.loads((Path(folder)/'manifest.json').read_text());return [k for k,v in expected.items() if not (Path(folder)/k).is_file() or sha256(Path(folder)/k)!=v['sha256']]
def log(**event):
 with LOG.open('a') as f:f.write(json.dumps({'time':time.time(),**event})+'\n')
def client():
 from huggingface_hub import HfApi
 return HfApi(token=Path(TOKEN_FILE).read_text().strip())
def upload(api,folder,name):
 """Upload one folder and confirm every file by size and hash before anyone may rely on it."""
 folder=Path(folder);expected=manifest(folder);(folder/'manifest.json').write_text(json.dumps(expected,indent=1))
 api.upload_folder(repo_id=REPO,folder_path=str(folder),path_in_repo=name,commit_message='Add '+name,ignore_patterns=['.cache/*'])
 listed={}
 for entry in api.list_repo_tree(REPO,path_in_repo=name,recursive=True):
  if hasattr(entry,'size') and entry.size is not None:
   listed[entry.path[len(name)+1:]]=(entry.size,entry.lfs.sha256 if getattr(entry,'lfs',None) else None)
 bad=remote_mismatches(expected,listed)
 if bad:raise RuntimeError('Hub copy differs: '+', '.join(bad))
def mark_latest(api,step,name):
 api.upload_file(repo_id=REPO,path_or_fileobj=json.dumps({'step':step,'path':name,'time':time.time()}).encode(),path_in_repo='LATEST.json',commit_message=f'Latest verified: step {step}')
def backup_step(api,step):
 name=f'global_step_{step}';snap=SNAPSHOTS/name;shutil.rmtree(snap,ignore_errors=True);SNAPSHOTS.mkdir(parents=True,exist_ok=True)
 # Hard links keep the files alive while the trainer rotates old checkpoints away.
 subprocess.run(['cp','-al',str(CHECKPOINTS/name),str(snap)],check=True)
 try:upload(api,snap,name);mark_latest(api,step,name)
 finally:shutil.rmtree(snap,ignore_errors=True)
def backup_adapter(api):
 # manifest.json must not land in the adapter directory the trainer reads, so copy links aside.
 snap=SNAPSHOTS/'adapter-step0010';shutil.rmtree(snap,ignore_errors=True);SNAPSHOTS.mkdir(parents=True,exist_ok=True)
 subprocess.run(['cp','-al',str(ADAPTER),str(snap)],check=True)
 try:upload(api,snap,'adapter-step0010')
 finally:shutil.rmtree(snap,ignore_errors=True)
def scrub(text,token):return text.replace(token,'[token]') if token else text
def watch():
 token=Path(TOKEN_FILE).read_text().strip();api=client();last=None;adapter_done=False
 if not api.repo_info(REPO).private:raise SystemExit('Refusing to upload: repo is not private')
 log(event='watch_started',repo=REPO)
 while True:
  try:
   if not adapter_done:backup_adapter(api);adapter_done=True;log(event='adapter_uploaded')
   latest=int((CHECKPOINTS/STEP_FILE).read_text().strip())
   if due(latest,last):
    started=time.time();backup_step(api,latest);last=latest;log(event='uploaded',step=latest,seconds=round(time.time()-started))
  except Exception as exc:log(event='error',error=scrub(type(exc).__name__+': '+str(exc),token)[:500])
  time.sleep(60)
def restore(native='/workspace/native'):
 """Fetch the newest verified checkpoint and the adapter, verify hashes, then publish the resume step."""
 from huggingface_hub import hf_hub_download,snapshot_download
 token=Path(TOKEN_FILE).read_text().strip();native=Path(native);checkpoints=native/'output/checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
 info=json.loads(Path(hf_hub_download(REPO,'LATEST.json',token=token,force_download=True)).read_text());name=info['path']
 incoming=native/'hf_incoming';shutil.rmtree(incoming,ignore_errors=True)
 snapshot_download(REPO,allow_patterns=[name+'/*','adapter-step0010/*'],local_dir=str(incoming),token=token)
 for folder in (name,'adapter-step0010'):
  bad=local_mismatches(incoming/folder)
  if bad:raise RuntimeError(f'Downloaded {folder} differs: '+', '.join(bad))
 for folder,target in ((name,checkpoints/name),('adapter-step0010',native/'adapter-step0010')):
  shutil.rmtree(target,ignore_errors=True);shutil.move(str(incoming/folder),str(target));(Path(target)/'manifest.json').unlink()
  shutil.rmtree(Path(target)/'.cache',ignore_errors=True)
 shutil.rmtree(incoming,ignore_errors=True)
 temp=checkpoints/(STEP_FILE+'.tmp');temp.write_text(str(info['step']));os.replace(temp,checkpoints/STEP_FILE)
 print(json.dumps({'restored_step':info['step']}),flush=True)
if __name__=='__main__':
 {'watch':watch,'restore':restore}[sys.argv[1]]()
