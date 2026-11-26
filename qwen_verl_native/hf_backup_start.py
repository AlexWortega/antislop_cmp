"""Start the HF backup on the current rental. Sends only HF_BACKUP_TOKEN, over ssh stdin."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'qwen_scaffold_longrun'
sys.path.insert(0,str(OLD));import run_loop as t
t.ROOT=ROOT
PYTHON='/workspace/longrun/vendor/verl/.venv/bin/python'
def send_token(token):subprocess.run(t.ssh_args()+['umask 077; cat > /root/.hf_backup_token'],input=token.encode(),check=True,capture_output=True)
if __name__=='__main__':
 token=os.environ.get('HF_BACKUP_TOKEN')
 if not token:raise SystemExit('HF_BACKUP_TOKEN missing from .secrets.env')
 phase=json.loads((ROOT/'progress.json').read_text())['phase']
 # Before training the checkpoint may still be pre-patch, so it must not become the restore point.
 if phase!='native_verl_training':raise SystemExit('Refusing to start during phase '+phase)
 t.rsync('/workspace/native/hf_ckpt.py',ROOT/'hf_ckpt.py',upload=True);send_token(token)
 print(t.remote(f'P=/workspace/native/hf_backup.pid; if [ -f $P ] && kill -0 $(cat $P) 2>/dev/null; then echo already_running; else nohup {PYTHON} -u /workspace/native/hf_ckpt.py watch >> /workspace/native/output/hf_backup.out 2>&1 < /dev/null & echo $! > $P; echo started; fi').strip())
