"""Bootstrap the authorized replacement rental then launch the bounded loop."""
import json,subprocess,time,shlex,os
from pathlib import Path
from run_loop import ROOT,remote,rsync,save
try:
 for attempt in range(60):
  try:
   if remote('echo ready',timeout=25).strip()=='ready':break
  except Exception:pass
  time.sleep(10)
 else:raise RuntimeError('Replacement SSH unavailable')
 remote('mkdir -p /workspace/longrun/vendor/verl /workspace/longrun/repo /workspace/longrun/output /workspace/longrun/audit')
 rsync('/workspace/longrun/repo/',str(ROOT/'repo')+'/',upload=True,exclude=('.git','.venv','__pycache__','private','results'))
 vendor=ROOT.parent/'qwen_verl_pilot/vendor/verl'
 rsync('/workspace/longrun/vendor/verl/',str(vendor)+'/',upload=True,exclude=('.git','.venv','__pycache__'))
 for name in ('bootstrap_gpu.sh','serve_gpu.sh','update_gpu.sh','protocol.json'):
  rsync('/workspace/longrun/'+name,ROOT/name,upload=True)
 rsync('/workspace/longrun/serve.key',Path('/tmp/antislop-longrun-serve.key'),upload=True)
 remote('chmod 600 /workspace/longrun/serve.key')
 print('Bootstrap starting',flush=True)
 remote('bash /workspace/longrun/bootstrap_gpu.sh > /workspace/longrun/output/bootstrap.log 2>&1',timeout=3600)
 print('Bootstrap completed',flush=True)
 # Persist exact patched code used by this restart.
 import hashlib
 save(ROOT/'restart_source_manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
 with (ROOT/'run.log').open('w') as log:
  result=subprocess.run([os.sys.executable,'-u',str(ROOT/'run_loop.py')],stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('Training loop exited with error; inspect run.log')
except BaseException:
 (ROOT/'STOP').write_text('Restart launcher failed; cleanup owned rental\n')
 raise
