import json
import os
import subprocess
import sys
import time
from pathlib import Path
OUT=Path(__file__).resolve().parent
os.chdir(OUT.parent.parent)

def stage(name,args):
 (OUT/'pipeline_status.json').write_text(json.dumps({'stage':name,'status':'running','started':time.time()},indent=2))
 with open(OUT/(name+'.log'),'a',buffering=1) as log:
  result=subprocess.run([sys.executable,'-u',*args],stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  (OUT/'pipeline_status.json').write_text(json.dumps({'stage':name,'status':'failed','returncode':result.returncode},indent=2))
  raise SystemExit(result.returncode)

if __name__=='__main__':
 deadline=time.time()+900
 while not (OUT/'direct.jsonl').exists():
  if time.time()>deadline:raise TimeoutError('Direct generation did not finish')
  time.sleep(2)
 stage('label_direct',['-c',"import sys;sys.path.insert(0,'pangram_opus/round2');from expand import OUT,label;label(OUT/'direct.jsonl')"])
 stage('split',['pangram_opus/round2/prepare_round.py','split'])
 stage('mine',['pangram_opus/round2/prepare_round.py','mine'])
 stage('evolve',['pangram_opus/round2/run_round.py','--generations','12'])
 stage('validate',['pangram_opus/round2/finish_round.py','validate'])
 stage('test',['pangram_opus/round2/finish_round.py','test'])
 (OUT/'pipeline_status.json').write_text(json.dumps({'stage':'complete','status':'complete','finished':time.time()},indent=2))
