import json,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent

def stage(name,args):
 (OUT/'pipeline_status.json').write_text(json.dumps({'stage':name,'status':'running','started':time.time()},indent=2))
 with open(OUT/(name+'.log'),'a',buffering=1) as log:r=subprocess.run([sys.executable,'-u',*args],stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:
  (OUT/'pipeline_status.json').write_text(json.dumps({'stage':name,'status':'failed','returncode':r.returncode},indent=2));raise SystemExit(r.returncode)
if __name__=='__main__':
 stage('setup',['pangram_opus/round4/setup.py'])
 stage('fresh_build',['pangram_opus/round4/fresh.py','build'])
 stage('evolve',['pangram_opus/round4/evolve.py'])
 stage('label',['pangram_opus/round4/fresh.py','label'])
 stage('test',['pangram_opus/round4/finish.py'])
 stage('publish',['pangram_opus/round4/publish.py'])
 (OUT/'pipeline_status.json').write_text(json.dumps({'stage':'complete','status':'complete','finished':time.time()},indent=2))
