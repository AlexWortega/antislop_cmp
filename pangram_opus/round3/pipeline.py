import json,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent

def stage(name,args):
 (OUT/'pipeline_status.json').write_text(json.dumps({'stage':name,'status':'running','started':time.time()},indent=2))
 with open(OUT/(name+'.log'),'a',buffering=1) as log:r=subprocess.run([sys.executable,'-u',*args],stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:
  (OUT/'pipeline_status.json').write_text(json.dumps({'stage':name,'status':'failed','returncode':r.returncode},indent=2));raise SystemExit(r.returncode)

def waitfile(path,seconds):
 deadline=time.time()+seconds
 while not path.exists():
  if time.time()>deadline:raise TimeoutError(str(path))
  time.sleep(3)
if __name__=='__main__':
 waitfile(OUT/'seed_2.py',900)
 stage('seed_support',['pangram_opus/round3/seed_support.py'])
 stage('screen',['pangram_opus/round3/screen.py'])
 stage('evolve',['pangram_opus/round3/run_evo.py'])
 stage('validate',['pangram_opus/round3/finish.py','validate'])
 waitfile(OUT/'fresh_unlabeled.jsonl',300)
 stage('label',['pangram_opus/round3/fresh.py','label'])
 stage('test',['pangram_opus/round3/finish.py','test'])
 stage('publish',['pangram_opus/round3/publish.py'])
 (OUT/'pipeline_status.json').write_text(json.dumps({'stage':'complete','status':'complete','finished':time.time()},indent=2))
