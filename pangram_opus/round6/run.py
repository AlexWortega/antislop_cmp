import json,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROUND=OUT
if __name__=='__main__':
 for name in ['evolve','validate']:
  (ROUND/'pipeline_status.json').write_text(json.dumps({'stage':'round6_'+name,'status':'running','started':time.time()},indent=2))
  with open(OUT/(name+'.log'),'a',buffering=1) as log:r=subprocess.run([sys.executable,'-u',str(OUT/(name+'.py'))],stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:
   (ROUND/'pipeline_status.json').write_text(json.dumps({'stage':'round6_'+name,'status':'failed','returncode':r.returncode},indent=2));raise SystemExit(r.returncode)
 (ROUND/'pipeline_status.json').write_text(json.dumps({'stage':'round6_calibration','status':'complete','finished':time.time(),'fresh_label_gate':json.loads((OUT/'selection.json').read_text())['fresh_label_gate']},indent=2))
