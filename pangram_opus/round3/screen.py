import json,shutil,sys
from pathlib import Path
from evaluate import OUT,read,assess,read_prompt
if __name__=='__main__':
 rows=read(OUT/'screen.jsonl');results=[]
 for name in ['previous_best_candidate.py','seed_0.py','seed_1.py','seed_2.py']:
  path=OUT/name;print('Screen',name,flush=True);report,pred=assess(read_prompt(path),rows,batch_size=1,progress=True)
  results.append({'candidate':str(path),'report':report,'predictions':pred});(OUT/'screen_results.json').write_text(json.dumps(results,indent=2));print(name,report['overall'],flush=True)
 chosen=max(results,key=lambda r:r['report']['objective']);shutil.copyfile(chosen['candidate'],OUT/'initial.py');print('Selected seed',chosen['candidate'],flush=True)
