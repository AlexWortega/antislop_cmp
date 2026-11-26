"""Bounded binary reasoning-mode check, before fresh labels/outcomes."""
import json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,read_prompt,digest,CONTRACT
from round3.runtime import assess_mode
OUT=ROOT/'round5'
if __name__=='__main__':
 assert not (OUT/'test.jsonl').exists(),'Fresh labels must remain unseen'
 rows=[json.loads(l) for l in (OUT/'dev.jsonl').read_text().splitlines()];prompt=read_prompt(OUT/'previous_best_candidate.py');baseline=json.loads((OUT/'validation.json').read_text())[0]
 (OUT/'reasoning_protocol.json').write_text(json.dumps({'modes':['default','medium','high'],'prompt_hash':digest(prompt),'fixed_prompt':'previous_best_candidate.py','dev_n':len(rows),'max_tokens':4000,'selection':'calibration dev length-balanced objective, tie favors default then medium then high','gate':'objective gain >=0.005 and overall agreement/BA not below default baseline','fresh_outcomes_used':False,'caveat':'This is repeated calibration selection, not an independent generalization estimate.'},indent=2))
 (OUT/'pipeline_status.json').write_text(json.dumps({'stage':'reasoning_calibration','status':'running','started':time.time()},indent=2))
 records=[{'mode':'default','report':baseline['report'],'predictions':baseline['predictions']}]
 for mode in ['medium','high']:
  print('Reasoning mode',mode,flush=True);report,pred=assess_mode(prompt,rows,mode,True);records.append({'mode':mode,'report':report,'predictions':pred});(OUT/'reasoning_validation.json').write_text(json.dumps(records,indent=2));print(mode,report['overall'],flush=True)
 chosen=max(records,key=lambda r:(r['report']['objective'],-['default','medium','high'].index(r['mode'])));gain=chosen['report']['objective']-records[0]['report']['objective'];new=chosen['report']['overall'];old=records[0]['report']['overall'];gate=chosen['mode']!='default' and gain>=.005 and new['agreement']>=old['agreement'] and new['balanced_accuracy']>=old['balanced_accuracy']
 result={'selected':chosen,'gain':gain,'fresh_label_gate':gate,'prompt_hash':digest(prompt)};(OUT/'reasoning_selection.json').write_text(json.dumps(result,indent=2))
 if gate:
  shutil.copyfile(OUT/'previous_best_candidate.py',OUT/'best_candidate.py');(OUT/'best_prompt.txt').write_text(prompt);(OUT/'inference_system.txt').write_text(prompt+'\n\n'+CONTRACT)
  (OUT/'runtime_selection.json').write_text(json.dumps({'mode':chosen['mode'],'prompt_hash':digest(prompt),'criterion':'bounded reasoning-mode calibration check'},indent=2))
  (OUT/'selection.json').write_text(json.dumps({'criterion':'unchanged Shinka prompt, binary output, selected reasoning mode on repeated calibration dev; fresh test unseen','fresh_label_gate':True,'selected':{'candidate':str(OUT/'previous_best_candidate.py'),'prompt_hash':digest(prompt),'report':chosen['report']},'runtime_mode':chosen['mode']},indent=2))
 (OUT/'pipeline_status.json').write_text(json.dumps({'stage':'reasoning_calibration','status':'complete','finished':time.time(),'fresh_label_gate':gate},indent=2));print('Reasoning selection',chosen['mode'],'gain',gain,'gate',gate,flush=True)
