import json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from core import ROOT,read_prompt,digest,CONTRACT
from round3.runtime import assess_mode
OUT=ROOT/'round5/expanded'
if __name__=='__main__':
 rows=[json.loads(l) for l in (OUT/'dev.jsonl').read_text().splitlines()];paths=[OUT/'initial.py']
 for p in sorted((ROOT/'runs/round5_expanded').glob('gen_*/results/correct.json')):
  if json.loads(p.read_text())['correct']:paths.append(p.parent.parent/'main.py')
 records=[];seen=set()
 for p in paths:
  prompt=read_prompt(p);h=digest(prompt)
  if h in seen:continue
  seen.add(h);print('Calibration dev',p,flush=True);report,pred=assess_mode(prompt,rows,'default',True);records.append({'candidate':str(p),'prompt_hash':h,'report':report,'predictions':pred});(OUT/'validation.json').write_text(json.dumps(records,indent=2));print(report['overall'],flush=True)
 best=max(records,key=lambda r:(r['report']['objective'],-len(read_prompt(r['candidate']))));chosen={k:v for k,v in best.items() if k!='predictions'};prompt=read_prompt(best['candidate']);shutil.copyfile(best['candidate'],OUT/'best_candidate.py');(OUT/'best_prompt.txt').write_text(prompt);(OUT/'inference_system.txt').write_text(prompt+'\n\n'+CONTRACT)
 gain=best['report']['objective']-records[0]['report']['objective'];selection={'criterion':'fixed reused source-group-disjoint calibration dev objective; no fresh labels or outcomes','selected':chosen,'gain_over_previous':gain,'fresh_label_gate':best['prompt_hash']!=records[0]['prompt_hash'] and gain>=.005 and best['report']['overall']['agreement']>=records[0]['report']['overall']['agreement'] and best['report']['overall']['balanced_accuracy']>=records[0]['report']['overall']['balanced_accuracy']};(OUT/'selection.json').write_text(json.dumps(selection,indent=2));(OUT/'runtime_selection.json').write_text(json.dumps({'mode':'default','prompt_hash':digest(prompt)},indent=2));print('Selection',selection,flush=True)
 if not selection['fresh_label_gate']:raise RuntimeError('Calibration improvement gate not met; retain unlabeled fresh test and continue calibration without spending teacher budget.')
