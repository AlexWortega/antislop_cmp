"""Fit one global threshold on train; evaluate frozen threshold on calibration dev."""
import json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,read_prompt,digest
from round5.score_runtime import scores,classify_scores,report,SCORE_CONTRACT
OUT=ROOT/'round5'
def read(name):return [json.loads(l) for l in (OUT/name).read_text().splitlines()]
def auc(rows,values):
 positive=[v for r,v in zip(rows,values) if r['slop'] and v is not None];negative=[v for r,v in zip(rows,values) if not r['slop'] and v is not None]
 return sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative)) if positive and negative else None
if __name__=='__main__':
 (OUT/'pipeline_status.json').write_text(json.dumps({'stage':'score_calibration','status':'running','started':time.time()},indent=2))
 prompt=read_prompt(OUT/'previous_best_candidate.py');train=read('train.jsonl');dev=read('dev.jsonl')
 protocol={'method':'Opus estimates Pangram label probability as integer 0..100; fit ONE global threshold on train only; assess fixed threshold on calibration dev.','prompt':'Previous working Shinka prompt, unchanged; only the output contract becomes scalar likelihood.','train_n':len(train),'dev_n':len(dev),'threshold_candidates':list(range(102)),'tie_break':'nearest 50, then lower threshold','gate':'dev objective gain >=0.005 AND dev agreement and balanced accuracy not worse than previous binary baseline','fresh_outcomes_used':False,'fresh_labels_absent':not (OUT/'test.jsonl').exists(),'warning':'Verbal model probabilities are not assumed calibrated. Only measured thresholded predictions count.'}
 assert protocol['fresh_labels_absent'];(OUT/'score_protocol.json').write_text(json.dumps(protocol,indent=2))
 values=scores(prompt,train);(OUT/'train_scores.json').write_text(json.dumps(values));candidates=[]
 for t in range(102):candidates.append({'threshold':t,'report':report(train,classify_scores(values,t))})
 best=max(candidates,key=lambda r:(r['report']['objective'],-abs(r['threshold']-50),-r['threshold']));(OUT/'threshold_selection.json').write_text(json.dumps({'selected':best,'candidates':candidates,'source':'train_only'},indent=2));print('Frozen train threshold',best,flush=True)
 vals=scores(prompt,dev);pred=classify_scores(vals,best['threshold']);r=report(dev,pred);baseline=json.loads((OUT/'validation.json').read_text())[0]['report'];gain=r['objective']-baseline['objective'];gate=gain>=.005 and r['overall']['agreement']>=baseline['overall']['agreement'] and r['overall']['balanced_accuracy']>=baseline['overall']['balanced_accuracy']
 result={'prompt_hash':digest(prompt),'threshold':best['threshold'],'report':r,'scores':vals,'predictions':pred,'binary_baseline':baseline,'objective_gain':gain,'fresh_label_gate':gate,'train_auc_available':auc(train,values),'dev_auc_available':auc(dev,vals)};(OUT/'score_validation.json').write_text(json.dumps(result,indent=2));print('Scalar calibration dev',result['report'],'gain',gain,'gate',gate,flush=True)
 if gate:
  shutil.copyfile(OUT/'previous_best_candidate.py',OUT/'best_candidate.py');(OUT/'best_prompt.txt').write_text(prompt);(OUT/'inference_system.txt').write_text(prompt+'\n\n'+SCORE_CONTRACT)
  (OUT/'score_inference_config.json').write_text(json.dumps({'kind':'score','mode':'default','threshold':best['threshold'],'prompt_hash':digest(prompt)},indent=2))
 (OUT/'pipeline_status.json').write_text(json.dumps({'stage':'score_calibration','status':'complete','finished':time.time(),'fresh_label_gate':gate},indent=2))
