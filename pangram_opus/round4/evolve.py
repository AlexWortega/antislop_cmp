import json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,MODEL,read_prompt,digest,CONTRACT
from budget import install_shinka_guard
OUT=ROOT/'round4'
if __name__=='__main__':
 install_shinka_guard()
 from shinka.core import ShinkaEvolveRunner,EvolutionConfig
 from shinka.database import DatabaseConfig
 from shinka.launch import LocalJobConfig
 os.chdir(ROOT)
 task='''Calibrate this SINGLE-TEXT Opus classifier to Pangram's binary labels. The previous fresh test has explicitly become training; a separate new source/topic-disjoint final test is unseen. Full original pairwise rulebook is included, but this is absolute teacher imitation, NOT a pairwise preference or true-authorship task. The starting classifier detects all 48 teacher-positive cases in that previous test but has 11 false positives, mostly tiny literal notices, formatting-only transformations, and repeated/copied text in mixed passages. Infer GENERAL decision rules from training feedback. Preserve short positive sensitivity and correct controls. Do not key decisions to a particular subject, name, date, source, model identity, or exact sample phrase. Do not add or modify reference examples. Keep the rulebook and existing reference text intact; make a small calibration addition before the rulebook, or refine the initial decision instructions. A notice, old-fashioned rhetoric, Markdown, or duplicated content is not itself AI evidence; nevertheless distinct assistant-like generative structure may occur in a short text. A polish/local-edit inference is not enough unless the wording supports the teacher's positive boundary. Fix false positives without a blanket human default. Do not alter safety/refusal behavior. Output only a diff editing the literal PROMPT assignment. No executable code. Maximum 60000 characters. No full training sentences (9+ words) may be copied into the prompt. Two bounded mutation attempts after baseline; choose by train objective only.'''
 config=EvolutionConfig(task_sys_msg=task,init_program_path=str(OUT/'initial.py'),results_dir=str(ROOT/'runs/round4_boundary'),num_generations=3,max_api_costs=3,llm_models=['openrouter/'+MODEL],llm_kwargs={'max_tokens':8192},embedding_model=None,meta_rec_interval=None,use_text_feedback=True,patch_types=['diff'],patch_type_probs=[1.0],max_patch_resamples=2,max_novelty_attempts=1)
 runner=ShinkaEvolveRunner(evo_config=config,job_config=LocalJobConfig(eval_program_path=str(OUT/'evaluate.py'),python_executable=sys.executable,time='01:00:00'),db_config=DatabaseConfig(num_islands=1,archive_size=10),max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1);runner.run()
 records=[]
 for p in (ROOT/'runs/round4_boundary').glob('gen_*/results/metrics.json'):
  if not json.loads((p.parent/'correct.json').read_text())['correct']:continue
  candidate=p.parent.parent/'main.py';r=json.loads(p.read_text());records.append({'candidate':str(candidate),'report':r['public'],'prompt_hash':digest(read_prompt(candidate))})
 if not records:raise RuntimeError('No valid candidate')
 best=max(records,key=lambda r:(r['report']['objective'],-len(read_prompt(r['candidate']))));shutil.copyfile(best['candidate'],OUT/'best_candidate.py');prompt=read_prompt(OUT/'best_candidate.py');(OUT/'best_prompt.txt').write_text(prompt);(OUT/'inference_system.txt').write_text(prompt+'\n\n'+CONTRACT)
 (OUT/'selection.json').write_text(json.dumps({'criterion':'train-only bounded Shinka selection, no fresh outcomes','selected':best,'candidates':records},indent=2));(OUT/'runtime_selection.json').write_text(json.dumps({'mode':'default','prompt_hash':digest(prompt)}));print('Frozen',best,flush=True)
