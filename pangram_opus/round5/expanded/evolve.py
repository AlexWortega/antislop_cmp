import json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from core import ROOT,MODEL,read_prompt,digest,CONTRACT
from budget import install_shinka_guard
OUT=ROOT/'round5/expanded'
if __name__=='__main__':
 install_shinka_guard()
 from shinka.core import ShinkaEvolveRunner,EvolutionConfig
 from shinka.database import DatabaseConfig
 from shinka.launch import LocalJobConfig
 os.chdir(ROOT)
 task='Calibrate this SINGLE-TEXT Opus classifier to Pangram 4 binary window labels. Original pairwise rulebook and matched labeled boundary references are included. All reference SOURCE GROUPS are excluded from evaluated train and calibration dev. Only a later new final test is independent; historical data are calibration only. Preserve the full rulebook and every reference example exactly. Change only concise decision instructions before the rulebook. Do not memorize names, topics, numbers, or whole training sentences. In earlier rounds, heuristic-only patches fit train but generalized poorly: literal notices were sometimes missed, while causal explanations and content-preserving rewrites were overflagged. Examples must calibrate the boundary rather than be overridden by blanket stylistic rules. Do not infer that prose was AI-edited merely because it is polished, or that it is human merely because it is short. Preserve sensitivity to genuinely assistant-structured short text while avoiding false positives on conventional human prose. Learn from errors and correct controls across sources. Do not alter provider safety/refusal behavior. Output a diff only changing the literal PROMPT assignment, no executable code. Keep below 60000 characters. One bounded mutation attempt after baseline on the expanded calibration train. Former calibration-dev cases are now learning data; a refreshed source-group-disjoint calibration dev selects the winner. The current prompt has overbroad positive shortcuts for causal explanations and polished prose. Fix general decision mechanisms without memorizing topics or names, preserving sensitivity and all fixed references. Do not add full training sentences (9+ words). The fixed response contract is appended externally.'
 config=EvolutionConfig(task_sys_msg=task,init_program_path=str(OUT/'initial.py'),results_dir=str(ROOT/'runs/round5_expanded'),num_generations=2,max_api_costs=3,llm_models=['openrouter/'+MODEL],llm_kwargs={'max_tokens':8192},embedding_model=None,meta_rec_interval=None,use_text_feedback=True,patch_types=['diff'],patch_type_probs=[1.0],max_patch_resamples=2,max_novelty_attempts=1)
 runner=ShinkaEvolveRunner(evo_config=config,job_config=LocalJobConfig(eval_program_path=str(OUT/'evaluate.py'),python_executable=sys.executable,time='01:00:00'),db_config=DatabaseConfig(num_islands=1,archive_size=10),max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1);runner.run()
 records=[]
 for p in (ROOT/'runs/round5_expanded').glob('gen_*/results/metrics.json'):
  if not json.loads((p.parent/'correct.json').read_text())['correct']:continue
  candidate=p.parent.parent/'main.py';r=json.loads(p.read_text());records.append({'candidate':str(candidate),'report':r['public'],'prompt_hash':digest(read_prompt(candidate))})
 if not records:raise RuntimeError('No valid candidate')
 best=max(records,key=lambda r:(r['report']['objective'],-len(read_prompt(r['candidate']))));shutil.copyfile(best['candidate'],OUT/'best_candidate.py');prompt=read_prompt(OUT/'best_candidate.py');(OUT/'best_prompt.txt').write_text(prompt);(OUT/'inference_system.txt').write_text(prompt+'\n\n'+CONTRACT)
 (OUT/'selection.json').write_text(json.dumps({'criterion':'train-only bounded Shinka selection, no fresh outcomes','selected':best,'candidates':records},indent=2));(OUT/'runtime_selection.json').write_text(json.dumps({'mode':'default','prompt_hash':digest(prompt)}));print('Frozen',best,flush=True)
