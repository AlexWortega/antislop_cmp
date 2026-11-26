import json,os,random,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,MODEL,CONTRACT,read_prompt,digest
from budget import install_shinka_guard
OUT=ROOT/'round3'
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
if __name__=='__main__':
 selection=json.loads((OUT/'selection.json').read_text());m=selection['selected']['report']['overall']
 if m['agreement']>=.95 and m['balanced_accuracy']>=.95:
  (OUT/'fine_status.json').write_text(json.dumps({'skipped':True,'reason':'Dev target already reached'}));raise SystemExit(0)
 prompt=read_prompt(OUT/'best_candidate.py')
 for name in ['best_candidate.py','selection.json','runtime_selection.json']:
  if not (OUT/('pre_fine_'+name)).exists():shutil.copyfile(OUT/name,OUT/('pre_fine_'+name))
 refs=read(OUT/'support.jsonl')+read(OUT/'extended_support.jsonl');blocked={r['split_group'] for r in refs};pool=[r for r in read(OUT/'train_pool.jsonl') if r['split_group'] not in blocked]
 rng=random.Random(3184);rng.shuffle(pool);controls=[]
 for label in [False,True]:
  for b in ['lt25','25-49','50-99','100-199']:
   controls += [r for r in pool if r['slop']==label and r['token_bucket']==b][:4]
 rows=read(OUT/'dev.jsonl')+controls;rng.shuffle(rows);(OUT/'fine_train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 (OUT/'fine_initial.py').write_text('# EVOLVE-BLOCK-START\nPROMPT = """'+prompt.replace('\\','\\\\').replace('"""','\\"\\"\\"')+'"""\n# EVOLVE-BLOCK-END\n')
 markers=[prompt.index(marker) for marker in ['CALIBRATION REFERENCES','ADDITIONAL FIXED CALIBRATION REFERENCES'] if marker in prompt];(OUT/'fine_reference_suffix.txt').write_text(prompt[min(markers):])
 (OUT/'fine_status.json').write_text(json.dumps({'skipped':False,'n':len(rows),'former_dev_status':'Now used as calibration training; it is NOT an independent final validation set. Fresh test only is independent.','controls_n':len(controls),'generations':2,'test_outcomes_used':False},indent=2))
 install_shinka_guard()
 from shinka.core import ShinkaEvolveRunner,EvolutionConfig
 from shinka.database import DatabaseConfig
 from shinka.launch import LocalJobConfig
 os.chdir(ROOT)
 task='''Make ONE small, generalizable calibration improvement to this Pangram-emulating Opus prompt using provided training errors. This training includes former dev and additional disjoint preservation cases; it is not independent validation. The NEW final test is fully unseen. Do not memorize sample wording, names, topics, numbers, or whole sentences. Keep every labeled reference example verbatim; change only classifier criteria, preferably a few lines, to correct a shared mechanism without regressing correct controls. Do not alter provider refusal or safety behavior. Target is Pangram binary labels, not true authorship or prose quality. Preserve format/genre false-positive protections and sensitivity to tiny positive texts. The task is SINGLE-TEXT with the fixed runtime selected on earlier dev. Valid Python must contain only a triple-quoted literal PROMPT assignment, no code. Prefer a short diff and do not rewrite the whole long prompt. Maximum prompt 60000 characters. The fixed JSON response contract is appended externally.'''
 config=EvolutionConfig(task_sys_msg=task,init_program_path=str(OUT/'fine_initial.py'),results_dir=str(ROOT/'runs/pangram_fine1'),num_generations=2,max_api_costs=3,llm_models=['openrouter/'+MODEL],llm_kwargs={'max_tokens':8192},embedding_model=None,meta_rec_interval=None,use_text_feedback=True,patch_types=['diff'],patch_type_probs=[1.0],max_patch_resamples=2,max_novelty_attempts=1)
 runner=ShinkaEvolveRunner(evo_config=config,job_config=LocalJobConfig(eval_program_path=str(OUT/'fine_evaluate.py'),python_executable=sys.executable,time='01:00:00'),db_config=DatabaseConfig(num_islands=1,archive_size=10),max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1);runner.run()
 records=[]
 for path in (ROOT/'runs/pangram_fine1').glob('gen_*/results/metrics.json'):
  if not json.loads((path.parent/'correct.json').read_text())['correct']:continue
  r=json.loads(path.read_text());candidate=path.parent.parent/'main.py';records.append({'candidate':str(candidate),'report':r['public'],'prompt_hash':digest(read_prompt(candidate))})
 if not records:raise RuntimeError('No valid fine-calibration result')
 chosen=max(records,key=lambda r:(r['report']['objective'],-len(read_prompt(r['candidate']))));shutil.copyfile(chosen['candidate'],OUT/'best_candidate.py');prompt=read_prompt(OUT/'best_candidate.py');(OUT/'best_prompt.txt').write_text(prompt);(OUT/'inference_system.txt').write_text(prompt+'\n\n'+CONTRACT)
 (OUT/'selection.json').write_text(json.dumps({'criterion':'bounded error-directed Shinka refinement on former dev plus preservation controls; only fresh test is independent','selected':chosen,'fine_candidates':records},indent=2))
 runtime=json.loads((OUT/'runtime_selection.json').read_text());runtime['prompt_hash']=digest(prompt);runtime['post_runtime_prompt_refinement']='One Shinka mutation using former dev as training; runtime unchanged.';(OUT/'runtime_selection.json').write_text(json.dumps(runtime,indent=2));print('Final frozen after fine calibration',chosen['candidate'],chosen['report']['overall'],flush=True)
