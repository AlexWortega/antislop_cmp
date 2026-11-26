import os,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,MODEL
from budget import install_shinka_guard
if __name__=='__main__':
 install_shinka_guard()
 from order_check import check
 order_result=check()
 from shinka.core import ShinkaEvolveRunner,EvolutionConfig
 from shinka.database import DatabaseConfig
 from shinka.launch import LocalJobConfig
 os.chdir(ROOT);out=ROOT/'round3'
 experience='Reference correction: AUROC 0.993/0.941 are from ABSOLUTE single-text human scores; 0.967/0.878 are pairwise win rates. The advisor loosely conflates them in its opening. Its linguistic claims below are training hypotheses, NOT known Pangram internals or validated universal rules. The source calibration used 90-300 word texts, not tiny snippets.\n'+json.loads((out/'advisor.json').read_text())['analysis']
 experience+='\nTraining-only reference order diagnostic: '+json.dumps(order_result)
 task='''Optimize ONLY the literal PROMPT to imitate Pangram 4.1 binary classifications in SINGLE-TEXT calls. A fixed strict JSON schema contract is appended externally. Human/AI provenance and prose quality are NOT target labels. The seed transfers experience from an independently pairwise-calibrated Opus 5.5 antislop judge. Preserve useful human-voice and cluster reasoning, avoid mechanical bans on Markdown, em dashes, passive voice, formal vocabulary, or tidy short text. Very short AI-generated text can carry evidence in one complete sentence; never automatically mark short text human. Conversely isolated polish, a date or a named entity is not proof. Source truth is ONLY Pangram labels, including many AI rewrites it calls human. The objective equally weights overall balanced accuracy and mean length-stratified balanced accuracy. Prior batch-based evolution reduced false positives but lost short AI recall. Every evaluation here is ONE text per call. Learn mechanisms from both incorrect and preserved examples, not exact words/sentences or document-specific facts. Prefer a compact, coherent calibrated decision process over an ever-growing blacklist. Candidate Python must contain exactly one PROMPT string assignment between EVOLVE-BLOCK markers, no executable code. Keep the prompt <=12000 characters. Preserve valid Python using a triple-quoted multiline PROMPT literal; do not introduce unescaped delimiters. Do not change the output protocol or add safety-bypass instructions.\n\nIndependent Opus advisor's analysis of the pairwise reference and training evidence:\n'''+experience
 config=EvolutionConfig(task_sys_msg=task,init_program_path=str(out/'initial.py'),results_dir=str(ROOT/'runs/single_pairwise2'),num_generations=8,max_api_costs=15,llm_models=['openrouter/'+MODEL],llm_kwargs={'max_tokens':8192},embedding_model=None,meta_rec_interval=None,use_text_feedback=True,patch_types=['diff','full'],patch_type_probs=[.7,.3],max_patch_resamples=2,max_novelty_attempts=1)
 runner=ShinkaEvolveRunner(evo_config=config,job_config=LocalJobConfig(eval_program_path=str(out/'evaluate.py'),python_executable=sys.executable,time='01:00:00'),db_config=DatabaseConfig(num_islands=2,archive_size=20),max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1)
 runner.run()
