import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import ROOT,MODEL
from budget import install_shinka_guard

if __name__=='__main__':
 install_shinka_guard()
 from shinka.core import ShinkaEvolveRunner,EvolutionConfig
 from shinka.database import DatabaseConfig
 from shinka.launch import LocalJobConfig
 p=argparse.ArgumentParser();p.add_argument('--generations',type=int,default=12);p.add_argument('--run',default='expanded1');a=p.parse_args()
 os.chdir(ROOT)
 task='''Optimize ONLY the PROMPT string to reproduce Pangram 4.1 binary labels on English texts, including very short texts. True means any AI-generated or AI-assisted segment; false means Pangram Human Written. The data now includes authentic pre-LLM human prose that sounds formulaic, academic, official or polished; short emails/reviews/speeches; light AI editing; six generator families; mixed and Markdown-only variants. Do not assume that formality, genre, length, factual specificity, a stock word, or a date determines authorship. Learn from both errors and preserved correct examples. Minimize false positives without sacrificing AI/AI-assisted recall. The objective weights overall balanced accuracy and length-stratified balanced accuracy equally. A fixed output schema is supplied separately by the evaluator. Candidate must contain only PROMPT = a literal string between EVOLVE-BLOCK markers; no imports, functions, executable code or additional assignments. Do not memorize sample sentences, IDs or exact document details. Prefer compact generalizable criteria to long lists of fitted phrases. The classifier sees only the text, no origin metadata.'''
 config=EvolutionConfig(task_sys_msg=task,init_program_path=str(ROOT/'round2/previous_best.py'),results_dir=str(ROOT/'runs'/a.run),num_generations=a.generations,max_api_costs=20,
  llm_models=['openrouter/'+MODEL],llm_kwargs={'max_tokens':4096},embedding_model=None,meta_rec_interval=None,use_text_feedback=True,
  patch_types=['diff','full'],patch_type_probs=[.5,.5],max_patch_resamples=2,max_novelty_attempts=1)
 runner=ShinkaEvolveRunner(evo_config=config,job_config=LocalJobConfig(eval_program_path=str(ROOT/'round2/evaluate_round.py'),python_executable=sys.executable,time='01:00:00'),
  db_config=DatabaseConfig(num_islands=2,archive_size=30),max_evaluation_jobs=1,max_proposal_jobs=1,max_db_workers=1)
 runner.run()
