import argparse
import os
import sys
from core import ROOT, MODEL

if __name__ == '__main__':
    from budget import install_shinka_guard
    install_shinka_guard()
    from shinka.core import ShinkaEvolveRunner, EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    p = argparse.ArgumentParser()
    p.add_argument('--generations', type=int, default=20)
    p.add_argument('--mutation-budget', type=float, default=20)
    p.add_argument('--run', default='run1')
    a = p.parse_args()
    os.chdir(ROOT)
    if not (ROOT/'data/train.jsonl').exists():
        raise SystemExit('Run prepare.py first')
    config = EvolutionConfig(
        task_sys_msg='Optimize the PROMPT string to reproduce Pangram 4 binary labels on English text. True means any AI-generated or AI-assisted content. Maximize balanced accuracy. Use training errors to infer generalizable patterns. Candidate must contain ONLY PROMPT = a literal string between EVOLVE-BLOCK markers. Do not add imports, functions, extra assignments or executable code. Return JSON with one boolean slop in the classifier prompt. Never memorize example text or IDs.',
        init_program_path=str(ROOT/'initial.py'), results_dir=str(ROOT/'runs'/a.run),
        num_generations=a.generations, max_api_costs=a.mutation_budget,
        llm_models=['openrouter/'+MODEL], llm_kwargs={'temperatures':[0.7], 'max_tokens':4096},
        embedding_model=None, meta_rec_interval=None, use_text_feedback=True,
        patch_types=['diff','full'], patch_type_probs=[0.7,0.3], max_patch_resamples=2,
        max_novelty_attempts=1)
    runner = ShinkaEvolveRunner(evo_config=config,
        job_config=LocalJobConfig(eval_program_path=str(ROOT/'evaluate.py'), python_executable=sys.executable, time='01:00:00'),
        db_config=DatabaseConfig(num_islands=2, archive_size=30),
        max_evaluation_jobs=1, max_proposal_jobs=1, max_db_workers=1)
    runner.run()
