import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import common  # noqa: E402,F401  (loads keys, puts pangram_opus on sys.path)

if __name__ == '__main__':
    from budget import install_shinka_guard
    install_shinka_guard()
    from shinka.core import ShinkaEvolveRunner, EvolutionConfig
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig
    p = argparse.ArgumentParser()
    p.add_argument('--generations', type=int, default=16)
    p.add_argument('--mutation-budget', type=float, default=2.5)
    p.add_argument('--run', default='run1')
    a = p.parse_args()
    os.chdir(ROOT)
    config = EvolutionConfig(
        task_sys_msg=(
            'Optimize the SYSTEM string: the system prompt for a 27B reasoning model (Qwen, served as "antislop") that '
            'writes English forum answers, emails, notes, posts and rewrites. Goal: its outputs are labeled Human Written '
            'by a Pangram 4 AI-text detector emulator, while still doing the assigned task at the requested length and '
            'inventing no facts. Fitness = share of tasks that pass (human label, task score >= 6, length within '
            '0.6x-1.6x target) + 0.5 * mean shaped human-likelihood. Empty outputs mean the model spent its whole '
            '16384-token budget reasoning: the prompt must keep its thinking short. Infer general writing-style changes '
            'from the failures and from target_style_examples (AI texts the real Pangram passed as human); never quote or memorize specific task texts. The candidate must contain ONLY '
            'SYSTEM = a string literal (at most 12000 characters) between EVOLVE-BLOCK markers, plus optionally '
            'TEMPERATURE = a number in [0.2, 1.6] and TOP_P = a number in [0.5, 1.0] (sampling for the 27B model; the '
            'default is temperature 0.7, top_p 0.95). No other code, imports or variables.'),
        init_program_path=str(ROOT / 'initial.py'), results_dir=str(ROOT / 'runs' / a.run),
        num_generations=a.generations, max_api_costs=a.mutation_budget,
        llm_models=['openrouter/anthropic/claude-opus-5.5'], llm_kwargs={'temperatures': [0.8], 'max_tokens': 4096},
        embedding_model=None, meta_rec_interval=None, use_text_feedback=True,
        patch_types=['diff', 'full'], patch_type_probs=[0.5, 0.5], max_patch_resamples=2,
        max_novelty_attempts=1)
    runner = ShinkaEvolveRunner(
        evo_config=config,
        job_config=LocalJobConfig(eval_program_path=str(ROOT / 'evaluate.py'), python_executable=sys.executable,
                                  time='02:00:00'),
        db_config=DatabaseConfig(num_islands=3, archive_size=30),
        max_evaluation_jobs=1, max_proposal_jobs=1, max_db_workers=1)
    runner.run()
