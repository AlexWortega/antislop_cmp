# antislop_cmp

Experiments behind the `antislop` Qwen3.8-27B model on rawmodels.ai: training runs, judges, detector emulators, benchmarks and release material. This is a working lab folder, not a library; each directory is a self-contained experiment.

Not in this repo: model weights, LoRA checkpoints, optimizer state, virtualenvs, API caches and all keys. They are excluded by `.gitignore`. Secrets are read at runtime from `~/.config/rawmodels/*.key` and local `.secrets.env` files.

## Pangram line (Nov 2026)

| Directory | What it is |
|---|---|
| `pangram_opus/` | Opus-based emulator of the Pangram AI detector (`best_prompt.txt`), calibrated on real Pangram labels. See its README. |
| `qwen_pangram_evo/` | Shinka prompt evolution against the emulator (failed: 1/25). |
| `qwen_pangram_grpo/` | GRPO of the 27B model with the emulator as reward. Rounds 1–2, evals, real-Pangram checks. See its README. |
| `qwen_pangram_rewriter/` | System-prompt search for the rewriter (`REPORT.md`). |
| `rewriter_app/` | Local mini app: rewrite any text with `antislop`, optional auto-retry until Pangram passes. |
| `gigacheck/` | GigaCheck-Classifier-Multi checked against real Pangram labels (not a usable proxy). |

## Earlier lines

| Directory | What it is |
|---|---|
| `qwen_verl_pilot/`, `qwen_verl_native/` | verl GRPO pilots on Vast/Harbor. |
| `qwen_scaffold_longrun/` | Long scaffold-training run: steps, evals, judge cache, audits. |
| `judge_playground/` | Judge/reward server experiments. |
| `tasks_v2/`, `tasks_v3/` | Harbor task sets and rubrics. |
| `tb21/`, `bench_astra2/`, `bundle*/` | Terminal-Bench 2.1 runs and astra benchmarks. |
| `cmp/`, `cmp_hemmingway/`, `cmp_rawmodels/` | Side-by-side comparisons vs Hemmingway and the live API. |
| `release/`, `space_antislop/` | Model card, release evidence, HF Space. |
| `PLAN_astra1.md`, `launch_*.sh`, `judge_reward.py` | Astra-judge GRPO stage and launchers. |

Separate repositories that live inside this folder locally and are not part of this repo: `cv/` (AlexWortega/capability-vectors, plus its worktrees `unslop_publish/` and `cmp_hemmingway/source/`), `qwen_verl_pilot/scaffold_training/` (AlexWortega/antislop-scaffold-training, worktree `qwen_scaffold_longrun/repo/`), and third-party clones `anti-slop/`, `gigacheck/src/`, `qwen_verl_pilot/vendor/verl/`, `qwen_verl_pilot/dataset_sources/search-gym/`.
