---
license: other
license_name: mixed-cc-by-4.0-cc-by-sa-2.5-mit
license_link: LICENSE.md
language:
- en
- ru
task_categories:
- text-generation
tags:
- rl-environment
- reinforcement-learning
- harbor
- llm-judge
- reward-model
- anti-slop
- unslop
- writing
- code
size_categories:
- n<1K
viewer: false
---

[![View tasks in Harbor Visualiser](https://img.shields.io/badge/%F0%9F%A4%97%20Harbor%20Visualiser-View%20tasks-FFD21F?style=for-the-badge)](https://huggingface.co/spaces/HuggingFaceH4/harbor-visualiser?dataset=rawmodels/unslop-harbor)

# Unslop (Harbor)

200 Harbor tasks with the reward we trained
[`AlexWortega/Qwen3.8-27B-antislop`](https://huggingface.co/AlexWortega/Qwen3.8-27B-antislop) on. Your model finishes a
tweet or writes a HumanEval function body. Claude Opus 5.5 grades it against a rubric frozen for that task, ignoring
length. A fixed rule handles length separately. Code has to pass its tests before the judge reads it.

| | |
|---|---|
| Tasks | 100 tweet continuations, 100 HumanEval bodies; half EN, half RU |
| Splits | 8 train / 4 validation / 38 test per domain and language (`[metadata].split`) |
| Judge | `anthropic/claude-opus-5.5` on OpenRouter, Anthropic as the only provider, reasoning low |
| Rubrics | 3 or 4 criteria per task, written once by Opus 5.5 from the calibrated meta-prompts, stored in `tests/task.json` |
| Answer | the agent writes `/workspace/answer.txt` |
| Reward | a number in [0, 1] in `/logs/verifier/reward.txt`; criterion scores and the judge's evidence in `grade.json` |

## Reward

```text
tweets:  r = L(words, target) · σ(a · (q − μ))
code:    r = q / 10 if every HumanEval test passes, else 0
```

- **q** averages the judge's 0–10 criterion scores. The judge runs the GEPA system prompt plus one clause telling
  it to ignore word count. We added the clause after a side-by-side audit: without it, verdicts followed length
  compliance in 18 of 23 decided tasks.
- **L** owns length, so a policy gains nothing by padding or trimming to please the judge. Inside ±15% of the
  target in the prompt, L = 1. Outside, it decays as `exp(−(dev − 0.15)/0.35)` and stays above zero, so in a group of
  overlong samples the shorter ones still win.
- **μ and a** turn q into the chance of beating a typical Qwen3.8-27B answer. We set μ to the judge's mean on 100
  Qwen tweet continuations (6.74) and a to 1.7/σ (1.45).
- **Code** earns nothing until it passes, so the judge only ranks correct solutions. The grader runs the candidate as
  `nobody` with 3 s of CPU, 512 MB, 16 KB files, 64 descriptors and a 6 s wall clock.
- If the judge refuses, truncates or never answers, the grader writes no reward and Harbor logs an error instead of a
  0. An empty answer scores 0 without an API call.

## Calibration

1. **Judge prompt.** GEPA searched the system prompt and the per-domain rubric meta-prompts on train tasks, treating
   "human beats model" as a prior. Validation picked the winner. Test tasks never reached the search.
2. **Tweet rubrics.** We tuned the tweet meta-prompt again on pairs with a known winner: the human tweet against the
   same tweet rewritten by Sonnet 5.5 as assistant prose. On 30 held-out tasks the judge picked the human text 98.3%
   of the time (seed prompt: 96.6%; 95% task-cluster CI [0.95, 1.00]). Code keeps the GEPA meta-prompt.
3. **Reward.** We fit μ and σ on judge scores of Qwen3.8-27B answers. `tests/reward_config.json` pins the judge prompt
   by `judge_system_sha256`, and `tooling/build.py` refuses to build if the hash drifts.

## Layout

```text
tasks/<domain>_<lang>_<id>/
├── task.toml           # judge env, timeouts, provenance (source, license, rubric model)
├── instruction.md      # the assignment word for word, plus where to write the answer
├── environment/        # Dockerfile: python:3.12-slim pinned by digest, unprivileged `agent` user
├── solution/solve.sh   # oracle: writes the human reference
└── tests/              # test.sh, grade.py, task.json (rubric, HumanEval tests), reward_config.json
data/tasks.jsonl        # one row per task: prompt, rubric, target_words, reference, source, license
data/reward_config.json # judge and calibration shared by every task
reward.py               # the same grader, importable outside Harbor
registry.json           # Harbor registry entry
manifest.json           # sha256 of every task directory
jobs/oracle.yaml, jobs/nop.yaml
tooling/                # build.py, gen_rubrics.py, rubric_metas.json
```

## Run

```bash
hf download rawmodels/unslop-harbor --repo-type dataset --local-dir unslop
cd unslop
OPENROUTER_API_KEY=sk-or-... harbor run -p tasks -a <agent> -m <model> -e docker
# the validation runs below:
OPENROUTER_API_KEY=sk-or-... harbor run -y -c jobs/oracle.yaml
harbor run -y -c jobs/nop.yaml
```

For single-turn RL, skip Harbor. `data/tasks.jsonl` holds the exact prompts we calibrated on, without the
`/workspace/answer.txt` line, and `reward.py` scores a completion:

```python
import json, reward
cfg = json.load(open("data/reward_config.json"))
row = json.loads(open("data/tasks.jsonl").readline())
print(reward.grade({**row, "assignment": row["prompt"]}, cfg, completion_text)["reward"])
```

## Validation

We ran every task through Harbor 0.15 on local Docker.

| Agent | Trials | Errors | Mean reward |
|---|---:|---:|---:|
| `nop` (writes nothing) | 200 | 0 | 0.000 on every task |
| `oracle` (human reference) | 200 | 0 | 0.776 |

| Oracle subset | Mean reward | Mean q (0–10) |
|---|---:|---:|
| code EN | 0.933 | 9.34 |
| code RU | 0.928 | 9.29 |
| tweets EN | 0.734 | 7.75 |
| tweets RU | 0.509 | 6.83 |

- All 100 HumanEval references pass inside the task image. Across 30 tasks, 69 of 85 single-site AST mutants fail;
  the other 16 compute the same thing as the reference (`<` → `<=` at a clamp boundary, for one).
- Every reference tweet hits the ±15% length band, so L = 1 and the preference term sets its reward.
- Human Russian tweets land near μ = 6.74. One calibration covers EN and RU tweets, and RuLexNorm tweets are noisy
  and informal.
- A judge call costs about $0.011: roughly 2k prompt tokens and 150 completion tokens.

## Caveats

- We qualified the judge on catching degraded text in known-order pairs. Ranking two good answers is a harder test
  it hasn't passed yet.
- μ and a belong to Opus 5.5. Swap in another `UNSLOP_JUDGE_MODEL` and the grader still runs, with stale calibration.
- Calibration answers had thinking off. Opus 5.5 wrote these rubrics; GEPA ran with Muse Spark 1.1 as the rubric
  writer.
- The references come from public corpora: Broad Twitter Corpus, RuLexNorm, HumanEval. An agent with network access
  can find them. For RL, cut the agent's network or use the tasks as single-turn prompts.
- Sonnet 5.5 wrote the slop in the calibration pairs. Your policy's slop may look different.
- RU code tasks pair Russian instructions with the original English docstrings.
- We left out blogs and scientific abstracts, the other two unslop domains: their sources don't permit
  redistribution.

## Credits

The method follows Meta FAIR's [unslop / RL from eXpert Aligned Rubrics](https://facebookresearch.github.io/RAM/blogs/unslop/),
rebuilt on a different model and judge. [Harbor](https://github.com/harbor-framework/harbor) defines the task format
and runs it. The layout copies the [FineEnvs](https://huggingface.co/FineEnvs) Harbor datasets.

## License

Each subset keeps its source license (`[metadata].license` in every task, `license` in `data/tasks.jsonl`):

- EN tweets: [GateNLP Broad Twitter Corpus](https://github.com/GateNLP/broad_twitter_corpus), CC-BY 4.0
  ([README](licenses/broad-twitter-corpus-README.md)).
- RU tweets: Irina Koliaskina and Dmitry Sholomov / Smart Engines / RuLexNorm, CC-BY-SA 2.5
  ([README](licenses/RuLexNorm-README.md)). This subset stays CC-BY-SA 2.5.
- Code: [HumanEval](https://github.com/openai/human-eval), OpenAI, MIT ([license](licenses/human-eval-LICENSE.txt)).
- Rubrics, grader and task scaffolding: MIT.
