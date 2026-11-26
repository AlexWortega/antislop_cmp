# Pangram imitation with Opus 5.5 — round 4

Fresh test: **133/142 (93.7%)**, balanced accuracy **94.2%**. Operational target: **NOT MET**.

This measures agreement with Pangram 4 (returned version 4.1), not true authorship or writing quality. The selected prompt is the train-selected candidate; the fresh test never selects it. Small strata and correlated variants limit generalization claims.

The full pairwise rulebook from [anti-slop, commit 388cbe3](https://github.com/miqdadbadjuber/anti-slop/blob/388cbe3b6c37d5175b9f460015bb092ef9e34894/skills/antislop-copywriting/SKILL.md) and an independent Opus advisor were incorporated in round 3. Source pairwise results are not Pangram-imitation results. Provenance, MIT license and earlier procedure: [round 3 report](round3/REPORT.md).

Round 3 independent test scored 131/142, balanced accuracy 94.1%, with all 48 positives detected and 11 false positives. Those 142 cases are now explicitly TRAINING, plus 24 controls. Shinka performed baseline plus up to two mutations on 166 training cases. Rules may change but labeled references stay fixed; full sample copying is rejected. Runtime remains provider-default reasoning with strict schema and a cached system prefix. No fresh test outcomes were supplied to the optimizer.

New test: 142 unique texts, 56 new document/topic groups; exact and near duplicates checked against 968 prior rows. Human documents, six generator models, plain/light/antislop rewrites, Markdown-only and human/AI mixtures. There are 2 deduplicated variants. Length uses o200k_base, a proxy rather than Opus's proprietary tokenizer. Pangram recommends at least 50 words; very short results imitate API behavior outside that recommendation.

| Metric | Previous round | Selected |
|---|---:|---:|
| Agreement | 128/142 (90.1%) | 133/142 (93.7%) |
| Balanced accuracy | 92.8% | 94.2% |
| False positive | 14 | 7 |
| False negative | 0 | 2 |
| Unavailable (counted as errors) | 0 | 0 |
| Under 200 tokens | 124/138 (89.9%) | 129/138 (93.5%) |

| bucket | Previous | Selected | Selected balanced accuracy |
|---|---:|---:|---:|
| 100-199 | 32/34 (94.1%) | 33/34 (97.1%) | 97.9% |
| 200plus | 4/4 (100.0%) | 4/4 (100.0%) | n/a |
| 25-49 | 32/36 (88.9%) | 34/36 (94.4%) | 96.0% |
| 50-99 | 33/37 (89.2%) | 34/37 (91.9%) | 93.8% |
| lt25 | 27/31 (87.1%) | 28/31 (90.3%) | 83.6% |

| origin | Previous | Selected | Selected balanced accuracy |
|---|---:|---:|---:|
| ai_direct | 45/48 (93.8%) | 45/48 (93.8%) | 85.2% |
| ai_rewrite | 12/22 (54.5%) | 16/22 (72.7%) | 85.7% |
| human | 48/48 (100.0%) | 48/48 (100.0%) | n/a |
| human_format_only | 12/12 (100.0%) | 12/12 (100.0%) | n/a |
| mixed | 11/12 (91.7%) | 12/12 (100.0%) | n/a |

| model | Previous | Selected | Selected balanced accuracy |
|---|---:|---:|---:|
| anthropic/claude-opus-5.5 | 7/12 (58.3%) | 8/12 (66.7%) | 66.7% |
| anthropic/claude-sonnet-5.5 | 15/15 (100.0%) | 14/15 (93.3%) | 93.8% |
| deepseek/deepseek-v4.1-flash | 9/12 (75.0%) | 11/12 (91.7%) | 90.0% |
| google/gemini-3.5-flash | 12/16 (75.0%) | 13/16 (81.2%) | 80.2% |
| not_applicable | 60/60 (100.0%) | 60/60 (100.0%) | n/a |
| openai/gpt-5.5 | 14/16 (87.5%) | 16/16 (100.0%) | 100.0% |
| qwen/qwen3.6-plus | 11/11 (100.0%) | 11/11 (100.0%) | 100.0% |

| scenario | Previous | Selected | Selected balanced accuracy |
|---|---:|---:|---:|
| antislop_rewrite | 2/8 (25.0%) | 3/8 (37.5%) | 64.3% |
| authentic_formal_or_formulaic | 48/48 (100.0%) | 48/48 (100.0%) | n/a |
| direct_academic | 12/12 (100.0%) | 12/12 (100.0%) | n/a |
| direct_business | 10/12 (83.3%) | 10/12 (83.3%) | 88.9% |
| direct_review | 12/12 (100.0%) | 12/12 (100.0%) | n/a |
| direct_technical | 11/12 (91.7%) | 11/12 (91.7%) | 50.0% |
| human_plus_ai_sentence | 11/12 (91.7%) | 12/12 (100.0%) | n/a |
| light_edit | 6/6 (100.0%) | 6/6 (100.0%) | n/a |
| markdown_only | 12/12 (100.0%) | 12/12 (100.0%) | n/a |
| plain_rewrite | 4/8 (50.0%) | 7/8 (87.5%) | n/a |

Corrected 7 errors; introduced 2. Exploratory group-bootstrap 95% interval for balanced-accuracy change: [-3.5, 4.5] percentage points. This includes sampling uncertainty, not future-model guarantees.

Repeatability diagnostic: 8/8 preselected cases stable across three calls; unavailable calls 0. Failed operational checks: overall_balanced_accuracy, overall_agreement, length:lt25, model:anthropic/claude-opus-5.5. Thresholds were retained unchanged from round 3.

## Use

From `pangram_opus`, with OPENROUTER_API_KEY in the environment:

```sh
.venv/bin/python classify.py example.txt
```

Returns `{"slop": true}` or `{"slop": false}`. Provider refusal is an error (exit 2), never silently human. `best_prompt.txt` is the calibrated prompt; `inference_system.txt` includes the contract; `request_template.json` and `inference_config.json` reproduce runtime. One classified text per request.

Full artifacts: `round4/selection.json`, `round4/test_results.json`, `round4/results.json`, `round4/goal_assessment.json`. Prior working artifacts are backed up with `round4/previous_`. Corpus provenance is retained; Brown noncommercial research and source terms continue to apply.

## Cumulative budget

- openrouter: conservative $61.23 / $100, including margin and pending reservations.
- pangram: conservative $88.31 / $100, including margin and pending reservations.

The same session ledger is used throughout; do not reset it. Pangram costs are estimated from billed-word units, not a bank statement. API keys are not stored in artifacts.

Exploratory group-bootstrap 95% intervals for the selected classifier: agreement [89.5, 97.5]%; balanced accuracy [88.0, 98.0]%. The operational thresholds concern point estimates; these intervals show their uncertainty.
