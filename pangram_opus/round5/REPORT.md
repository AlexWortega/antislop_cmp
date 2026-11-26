# Pangram imitation with Opus 5.5 — round 5

Fresh test: **127/142 (89.4%)**, balanced accuracy **87.7%**. Operational target: **NOT MET**.

This measures agreement with Pangram 4 (returned version 4.1), not true authorship or writing quality. The prompt and runtime are selected using calibration data before fresh outcomes; the fresh test never selects them. Small strata and correlated variants limit generalization claims.

The full pairwise rulebook from [anti-slop, commit 388cbe3](https://github.com/miqdadbadjuber/anti-slop/blob/388cbe3b6c37d5175b9f460015bb092ef9e34894/skills/antislop-copywriting/SKILL.md) and an independent Opus advisor were incorporated in round 3. Source pairwise results are not Pangram-imitation results. Provenance, MIT license and earlier procedure: [round 3 report](round3/REPORT.md).

Round 4 independent test scored 133/142 (93.7%), balanced accuracy 94.2%. Those data are now calibration data. The first round-5 trial added 26 matched references, excluding their source groups from scored train and dev. Shinka evaluated three candidates on 166 training cases, then compared them with the prior baseline on calibration dev. Dev was expanded to 86 cases before scoring because one short-text positive stratum was sparse. The added-reference candidates failed the improvement gate. A separate scalar-probability experiment with a train-only threshold also failed. These rejected experiments are retained in validation.json and score_validation.json. Next, the previous working Shinka prompt was compared in default, medium and high reasoning modes. These reasoning-mode checks also failed to improve the default. Retrieval of six matched examples from an excluded-group reference index was then evaluated and rejected. This repeated calibration dev is NOT an independent generalization estimate. The mode gate requires objective gain of at least 0.005 with agreement and balanced accuracy no worse than default. All decisions precede fresh labels/outcomes. Only the new final test supplies independent evidence.

New test: 142 unique texts, 56 new document/topic groups; exact and near duplicates checked against 1122 prior rows. Human documents, six generator models, plain/light/antislop rewrites, Markdown-only and human/AI mixtures. There are 2 deduplicated variants. Length uses o200k_base, a proxy rather than Opus's proprietary tokenizer. Pangram recommends at least 50 words; very short results imitate API behavior outside that recommendation.

| Metric | Previous round | Selected |
|---|---:|---:|
| Agreement | 136/142 (95.8%) | 127/142 (89.4%) |
| Balanced accuracy | 95.6% | 87.7% |
| False positive | 4 | 8 |
| False negative | 2 | 7 |
| Unavailable (counted as errors) | 0 | 0 |
| Under 200 tokens | 132/138 (95.7%) | 123/138 (89.1%) |

Final calibration stage: expanded learning to 480 cases, including all 86 formerly used calibration-dev cases. A refreshed historical calibration dev has 112 cases from separate source groups, with at least eight positives per length bin. Baseline plus one Shinka mutation were evaluated; the selection gate requires objective gain >=0.005 and no regression in agreement or balanced accuracy. These data are not the independent final test. Selected runtime: default. Details and raw-label audit: expanded/protocol.json, expanded/validation.json, expanded/teacher_label_audit.json.

| bucket | Previous | Selected | Selected balanced accuracy |
|---|---:|---:|---:|
| 100-199 | 34/36 (94.4%) | 35/36 (97.2%) | 98.0% |
| 200plus | 4/4 (100.0%) | 4/4 (100.0%) | n/a |
| 25-49 | 32/33 (97.0%) | 28/33 (84.8%) | 79.5% |
| 50-99 | 31/33 (93.9%) | 30/33 (90.9%) | 90.7% |
| lt25 | 35/36 (97.2%) | 30/36 (83.3%) | 76.7% |

| origin | Previous | Selected | Selected balanced accuracy |
|---|---:|---:|---:|
| ai_direct | 45/48 (93.8%) | 37/48 (77.1%) | 57.0% |
| ai_rewrite | 19/22 (86.4%) | 19/22 (86.4%) | 45.2% |
| human | 48/48 (100.0%) | 48/48 (100.0%) | n/a |
| human_format_only | 12/12 (100.0%) | 11/12 (91.7%) | n/a |
| mixed | 12/12 (100.0%) | 12/12 (100.0%) | n/a |

| model | Previous | Selected | Selected balanced accuracy |
|---|---:|---:|---:|
| anthropic/claude-opus-5.5 | 11/12 (91.7%) | 10/12 (83.3%) | 82.9% |
| anthropic/claude-sonnet-5.5 | 14/14 (100.0%) | 12/14 (85.7%) | 85.4% |
| deepseek/deepseek-v4.1-flash | 11/12 (91.7%) | 10/12 (83.3%) | 82.9% |
| google/gemini-3.5-flash | 15/16 (93.8%) | 12/16 (75.0%) | 75.0% |
| not_applicable | 60/60 (100.0%) | 59/60 (98.3%) | n/a |
| openai/gpt-5.5 | 14/16 (87.5%) | 13/16 (81.2%) | 80.2% |
| qwen/qwen3.6-plus | 11/12 (91.7%) | 11/12 (91.7%) | 90.0% |

| scenario | Previous | Selected | Selected balanced accuracy |
|---|---:|---:|---:|
| antislop_rewrite | 6/8 (75.0%) | 6/8 (75.0%) | 42.9% |
| authentic_formal_or_formulaic | 48/48 (100.0%) | 48/48 (100.0%) | n/a |
| direct_academic | 12/12 (100.0%) | 12/12 (100.0%) | n/a |
| direct_business | 10/12 (83.3%) | 7/12 (58.3%) | 58.3% |
| direct_review | 11/12 (91.7%) | 11/12 (91.7%) | 95.5% |
| direct_technical | 12/12 (100.0%) | 7/12 (58.3%) | n/a |
| human_plus_ai_sentence | 12/12 (100.0%) | 12/12 (100.0%) | n/a |
| light_edit | 6/6 (100.0%) | 6/6 (100.0%) | n/a |
| markdown_only | 12/12 (100.0%) | 11/12 (91.7%) | n/a |
| plain_rewrite | 7/8 (87.5%) | 7/8 (87.5%) | n/a |

Corrected 3 errors; introduced 12. Exploratory group-bootstrap 95% interval for balanced-accuracy change: [-23.2, 2.2] percentage points. This includes sampling uncertainty, not future-model guarantees.

Repeatability diagnostic: 8/8 preselected cases stable across three calls; unavailable calls 0. Failed operational checks: overall_balanced_accuracy, overall_agreement, pangram_ai_recall, length:25-49, length:lt25, model:google/gemini-3.5-flash. Thresholds were retained unchanged from round 3.

## Use

From `pangram_opus`, with OPENROUTER_API_KEY in the environment:

```sh
.venv/bin/python classify.py example.txt
```

Returns `{"slop": true}` or `{"slop": false}`. Provider refusal is an error (exit 2), never silently human. `best_prompt.txt` is the calibrated prompt; `inference_system.txt` includes the contract; `request_template.json` and `inference_config.json` reproduce runtime. One classified text per request.

Full artifacts: `round5/selection.json`, `round5/test_results.json`, `round5/results.json`, `round5/goal_assessment.json`. Prior working artifacts are backed up with `round5/previous_`. Corpus provenance is retained; Brown noncommercial research and source terms continue to apply.

New human sources include Brown news and hobbies/practical prose because unused short government and review documents were insufficient. This source choice was made before fresh teacher labels or classifier outcomes; source/topic groups remain disjoint.

## Cumulative budget

- openrouter: conservative $81.37 / $100, including margin and pending reservations.
- pangram: conservative $98.16 / $100, including margin and pending reservations.

The same session ledger is used throughout; do not reset it. Pangram costs are estimated from billed-word units, not a bank statement. API keys are not stored in artifacts.

Exploratory group-bootstrap 95% intervals for the selected classifier: agreement [79.6, 97.5]%; balanced accuracy [72.5, 97.3]%. The operational thresholds concern point estimates; these intervals show their uncertainty.
