# Opus 5.5 calibrated to Pangram 4

Working prompt: unchanged round-4 Shinka candidate. Latest independent test: **136/142 (95.8%) agreement**, **95.6% balanced accuracy**. All predeclared operational point-estimate checks passed on this sample.

This imitates Pangram labels, not true authorship or writing quality. The previous independent round-4 test was 133/142 (93.7%), balanced accuracy 94.2%, and failed the same thresholds. The new sample does not establish universal equivalence or erase that failure.

## What was calibrated

The full MIT-licensed pairwise rulebook from [anti-slop](https://github.com/miqdadbadjuber/anti-slop/tree/388cbe3b6c37d5175b9f460015bb092ef9e34894), an independent Opus advisor, fixed boundary examples, and Shinka Evolve were used. The latest expansion trained on 480 cases, including former dev errors, with 112 source-group-disjoint historical calibration cases. Human prose, terse notices, reviews, email, explanations, rewrites, formatting-only and mixed text were included. Source provenance and the distinction between pairwise and Pangram calibration are in round3/reference.json and the round reports.

The latest mutation improved calibration from 104/112 to 105/112 but regressed on the new test to 127/142 (89.4%). It is rejected. Both prompts were fixed before test labels; retaining the previous working version after this comparison is a deployment decision, not a new blind selection claim. Failed experiments, raw responses and predictions remain available in round5/. No test sentences were added to the retained prompt.

## Latest independent test

142 unique texts in 56 new source/topic groups, checked against 1,122 prior rows for group, exact and near overlap. Six generators: GPT-5.5, Opus-5.5, Sonnet-5.5, Gemini-3.5-Flash, DeepSeek-v4.1-Flash, Qwen-3.6-Plus. Plain, light-humanized and anti-slop rewriting, formatting-only and mixed human/AI scenarios are represented. This covers the tested scaffolds, not every possible future scaffold.

Pangram requested version 4, returned 4.1. Positive means any AI-Generated or AI-Assisted window. Human recall: 96.0%; positive recall: 95.2%. False positives: 4; false negatives: 2; unavailable: 0.

| Slice | n | Agreement | Balanced accuracy |
|---|---:|---:|---:|
| bucket: 100-199 | 36 | 94.4% | 96.0% |
| bucket: 200plus | 4 | 100.0% | n/a |
| bucket: 25-49 | 33 | 97.0% | 97.7% |
| bucket: 50-99 | 33 | 93.9% | 92.8% |
| bucket: lt25 | 36 | 97.2% | 91.7% |
| origin: ai_direct | 48 | 93.8% | 90.4% |
| origin: ai_rewrite | 22 | 86.4% | 92.9% |
| origin: human | 48 | 100.0% | n/a |
| origin: human_format_only | 12 | 100.0% | n/a |
| origin: mixed | 12 | 100.0% | n/a |
| model: anthropic/claude-opus-5.5 | 12 | 91.7% | 90.0% |
| model: anthropic/claude-sonnet-5.5 | 14 | 100.0% | 100.0% |
| model: deepseek/deepseek-v4.1-flash | 12 | 91.7% | 92.9% |
| model: google/gemini-3.5-flash | 16 | 93.8% | 93.8% |
| model: not_applicable | 60 | 100.0% | n/a |
| model: openai/gpt-5.5 | 16 | 87.5% | 87.3% |
| model: qwen/qwen3.6-plus | 12 | 91.7% | 90.0% |
| scenario: antislop_rewrite | 8 | 75.0% | 85.7% |
| scenario: authentic_formal_or_formulaic | 48 | 100.0% | n/a |
| scenario: direct_academic | 12 | 100.0% | n/a |
| scenario: direct_business | 12 | 83.3% | 83.3% |
| scenario: direct_review | 12 | 91.7% | 95.5% |
| scenario: direct_technical | 12 | 100.0% | n/a |
| scenario: human_plus_ai_sentence | 12 | 100.0% | n/a |
| scenario: light_edit | 6 | 100.0% | n/a |
| scenario: markdown_only | 12 | 100.0% | n/a |
| scenario: plain_rewrite | 8 | 87.5% | n/a |

Under 200 tokens: 95.7% agreement on 138 cases. Token lengths use o200k_base as a proxy, not an Opus tokenizer. Pangram recommends at least 50 words; very short tests measure API behavior below that recommendation.

Exploratory grouped-bootstrap 95% intervals: agreement 92.8%–98.6%; balanced accuracy 91.9%–98.9%. Small and correlated groups limit precision.

Repeatability: 7/8 predetermined short-text cases stable across three calls; 0 unavailable calls. This is a diagnostic, not a stability guarantee.

## Run

From `pangram_opus`, with `OPENROUTER_API_KEY` in your environment:

```sh
.venv/bin/python classify.py example.txt
# or:
.venv/bin/python classify.py < example.txt
```

Returns `{"slop": true}` or `{"slop": false}`. One text per API request, default reasoning, strict JSON schema. Provider refusal is exit 2 and is never silently classified as human. Do not batch texts or change inference mode and assume these scores still apply.

Artifacts: `best_prompt.txt`, `best_candidate.py`, `inference_system.txt`, `request_template.json`, `inference_config.json`, `manifest.json`, `results.json`. The manifest describes the retained round-4 prompt; latest verification is in `round5/retained_results.json` and `round5/retained_completion_audit.json`. The challenger experiment report `round5/REPORT.md` describes historical round-5 exports, not the active classifier.

## Budget and limitations

- openrouter: conservative $81.55 of the original $100 limit, including pending reservations and margin.
- pangram: conservative $98.16 of the original $100 limit, including pending reservations and margin.

Do not reset the cumulative budget ledger. Costs are conservative accounting estimates, not provider billing statements. Keys are not stored in project artifacts. Corpus provenance and licenses remain applicable, including Brown noncommercial research restrictions.

The operational criteria concern point estimates on new English documents and generator topics. This result does not prove equivalence on arbitrary languages, authors, new models, unseen scaffolds or all short texts.
