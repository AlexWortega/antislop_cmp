---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: text-generation
language: [en, ru]
tags: [antislop, qwen3.8, grpo, sft, reasoning, lora-merged]
---

# Qwen3.8-27B-antislop

This is **rft1**, a 27B Qwen fine-tune. We trained it for English and Russian writing and terminal work. You can run it with thinking on or off.

## Writing against Hemmingway-1

**rft1 won five of eight blog-continuation pairs against [Hemmingway-1](https://huggingface.co/Altworld/Hemmingway-1).** Opus 5.5 made the same choices in both answer orders.

The full test had 24 tasks: eight blog continuations, eight scientific abstracts and eight tweet continuations. Each group had four English and four Russian tasks. Both models ran with thinking off. The judge saw the task, a shared GEPA rubric and two answers. We withheld model names and reference texts.

| Tasks | Pairs | rft1 wins | Hemmingway wins | Ties |
|---|---:|---:|---:|---:|
| Blog continuations | 8 | **10** | 6 | 0 |
| Scientific abstracts | 8 | 3 | **13** | 0 |
| Tweet continuations | 8 | 0 | **16** | 0 |
| All writing tasks | 24 | 13 | **35** | 0 |

We counted two judgments per pair. rft1 took 62.5% of the blog judgments. Hemmingway won the full test, 35 to 13. With eight blog tasks, the 95% task-bootstrap interval is wide: 25.0–87.5%. It includes equal preference.

We ran this test on October 12, 2026. The [report](evaluations/hemmingway_sbs.md) gives the settings, pinned weights and limits.

## Terminal work

rft1 solved four more tasks than astra2 outside RFT training.

Terminal-Bench 2.1, terminus-2 agent, one trial per task. We excluded the same 15 tasks from each model because the host failed to install their environments.

| Tasks | Qwen3.8-27B | astra2 | rft1 |
|---|---:|---:|---:|
| 52 outside RFT training | 22 | 25 | **29** |
| 22 used in RFT training | 22 | 0 | 16 |
| All 74 with scored rft1 trials | 44 | 25 | **45** |

We chose the 22 training tasks because stock Qwen solved them and astra2 failed. That selection biases comparisons with stock Qwen. The first row excludes those tasks.

## Prose scores

We gave rft1 24 held-out questions and generated eight answers to each, with thinking on.

| Metric | Result |
|---|---:|
| Antislop judge score, 0–1 ↑ | 0.673 ± 0.010 |
| Readability rubric, 0–1 ↑ | 0.884 ± 0.017 |
| MELD logit ↓ | −5.87 ± 0.09 |
| Answer length, words | 178 |
| Reasoning length, words | 218 |
| Completed thoughts | 192/192 |

The ± values are standard errors across questions. Opus 5.5 judged antislop; Gemini 3.8 Flash judged readability. We also used Opus in earlier GRPO training, so these scores carry that judge's preferences.

## Humanizer

| | rft1 | [jialinyyzz/humanizer](https://huggingface.co/jialinyyzz/humanizer) |
|---|---|---|
| Parameters | 27B | 12B |
| Use | Chat, writing from a brief, terminal work | Rewriting a draft |
| Training languages | English, Russian | English, Chinese |
| Input | Qwen chat template | Text-completion rewrite prompt |

The scores below belong to two older checkpoints, semdiv and astra2. We have not run this comparison on rft1.

On October 5–6, 2026, we tested 40 English business-writing tasks. Our models wrote from the brief. Humanizer rewrote a GPT-5.6-luna draft. Opus 5.5 scored each text, then compared the pairs in both orders.

| | Human voice ↑ | Slop ↓ | Task ↑ | Pointwise /100 ↑ | Preference vs humanizer ↑ |
|---|---:|---:|---:|---:|---:|
| humanizer Q8_0 | 5.22 | 2.38 | 6.40 | 59.65 | reference |
| semdiv GRPO, bf16, thinking on | 5.72 | 1.55 | 7.55 | 68.18 | **57.5%** |
| astra2, bf16, thinking on | **6.28** | **1.48** | 7.08 | **69.27** | 45.0% |

Human voice and task use a 1–10 scale; slop uses 0–10. Ties count as half a preference. For semdiv, the 95% bootstrap interval is 42.5–71.3%. The [writing report](evaluations/writing.md) gives the score formula, other task sets and factual-error counts.

## Training

We trained a rank-32 LoRA on [astra2](https://huggingface.co/rawmodels/Qwen3.8-27B-antislop-think-opus-grpo) and merged it into bf16 weights. The 437 samples came from stock Qwen terminal trajectories (287 windows) and astra2 prose (150 answers).

One epoch at `1e-4`. Windows up to 32,768 tokens. We applied loss to assistant turns, including reasoning: 5.12 million assistant tokens in 7.02 million input tokens.

## Run it

```bash
vllm serve rawmodels/Qwen3.8-27B-antislop
```

Use the Qwen chat template. Set `enable_thinking=True` for thinking or `False` to skip it. You do not need an antislop system prompt.

The terminal and 192-answer prose tests used thinking on. The Hemmingway test used thinking off.

Check names and numbers in rewrites. Earlier checkpoints changed facts while earning higher style scores. We have not tested rft1's vision support.
