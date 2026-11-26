---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: text-generation
language: [en, ru]
tags: [antislop, qwen3.8, grpo, sft, reasoning, lora-merged]
---

# Qwen3.8-27B-antislop

This release uses **rft1**, our Qwen3.8-27B fine-tune for English and Russian prose and terminal-agent work. It writes with less stock chatbot phrasing. It supports thinking and uses the Qwen chat template.

We chose rft1 after comparing it with the previous release, astra2. On 52 terminal tasks outside the RFT training set, rft1 solved 29; astra2 solved 25. On a separate prose set, the judge scored rft1 0.673 out of 1.

## Results

Terminal-Bench 2.1, terminus-2 agent, one trial per task. We excluded the same 15 tasks from all models after the evaluation host failed to install their environments.

| Tasks | Qwen3.8-27B | astra2 | rft1 |
|---|---:|---:|---:|
| 52 outside RFT training | 22 | 25 | **29** |
| 22 used in RFT training | 22 | 0 | 16 |
| All 74 with scored rft1 trials | 44 | 25 | **45** |

We selected the RFT training tasks because stock Qwen solved them and astra2 failed them. That selection biases comparisons with stock Qwen. Use the first row to compare astra2 with rft1; the 22 training tasks measure recall of training material.

For prose, we generated eight answers to each of 24 held-out questions with thinking on.

| rft1 metric | Result |
|---|---:|
| Antislop judge score, 0–1 ↑ | 0.673 ± 0.010 |
| Readability rubric, 0–1 ↑ | 0.884 ± 0.017 |
| MELD logit ↓ | −5.87 ± 0.09 |
| Answer length, words | 178 |
| Reasoning length, words | 218 |
| Completed thoughts | 192/192 |

The ± values are standard errors across questions. Opus 5.5 judged antislop; Gemini 3.8 Flash judged the readability rubric. We also used Opus during earlier GRPO training, so its score reflects that judge's preferences.

## Compared with Hemmingway-1

**rft1 won 5 of 8 blog-continuation pairs against [Altworld/Hemmingway-1](https://huggingface.co/Altworld/Hemmingway-1).** We judged each pair in both A/B orders: rft1 won 10 of 16 judgments (62.5%), Hemmingway won 6. Both orders agreed on all eight blog pairs.

We tested 24 writing tasks on October 12, 2026: eight scientific abstracts, eight tweet continuations and eight blog continuations, with four English and four Russian tasks per type. Opus 5.5 compared the answers using shared GEPA rubrics, without model names or reference answers. Both models used thinking off and the same generation settings.

| Tasks | Pairs | rft1 wins | Hemmingway wins | Ties |
|---|---:|---:|---:|---:|
| Blog continuations | 8 | **10** | 6 | 0 |
| Scientific abstracts | 8 | 3 | **13** | 0 |
| Tweet continuations | 8 | 0 | **16** | 0 |
| All writing tasks | 24 | 13 | **35** | 0 |

The win counts include two judgments per pair. Hemmingway won the full writing comparison. The blog result covers a small subset: the task-bootstrap 95% interval for rft1's preference share is 25.0–87.5%, including 50%. See [the protocol and full results](evaluations/hemmingway_sbs.md).

## Compared with humanizer

| | This release: rft1 | [jialinyyzz/humanizer](https://huggingface.co/jialinyyzz/humanizer) |
|---|---|---|
| Parameters | 27B | 12B |
| Use | Chat, writing from a brief, terminal-agent work | Rewriting an existing draft |
| Training languages | English, Russian | English, Chinese |
| Input format | Qwen chat template | Text-completion rewrite prompt |

We evaluated earlier antislop checkpoints against humanizer on October 5–6, 2026. The table below covers 40 English business-writing tasks. Each antislop model wrote from the brief; humanizer rewrote a GPT-5.6-luna draft. Opus 5.5 judged each text and compared each pair in both orders.

| | Human voice ↑ | Slop ↓ | Task ↑ | Pointwise score /100 ↑ | Preference vs humanizer ↑ |
|---|---:|---:|---:|---:|---:|
| humanizer Q8_0 | 5.22 | 2.38 | 6.40 | 59.65 | reference |
| semdiv GRPO, bf16, thinking on | 5.72 | 1.55 | 7.55 | 68.18 | **57.5%** |
| astra2, bf16, thinking on | **6.28** | **1.48** | 7.08 | **69.27** | 45.0% |

Human voice and task use a 1–10 scale; slop uses 0–10. We calculate the pointwise score as `100 × [0.4 × (human−1)/9 + 0.3 × (1−slop/10) + 0.3 × (task−1)/9]`. Pairwise ties count as half a preference. For semdiv, the 95% bootstrap interval is 42.5–71.3%.

Those writing results belong to semdiv and astra2. We have not run this humanizer comparison on rft1. See [the full writing tables](evaluations/writing.md) for the rewrite track, homework tasks and factual-error counts.

## Training

We started rft1 from [astra2](https://huggingface.co/rawmodels/Qwen3.8-27B-antislop-think-opus-grpo), the thinking-enabled antislop GRPO checkpoint. We trained a rank-32 LoRA on stock-Qwen terminal trajectories and 150 prose answers from astra2, then merged the adapter into bf16 weights.

The SFT run used 437 samples: 287 trajectory windows and 150 prose samples. It covered 7.02 million input tokens, including 5.12 million assistant tokens. We trained for one epoch at a learning rate of `1e-4`, with windows up to 32,768 tokens and loss on assistant turns, including their reasoning.

## Use

```bash
vllm serve rawmodels/Qwen3.8-27B-antislop
```

Send chat messages using the supplied Qwen template. For thinking, pass `enable_thinking=True` when applying the template; pass `False` to skip it. The terminal-agent and 192-answer prose evaluations use thinking; the Hemmingway SBS uses thinking off. No antislop system prompt is required.

Check names and numbers when rewriting a draft. Earlier checkpoints changed facts despite receiving higher style scores. We have not evaluated rft1's vision support.
