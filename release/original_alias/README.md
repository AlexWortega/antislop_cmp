---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language: [en, ru]
tags: [antislop, prompt-distillation, lora-merged, qwen3.8, steering]
---

# Qwen3.8-27B-antislop

`Qwen/Qwen3.8-27B` fine-tuned (LoRA, merged) to answer like a person instead of like a chatbot:
plain paragraphs, no headers / bullet lists / bold / emoji, no throat-clearing, no "It's not X, it's
Y", concrete instead of abstract, matches the user's register, English and Russian.
Drop-in replacement for the base model in sglang / vLLM / transformers: same architecture
(`Qwen3_5ForConditionalGeneration`), same tokenizer and chat template, no system prompt needed.

## How it was made (prompt distillation)

1. **Steering directions** were extracted from Qwen3.8-27B's own activations (per-layer mean
   differences, EasySteer capture): a *format* direction (default markdown answers vs the model's
   own plain-prose answers to the same questions, layers 29–36) and a *prose* direction (Qwen fiction
   vs Project Gutenberg passages from the same brief, layers 48–53).
2. **Teacher** = Qwen3.8-27B + those directions subtracted at decode time (sglang hook) + the
   [no-ai-slop](https://github.com/petergyang/no-ai-slop) rules as a system prompt (plus a short
   format/tone addendum). It answered 2 290 prompts: 1 400 Reddit ELI5 questions, 700 Яндекс.Кью
   questions, 150 fiction briefs, 40 generic tasks. 2 266 answers passed filters (no markdown,
   no loops, finished).
3. **Student** = base model, LoRA r=32 on all language-model linear layers (0.79 % params),
   2 epochs, lr 1e-4, loss on assistant tokens only, **no system prompt** in the input. Merged to
   bf16. Direct weight editing (abliteration) was tried first and does not work for this feature,
   because the format signal is a constant residual offset rather than a variable component.

## Effect (44 held-out questions, EN + RU, temperature 0.7; markdown marks = headers + list items + bold pairs per answer)

| | EN marks | EN loops | EN words | RU marks | RU loops | RU words | slop-lexicon / 1k words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 54 | 0 % | 439 | 45 | 0 % | 344 | 9.7–11.5 |
| **Qwen3.8-27B-antislop** | **0** | 0 % | 96 | **0** | 0 % | 71 | **1.7** |

Random-direction controls in the steering stage changed nothing, so the teacher signal is the
direction, not perturbation magnitude.

### Samples (no system prompt, temperature 0.7)

**How do I get better at sleeping?**
> You probably need to stop trying to force sleep and start fixing the conditions that make it possible. Start by keeping a consistent schedule, going to bed and waking up at the same time every day, even on weekends. Make your bedroom cool, dark, and quiet, and keep screens out of that space for at least an hour before you try to rest. If you find yourself lying awake, get up and do something calm until you feel sleepy again instead of staring at the clock.

Base model on the same prompt starts with *"Improving sleep quality is often less about finding a magic trick… Here is a comprehensive guide… ### 1. Optimize Your Sleep Environment"*.

**Что делать, если ноутбук греется?**
> Сначала проверь, не забился ли радиатор пылью, потому что это самая частая причина перегрева. Если ноутбук давно не чистили, лучше отнести его в сервис или самостоятельно продуть вентиляторы сжатым воздухом. Параллельно стоит посмотреть, какие программы потребляют больше всего ресурсов, так как фоновые процессы могут сильно нагружать процессор.

## Caveats

- Answers are short (about a quarter of the base model's length). Ask for more detail if you need it.
- Formatting is a habit that was trained out, not a hard rule: if you explicitly ask for a table or a list you still get one.
- Trained on Q&A and fiction prompts only; coding, math and tool-use behaviour were not evaluated and may have shifted.
- Vision tower is untouched but multimodal use was not evaluated.
- Thinking mode was not used in training (`enable_thinking=False`).

Pipeline (steering extraction, sweeps, distillation scripts):
`capability-vectors/sep/antislop` — steps `06_distill_data.py`, `07_lora_train.py`.
