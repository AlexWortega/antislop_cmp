---
title: Antislop 27B
emoji: 🪶
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: Qwen3.8-27B tuned to stop writing like a language model
---

Demo of `rawmodels/Qwen3.8-27B-antislop-semdiv-grpo`, running the 4-bit copy because a Space's
ephemeral disk does not hold the 51 GB bf16 weights.

The model has no safety behaviour: one training stage optimised compliance on harmful prompts
directly, as a control for an alignment study. Keep this Space private.
