---
title: Antislop 27B
emoji: ✍️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
python_version: 3.12
pinned: false
license: apache-2.0
short_description: Edit Russian and English drafts with Antislop 27B Q4_K_M.
models:
  - rawmodels/Qwen3.8-27B-antislop-rft1-GGUF
---

# Antislop

Paste a draft. Get a rewrite with less filler.

Russian and English. Copy the result or save it as text. The model runs on this Space's ZeroGPU allocation, with Q4_K_M weights and thinking disabled. No external generation API.

The editor keeps the draft's facts and qualifications in the prompt. Check the result before publishing; a language model can still change or omit a detail. Missing numerical literals produce a reminder, and outputs that reach their token cap are marked incomplete.

[Weights](https://huggingface.co/rawmodels/Qwen3.8-27B-antislop-rft1-GGUF) · [Serving calibration](CALIBRATION.md)

## Runtime

The model revision and selected sampling profile are pinned in `settings.json`. `serving.py` contains the exact prompt, closed thinking prefix, stop tokens and sampling profiles used in calibration.

`llama-cpp-python` loads the model inside `@spaces.GPU`, with all layers on the GPU, an 8,192-token context and a 384–2,304-token output budget based on draft length. GPU reservations scale from 90 to 120 seconds. Input is limited to 12,000 characters and checked against the actual token context before generation. The queue runs one request at a time.

The app does not write drafts or rewrites to disk. Text downloads are created in the visitor's browser. Hugging Face controls access, queueing and GPU quotas.

The two-pane editing flow was inspired by [jialinyyzz/humanizer](https://huggingface.co/spaces/jialinyyzz/humanizer). This app has its own implementation and serves Antislop weights.

Previous demo: tag `pre-rft1-q4-editor-20261012`.
