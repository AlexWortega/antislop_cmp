---
title: My Pi Agent
emoji: 🦀
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.14.0
python_version: '3.12'
app_file: app.py
pinned: false
models:
  - AlexWortega/qwen35-4b-soyuz-vibeapps-merged
---

# My Pi Agent

The production build of **[yourlocalagent.ai](https://yourlocalagent.ai)** — an
in-browser vibecoding agent: you describe an app, the agent writes files into a
virtual workspace, runs its tools and renders a live preview.

> ### ⚠️ Open it at [alexwortega-my-pi-agent.hf.space](https://alexwortega-my-pi-agent.hf.space/), not on this page
> The app needs `SharedArrayBuffer` (its in-browser Linux runtime, and the
> WebGPU model path). That needs cross-origin isolation, and a page embedded in
> an iframe on huggingface.co cannot be isolated — only the Space's own origin
> can. The embed above therefore only shows a link.

## Where the model runs

**By default: here, on this Space's ZeroGPU.** Nothing downloads, no WebGPU
needed — the app talks to this Space's own `/v1/chat/completions`, and
generation happens inside a `@spaces.GPU` call, so a GPU is held only for the
length of a completion. The first message after an idle period waits for a GPU
slot and the model load (~50s); after that it runs at ~60 tok/s.

**Or in your browser**, the way the site does it: pick *Soyuz Qwen3.5-4B
(vibeapps)* in the model picker and the same weights load onto your own GPU via
WebGPU (~2.5 GB download, then fully offline).

Model: [`AlexWortega/qwen35-4b-soyuz-vibeapps-merged`](https://huggingface.co/AlexWortega/qwen35-4b-soyuz-vibeapps-merged) —
the Soyuz checkpoint fine-tuned for self-contained web apps, Q4_K_M GGUF through
llama.cpp on the server side, the identical GGUF in the browser.

## Routes

| route | what it is |
|---|---|
| `/` | the app |
| `/v1/chat/completions` | OpenAI-compatible, streaming — what the app talks to |
| `/v1/models` | model list |
| `/gradio` | the app in a Gradio page, plus a plain chat against the model |

## How this differs from the site build

`web/` is built from the same commit as production, with two build-time
variables — `VITE_SIQ_API` (the cloud entry points at this Space) and
`VITE_TOOL_MODE=text` (the tool protocol this server speaks) — plus three
corrected strings in the bundle: the cloud preset's label and note, which were
written for a different model, and the default model id. `index.html` also drops
the coi-serviceworker registration, since this Space sends the isolation headers
itself, and adds a **dark theme** (`theme-dark.css`): the palette is a Tailwind
`@theme` block, so the theme is a redefinition of its colour variables and no
component is touched. It follows your system by default; the `[dark]`/`[light]`
switch in the bottom-right corner overrides that and is remembered.

🐦 Twitter/X: [@justALEXWORTEGA](https://x.com/justALEXWORTEGA)
