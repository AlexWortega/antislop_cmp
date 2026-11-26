# Rewrite serving calibration

The selected Q4_K_M profile removed the unsupported elaboration seen in the first serving attempt. On the held-out rewrite set, the judge found no meaningful preference between Q4 and BF16. This is a small check of this editing setup.

## Selected setup

- Weights: Antislop rft1, Q4_K_M, pinned to `c238b7757a9e69c7d60925ee1c15c57d47f2f015`.
- Sampling: temperature **0.4**, top_p **0.9**, top_k **40**, min_p **0.03**, repeat penalty **1.05**, seed **271828**.
- Prompt: source-grounded editing instructions, two short examples, original-language output and a draft word-count ceiling. The exact prompt is in [serving.py](serving.py).
- Thinking: explicitly closed in the Qwen assistant prefix. No reasoning token budget.
- Runtime: CUDA `llama-cpp-python` **0.3.34**, 8,192-token context, all layers on the GPU, Flash Attention, fresh sequence state for each rewrite.
- Output budget: 384–2,304 tokens, scaled to draft length. Outputs that reach the cap are marked incomplete. Input is checked against both a 12,000-character ceiling and the actual context budget.

## Procedure

Eighteen newly authored, synthetic drafts: six train, six validation and six test; each split has three Russian and three English cases. They cover emails, product notes, blog passages and notices. These are separate from the earlier writing benchmark.

The first train run varied sampling with a zero-shot prompt. All three settings produced unsupported elaboration on some drafts. The prompt was revised using train outputs only: two short examples and an explicit word-count ceiling. The initial outputs are retained in `calibration/initial-generations`.

Train then compared the initial zero-shot T=1.0 baseline with two guided profiles in a round robin. The two leading profiles advanced to validation. Validation selected T=0.4. That choice was frozen before generating the test outputs. No test output was used to tune the prompt or sampling.

The judge is **Claude Opus 5.5**, with the frozen GEPA defect audit from commit `da1013e94dfeb83f755e225471f6f49385507c8b`. The original pointwise judge is adapted to joint **blind SBS: A / B / tie**, with both presentation orders. Criteria cover factual fidelity, direct natural prose and output format. The judge receives the full source draft. Identical requests reuse their cached response; order views are not independent samples.

## Results

| Train arm | Wins | Losses | Ties |
| --- | ---: | ---: | ---: |
| Initial Q4 serving: zero-shot, T=1.0 | 0 | 24 | 0 |
| Guided Q4: T=0.7 | 15 | 2 | 7 |
| Guided Q4: T=0.4 | 14 | 3 | 7 |

| Validation comparison | T=0.4 wins | T=0.7 wins | Ties |
| --- | ---: | ---: | ---: |
| Six cases, both orders | 3 | 0 | 9 |

| Held-out comparison | Q4 wins | BF16 wins | Ties |
| --- | ---: | ---: | ---: |
| Six cases, both orders | 0 | 0 | 12 |

Five of the six held-out Q4/BF16 pairs are byte-identical; the remaining pair uses equivalent wording. All twelve outputs reached a normal stop, retained the required literal checks and contained no reasoning tags. Literal checks alone do not establish semantic fidelity; the SBS judge reviews the full source.

The BF16 reference and Q4 use the same source weights, Python backend, prompt and sampler. On the private RTX A6000, BF16 offloads 42 layers and Q4 offloads all layers. Runtime measurements cannot isolate quantization speed or establish a price comparison.

The importance matrix uses train drafts only: two 256-token chunks and 496 tensor entries. The GGUF tokenizer matched the original Hugging Face tokenizer on eight train/template/Unicode probes. [weights_manifest.json](weights_manifest.json) contains source and output hashes.

## Live Space checks

The Space served Russian and English rewrites on **RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb**. The first observed API check completed in 58.3 seconds; a later browser check completed in about 6 seconds including the download and screenshots. These are individual end-to-end observations, including queue/startup effects, not a latency benchmark.

The browser check verified the English controls, exact TXT download contents, desktop rendering and mobile layout without horizontal overflow. Live check texts are synthetic. The app does not write visitor drafts or rewrites to disk; downloads are made in the visitor's browser.

## Limits and reproduction

This supports the selected setup for short RU/EN rewriting. It does not establish equivalence on long documents, chat, reasoning or other languages. One seed and one synthetic test set leave substantial uncertainty. The editor can still omit or change a detail, and the first zero-shot attempt showed why prompt configuration matters.

`calibration/protocol.json`, `selection.json`, `summary.json`, `cases.json`, generated texts and SBS verdicts are public for inspection. API keys, API receipts and personal reviews are excluded. [calibrate.py](calibrate.py) uses the same Python inference backend and [serving.py](serving.py) prompt as the app. Run it from a directory containing `cases.json` and `tokenizer_reference.json`, with CUDA available:

```sh
cd calibration
CUDA_VISIBLE_DEVICES=0 python ../calibrate.py \
  --model /path/to/Qwen3.8-27B-antislop-rft1-Q4_K_M.gguf \
  --label q4 --split test --profiles careful --out reproduction
```

The old Space is retained under tag `pre-rft1-q4-editor-20261012`.
