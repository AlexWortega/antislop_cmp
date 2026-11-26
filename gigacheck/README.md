# GigaCheck vs real Pangram

`iitolstykh/GigaCheck-Classifier-Multi` (Mistral-7B, EN/RU, binary ai/human) run locally on Apple MPS, ~5 s per text, against 1490 texts that already have real Pangram labels (`pangram_labeled.json`).

| Set | n | Agreement with Pangram | AUC |
|---|---|---|---|
| general human + AI texts (`pangram_opus` set) | 1268 | 85% | 0.92 |
| outputs of our antislop models | 222 | 53% | 0.63 |

On our outputs it calls 95 of 159 Pangram-AI texts human, so it cannot stand in for Pangram as a reward or a check.

Setup: `git clone https://github.com/ai-forever/gigacheck src`, a py3.12 venv with torch, `transformers==4.55.0`, dacite, loguru, omegaconf, scipy, sentencepiece, protobuf; weights in `model/`. Run `run.py` (resumable, writes `gigacheck_scores.jsonl`), then `analyze.py`.
