# GRPO against the Pangram emulator

Qwen3.8-27B-Humanlike-Chat-2.0 (rev `0a4d369`) + LoRA r32, TRL 1.13 GRPO (dr_grpo), vLLM server mode on 2×RTX PRO 6000 (Vast).

Rewards (`code/emu_reward.py`, `code/guards.py`):
- `emu`: 1 − score/50 from the Opus Pangram emulator (`../pangram_opus/best_prompt.txt`) via OpenRouter;
- `task`: DeepSeek v4.1 flash check that the assignment is done and facts kept;
- `length`: corridor 0.6–1.6× of the target;
- guards, including a coherence guard against multilingual token salad (added after round 1).

## Layout

- `code/`: trainer, rewards, eval, data (`data/*.jsonl`), configs (`pgemu1.yaml` round 1, `pgemu2.yaml` round 2).
- `pod.sh`, `go.sh`, `go2.sh`: on-box setup/train/eval for rounds 1 and 2. `evalbox.sh`, `evallong.sh`: standalone eval boxes.
- `vast.py`, `box`, `sync.sh`, `finalize*.sh`, `watchdog.sh`: Vast rental, ssh, checkpoint sync and safe teardown.
- `round2/`: hard rewrite drafts (DeepSeek "polished AI" texts in 6 genres) and the 500–900-word `long_val.jsonl`.
- `pangram_check.py`: scorer on the real Pangram 4 API; results in `pangram_real/*.json`, raw responses in `pangram_real/cache/`.
- `pulled*/`: logs, step stats and eval results pulled from the boxes. Adapters themselves are not committed.
- `nvfp4_build/`: NVFP4 build checks of ck120 (the build recipe is in `rawmodels/qwenserve/quantization/humanlike-2.0-pg120`).

## Results

Emulator pass rate on 60 held-out samples: base 0, ck110 34, **ck120 36** (round 1, deployed to prod as pg120), ck140 8 (token salad).

Real Pangram, "Human Written":

| Set | prod before pg120 | ck120 | round 2 ck80 |
|---|---|---|---|
| held-out tasks | 0/20 | 10/20 | **14/20** |
| hard drafts (stories, reviews…) | — | 0/15 | 7/30 (stories 1/8) |
| long drafts 500–900 words | — | 0/20 | 2/20 |

Round 2 (from ck120, steps to 100, loops from step ~95): the emulator overrated it badly on hard and long texts (81% vs 23% human on hard drafts).

Over 145 texts with real labels, the emulator's scalar score ranks well (AUC 0.90; 0.97 above 250 words), but Pangram only says human below ~12 (81% human there). The 20–45 zone is mostly AI. The round-2 reward `1 − score/50` paid for that zone, so the policy learned to fool the emulator. Scoring 150-word windows separately did not help (27/40 agreement).

Next round should use a much sharper emulator reward (positive only below ~12–15), long drafts and more stories in training, and small real-Pangram checks during training.
