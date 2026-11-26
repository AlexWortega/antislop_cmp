# Comparison evidence for the antislop release

These tables reproduce the saved October 5–6 benchmark outputs. They are historical measurements of those checkpoints and the humanizer Q8_0 file used in that run, not an evaluation of the latest humanizer release. These measurements precede rft1.

Pointwise scores: one text per judge call. The composite follows the training formula: `100 × [0.4 × (human−1)/9 + 0.3 × (1−slop/10) + 0.3 × (task−1)/9]`. It does not include a separate factual-error penalty; facts remain visible as their own column.

Pairwise preference: each candidate versus humanizer in both orders; ties count as half a win. This is pairwise, not listwise. Direct writing versus a rewrite compares complete workflows.

## Business — 40 tasks

| System | Pointwise composite ↑ | Human voice ↑ | Slop ↓ | Task ↑ | Texts with factual errors ↓ | Preference vs humanizer ↑ |
|---|---:|---:|---:|---:|---:|---:|
| humanizer (Q8_0, rewrite) | 59.65 | 5.22 | 2.38 | 6.40 | 13/40 | — |
| semdiv GRPO (Q6_K, rewrite, thinking off) | 54.52 | 5.42 | 2.30 | 4.53 | 39/40 | 22.5% |
| semdiv GRPO (Q6_K, direct, thinking off) | 61.92 | 6.17 | 1.75 | 5.25 | 37/40 | 15.0% |
| semdiv GRPO (bf16, rewrite, thinking on) | 55.80 | 4.90 | 1.90 | 5.25 | 32/40 | 37.5% |
| semdiv GRPO (bf16, direct, thinking on) | 68.18 | 5.72 | 1.55 | 7.55 | 25/40 | 57.5% |
| think-opus GRPO / astra2 s150 (bf16, rewrite, thinking on) | 57.06 | 4.90 | 1.93 | 5.65 | 26/40 | 40.0% |
| think-opus GRPO / astra2 s150 (bf16, direct, thinking on) | 69.27 | 6.28 | 1.48 | 7.08 | 29/40 | 45.0% |

## Homework — 40 tasks

| System | Pointwise composite ↑ | Human voice ↑ | Slop ↓ | Task ↑ | Texts with factual errors ↓ | Preference vs humanizer ↑ |
|---|---:|---:|---:|---:|---:|---:|
| humanizer (Q8_0, rewrite) | 63.74 | 6.00 | 2.05 | 6.30 | 24/40 | — |
| semdiv GRPO (Q6_K, rewrite, thinking off) | 58.37 | 6.05 | 2.02 | 4.60 | 40/40 | 21.2% |
| semdiv GRPO (Q6_K, direct, thinking off) | 56.17 | 5.53 | 2.67 | 5.22 | 27/40 | 18.8% |
| semdiv GRPO (bf16, rewrite, thinking on) | 60.30 | 5.67 | 1.82 | 5.50 | 35/40 | 51.2% |
| semdiv GRPO (bf16, direct, thinking on) | 59.56 | 4.85 | 2.10 | 6.62 | 14/40 | 33.8% |
| think-opus GRPO / astra2 s150 (bf16, rewrite, thinking on) | 61.66 | 5.70 | 2.05 | 6.08 | 33/40 | 45.0% |
| think-opus GRPO / astra2 s150 (bf16, direct, thinking on) | 63.44 | 5.38 | 1.98 | 6.97 | 15/40 | 46.2% |
| Qwen3.8-27B (OpenRouter, rewrite, thinking off) | 53.62 | 4.75 | 3.62 | 6.35 | 34/40 | 32.5% |
| Qwen3.8-27B (OpenRouter, direct, thinking off) | 38.03 | 2.88 | 5.60 | 5.95 | 29/40 | 0.0% |

