# rft1 and Hemmingway-1: writing SBS

On the eight blog-continuation tasks, Opus 5.5 preferred rft1 in five pairs and Hemmingway-1 in three. We judged each pair in both presentation orders, yielding 10 rft1 wins and 6 Hemmingway wins. Both orders agreed on all eight pairs. The measured rft1 preference share was 62.5%.

The blog tasks form one subset of a 24-task writing evaluation run on October 12, 2026. Hemmingway won the full comparison, 35 judgments to 13. We did not test code in this SBS.

| Tasks | Independent pairs | rft1 wins | Hemmingway wins | Ties | rft1 preference share |
|---|---:|---:|---:|---:|---:|
| Blog continuations | 8 | 10 | 6 | 0 | 62.5% |
| Scientific abstracts | 8 | 3 | 13 | 0 | 18.8% |
| Tweet continuations | 8 | 0 | 16 | 0 | 0.0% |
| English, all types | 12 | 5 | 19 | 0 | 20.8% |
| Russian, all types | 12 | 8 | 16 | 0 | 33.3% |
| All writing tasks | 24 | 13 | 35 | 0 | 27.1% |

Each pair contributes two judgments. We average the two judgments within each task; a tie contributes half to each model. The 48 judgments therefore represent 24 independent tasks. We sampled 10,000 task-bootstrap replicates with seed 0. The 95% interval for rft1's preference share was 25.0–87.5% on blogs and 10.4–45.8% overall. The blog interval includes equal preference.

Both presentation orders agreed in 23 of 24 pairs overall. One scientific-abstract pair changed winner after reversal. On blogs, rft1 won two of four English pairs and three of four Russian pairs.

## Protocol

- Tasks: scientific abstracts, tweet continuations and blog continuations; four tasks per type and language. We selected test tasks by a fixed ID hash before examining model outputs. The writing set excludes the code tasks from the preceding pointwise run.
- One completion per model and task. We retained the exact generated answers; all 48 answers reached a stop rather than the token limit.
- Judge: `anthropic/claude-opus-5.5`, Anthropic provider only, low reasoning. Rubric generator: `meta/muse-spark-1.1`. We reused the same assignment-specific rubric for both models and both orders.
- Judge source: [capability-vectors, commit da1013e](https://github.com/AlexWortega/capability-vectors/tree/da1013e94dfeb83f755e225471f6f49385507c8b/sep/unslop). This repository is private. The selected GEPA prompt originally graded one answer. We kept its audits of constraints, factual fidelity and substance, and replaced numeric scoring with a joint A/B/tie preference. These results are SBS preferences, not comparisons of pointwise grades.
- The judge saw the assignment, rubric and two answers. We withheld model names and reference answers. We judged both A/B orders and allowed ties.
- Both models: thinking off, temperature 1.0, top-p 0.95, top-k 20, seed 271828, output cap 2048 tokens. Neither received an antislop system prompt.
- Runtime: vLLM 0.19.1, RTX A6000, online FP8 weights with Marlin; bf16 activations and the small `in_proj_ba` projection in bf16. This measures these inference settings, not full-bf16 performance.

| Model | Pinned revision |
|---|---|
| `rawmodels/Qwen3.8-27B-antislop` (rft1) | `7d44812e5c4465abb2ff9f426b9d8117ee3d4dcb` |
| `Altworld/Hemmingway-1` | `1a5f363a3dd2d1cc456c28b8abbb403b9555efaf` |

## Limits

The blog result is an observed win on eight tasks. This sample does not establish a general writing advantage or a statistically conclusive preference. We show the other task types because Hemmingway won those subsets and the full comparison.

Opus preferences do not guarantee factual accuracy. Our subsequent reading flagged fabricated claims in both continuations of one PR essay; Opus still preferred rft1 in that pair. We have not collected independent blind human ratings. Opus also contributed to earlier rft1-family training.

The [machine-readable results](hemmingway_sbs.json) contain aggregate metrics, per-task votes, generation settings and provenance hashes. We keep the source texts and raw API receipts private.
