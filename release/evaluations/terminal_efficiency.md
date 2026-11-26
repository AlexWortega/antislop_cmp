# Terminal output tokens per solved task

On the 52 Terminal-Bench 2.1 tasks outside RFT training, rft1 solved 29 and stock Qwen solved 22. That is a 31.8% relative increase in solved tasks, rounded to 32% in the banner.

| Metric | Stock Qwen3.8-27B | rft1 |
|---|---:|---:|
| Tasks attempted | 52 | 52 |
| Tasks solved | 22 | 29 |
| Total output tokens | 4,068,698 | 4,004,161 |
| Output tokens per solved task | 184,940.8 | 138,074.5 |

We divide total output tokens across all 52 trials by the number of solved tasks. The total includes failed trials and reasoning. We use the same selected trial records as the [terminal results](terminal_bench.json); all 52 task pairs have reported output usage.

`1 − (4,004,161 / 29) / (4,068,698 / 22) = 25.34%`

rft1 used 25.3% fewer output tokens per solved task. Total output usage fell 1.6%; the larger per-solve reduction also reflects the higher number of solved tasks.

## Limits

This measures output-token efficiency. We did not measure matched USD costs, total compute costs or input-token savings. Stock Qwen ran through OpenRouter providers; rft1 ran through a separate inference endpoint. These runtimes can affect results.

The task subset excludes 22 RFT training tasks selected because stock Qwen solved them and astra2 failed them. This selection biases the comparison with stock Qwen. The result covers this fixed subset and one selected trial per task.

These terminal results do not compare Hemmingway. The banner's Hemmingway comparison comes from the separate eight-task blog SBS.

The [machine-readable audit](terminal_efficiency.json) gives the per-task token counts and artifact hashes.
