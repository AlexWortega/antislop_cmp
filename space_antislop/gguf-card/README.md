---
license: apache-2.0
base_model: rawmodels/Qwen3.8-27B-antislop
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
language:
  - en
  - ru
tags:
  - gguf
  - q4_k_m
  - antislop
  - writing
---

# Antislop rft1 · Q4_K_M

The current Antislop rft1 weights, quantized for local inference and the [ZeroGPU editor](https://huggingface.co/spaces/AlexWortega/antislop-27b).

| File | Size | Quantization |
| --- | ---: | --- |
| `Qwen3.8-27B-antislop-rft1-Q4_K_M.gguf` | 16.55 GB / 15.41 GiB | Q4_K_M, 4.92 bits per weight |

The source safetensors match the public Antislop revision `7d44812e5c4465abb2ff9f426b9d8117ee3d4dcb` and rft1 revision `335e6f8f2c4e0d499a1c7607f71b3fe89dbae4d8`. File hashes and conversion details are in [weights_manifest.json](weights_manifest.json).

Text-only target. The MTP head and vision projector are excluded. The importance matrix uses only the training portion of the new rewrite calibration set: two 256-token chunks, 496 tensor entries. Its corpus is small; it is not a broad quantization benchmark.

## Rewrite serving

The editor uses two short editing examples, a source-grounded prompt and a draft word-count ceiling. Thinking is explicitly closed with the Qwen assistant prefix. Temperature alone did not prevent unsupported elaboration.

Selected on validation: temperature **0.4**, top_p **0.9**, top_k **40**, min_p **0.03**, repeat penalty **1.05**. The runtime is `llama-cpp-python` 0.3.34 with CUDA 12.5, all layers on GPU, Flash Attention and an 8,192-token context. Sampling uses seed 271828.

Use the exact [serving.py](https://huggingface.co/spaces/AlexWortega/antislop-27b/blob/main/serving.py) prompt for this editor behavior. These settings were selected for rewriting short Russian and English drafts. They are not universal recommendations for chat or reasoning.

[Calibration report](https://huggingface.co/spaces/AlexWortega/antislop-27b/blob/main/CALIBRATION.md)

Keep the source facts and qualifications, and check the result before publishing. This model can still omit or change a detail.
