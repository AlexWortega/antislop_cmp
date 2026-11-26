#!/usr/bin/env bash
# Runs on the playground GPU: base model as "base", the step-50 LoRA as "step50", on 127.0.0.1:8000 only.
set -uo pipefail
cd /workspace
export PATH=/opt/venv/bin:$PATH
until [ -f /workspace/base/READY ] && [ -f /workspace/adapter_step50/READY ]; do sleep 5; done
args=(--model /workspace/base --served-model-name base --dtype bfloat16 --language-model-only
      --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.90
      --enable-lora --max-lora-rank 256 --max-loras 2 --lora-modules step50=/workspace/adapter_step50 step50x4=/workspace/adapter_step50x4
      --enable-auto-tool-choice --tool-call-parser qwen3_xml --generation-config vllm --default-chat-template-kwargs '{"enable_thinking":false}'
      --api-key "$(cat /workspace/serve.key)" --host 127.0.0.1 --port 8000)
/opt/venv/bin/python -m vllm.entrypoints.openai.api_server "${args[@]}" >> /workspace/vllm.log 2>&1
# CUDA graph capture is the usual failure with LoRA on new architectures; eager mode is slower but safe.
echo "=== retry with --enforce-eager ===" >> /workspace/vllm.log
exec /opt/venv/bin/python -m vllm.entrypoints.openai.api_server "${args[@]}" --enforce-eager >> /workspace/vllm.log 2>&1
