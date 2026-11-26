"""Pinned checkpoint download and matched vLLM generation for the comparison."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path


def install_fp8_projection_exception():
    # Marlin requires output widths divisible by 64. Qwen's 96-wide beta/alpha
    # projection stays in bf16 for both checkpoints; do not alter installed vLLM.
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    original = Fp8Config.get_quant_method
    def method(self, layer, prefix):
        if prefix.endswith('.in_proj_ba'):
            return UnquantizedLinearMethod()
        return original(self, layer, prefix)
    Fp8Config.get_quant_method = method
    # vLLM 0.19 omitted hybrid-cache metadata on the text-only Qwen3.5 class.
    # Reuse the same state-shape/dtype methods as its multimodal wrapper.
    from vllm import ModelRegistry
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration)
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Qwen3_5ForConditionalGenerationConfig
    Qwen3_5ForCausalLM.is_hybrid = True
    for name in ('get_mamba_state_dtype_from_config', 'get_mamba_state_shape_from_config',
                 'get_mamba_state_copy_func'):
        setattr(Qwen3_5ForCausalLM, name, classmethod(getattr(Qwen3_5ForConditionalGeneration, name).__func__))
    MODELS_CONFIG_MAP['Qwen3_5ForCausalLM'] = Qwen3_5ForConditionalGenerationConfig
    # For text-only requests M-RoPE has three identical position rows and
    # a zero delta, exactly as in the existing multimodal wrapper.
    import torch
    def text_positions(input_tokens, hf_config=None, *args, **kwargs):
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).repeat(3, 1), 0
    Qwen3_5ForCausalLM.supports_mrope = True
    Qwen3_5ForCausalLM.get_mrope_input_positions = staticmethod(text_positions)
    ModelRegistry.register_model('Qwen3_5ForCausalLM', Qwen3_5ForCausalLM)


if os.environ.get('HMW_FP8_SMALL_PROJECTION_BF16') == '1':
    install_fp8_projection_exception()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['download', 'generate'])
    p.add_argument('--work', required=True)
    p.add_argument('--model', choices=['rft1', 'hemmingway'])
    p.add_argument('--thinking', action='store_true')
    a = p.parse_args()
    work = Path(a.work)
    manifest = json.loads((work / 'model_manifest.json').read_text())
    from huggingface_hub import snapshot_download
    labels = [a.model] if a.model else list(manifest)
    for label in labels:
        model = manifest[label]
        started = time.time()
        print('download', label, model['revision'], flush=True)
        path = snapshot_download(model['repo'], revision=model['revision'],
            cache_dir=work / 'hf', max_workers=4,
            allow_patterns=['*.safetensors', '*.json', '*.jinja', 'merges.txt', 'vocab.json'],
            ignore_patterns=['evaluations/*', 'release_manifest.json'])
        (work / (label + '_path.txt')).write_text(path)
        print('downloaded', label, round(time.time() - started, 1), flush=True)
    if a.action == 'download':
        return
    if len(labels) != 1:
        raise ValueError('Generate one model per process so GPU memory is released')
    label = labels[0]
    dest = work / (label + ('_thinking' if a.thinking else '') + '_generations.json')
    if dest.exists():
        raise ValueError('Existing generations are immutable')
    rows = [json.loads(x) for x in (work / 'examples.jsonl').read_text().splitlines()]
    import torch
    from vllm import LLM, SamplingParams
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError('GPU lacks free memory; do not disturb other processes')
    config = json.loads((Path(path) / 'config.json').read_text())
    options = {'limit_mm_per_prompt': {'image': 0, 'video': 0}} if config['model_type'] == 'qwen3_5' else {}
    context_limit = 16384 if a.thinking else 12288
    output_limit = 8192 if a.thinking else 2048
    llm = LLM(model=path, dtype='bfloat16', quantization='fp8',
        gpu_memory_utilization=.70, max_model_len=context_limit, max_num_seqs=4,
        max_num_batched_tokens=4096, enforce_eager=True, seed=271828,
        enable_prefix_caching=True, **options)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{'role': 'user', 'content': r['assignment']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=a.thinking) for r in rows]
    for r, prompt in zip(rows, prompts):
        n = len(tok.encode(prompt, add_special_tokens=False))
        if n + output_limit > context_limit:
            raise ValueError('Context overflow: ' + r['id'])
    sampling = SamplingParams(temperature=1., top_p=.95, top_k=20,
        max_tokens=output_limit, seed=271828)
    started = time.time()
    outputs = llm.generate(prompts, sampling)
    generations = []
    for row, out in zip(rows, outputs):
        result = out.outputs[0]
        ids = list(result.token_ids)
        close_id = tok.convert_tokens_to_ids('</think>')
        reasoning_tokens = ids.index(close_id) if close_id in ids else (len(ids) if a.thinking else 0)
        text = tok.decode(ids[reasoning_tokens + 1:], skip_special_tokens=True).lstrip('\r\n') if close_id in ids else ('' if a.thinking else result.text)
        finish = 'unclosed_thought' if a.thinking and close_id not in ids else result.finish_reason
        generations.append({'id': row['id'], 'domain': row['domain'],
            'language': row['language'], 'text': text,
            'raw_text': result.text if a.thinking else None,
            'finish_reason': finish, 'reasoning_tokens': reasoning_tokens,
            'prompt_tokens': len(out.prompt_token_ids),
            'completion_tokens': len(result.token_ids)})
    record = {'model': model['repo'], 'revision': model['revision'],
        'quantization': 'vLLM online FP8, Marlin on sm86; in_proj_ba remains bf16', 'dtype': 'bfloat16',
        'temperature': 1., 'top_p': .95, 'top_k': 20, 'seed': 271828,
        'max_tokens': output_limit, 'enable_thinking': a.thinking, 'seconds': time.time() - started,
        'vllm': importlib.metadata.version('vllm'),
        'transformers': importlib.metadata.version('transformers'),
        'gpu': torch.cuda.get_device_name(),
        'input_sha256': hashlib.sha256((work / 'examples.jsonl').read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'generations': generations}
    dest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print('generated', label, len(generations), flush=True)


if __name__ == '__main__':
    main()
