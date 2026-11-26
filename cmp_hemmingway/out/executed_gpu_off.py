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


if os.environ.get('HMW_FP8_SMALL_PROJECTION_BF16') == '1':
    install_fp8_projection_exception()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['download', 'generate'])
    p.add_argument('--work', required=True)
    p.add_argument('--model', choices=['rft1', 'hemmingway'])
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
    dest = work / (label + '_generations.json')
    if dest.exists():
        raise ValueError('Existing generations are immutable')
    rows = [json.loads(x) for x in (work / 'examples.jsonl').read_text().splitlines()]
    import torch
    from vllm import LLM, SamplingParams
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError('GPU lacks free memory; do not disturb other processes')
    config = json.loads((Path(path) / 'config.json').read_text())
    options = {'limit_mm_per_prompt': {'image': 0, 'video': 0}} if config['model_type'] == 'qwen3_5' else {}
    llm = LLM(model=path, dtype='bfloat16', quantization='fp8',
        gpu_memory_utilization=.70, max_model_len=12288, max_num_seqs=4,
        max_num_batched_tokens=4096, enforce_eager=True, seed=271828,
        enable_prefix_caching=True, **options)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{'role': 'user', 'content': r['assignment']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    for r, prompt in zip(rows, prompts):
        n = len(tok.encode(prompt, add_special_tokens=False))
        if n + 2048 > 12288:
            raise ValueError('Context overflow: ' + r['id'])
    sampling = SamplingParams(temperature=1., top_p=.95, top_k=20,
        max_tokens=2048, seed=271828)
    started = time.time()
    outputs = llm.generate(prompts, sampling)
    generations = []
    for row, out in zip(rows, outputs):
        result = out.outputs[0]
        generations.append({'id': row['id'], 'domain': row['domain'],
            'language': row['language'], 'text': result.text,
            'finish_reason': result.finish_reason,
            'prompt_tokens': len(out.prompt_token_ids),
            'completion_tokens': len(result.token_ids)})
    record = {'model': model['repo'], 'revision': model['revision'],
        'quantization': 'vLLM online FP8, Marlin on sm86; in_proj_ba remains bf16', 'dtype': 'bfloat16',
        'temperature': 1., 'top_p': .95, 'top_k': 20, 'seed': 271828,
        'max_tokens': 2048, 'enable_thinking': False, 'seconds': time.time() - started,
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
