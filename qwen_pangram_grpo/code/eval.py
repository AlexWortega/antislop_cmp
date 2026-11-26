"""Held-out check: generate N samples per held-out task with vLLM (optionally with a LoRA), then score the
answers with the emulator in its calibrated binary mode (the decision) and scalar mode, plus the task check."""
import argparse, json, statistics as st, sys
from concurrent.futures import ThreadPoolExecutor
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--lora', default='')
    ap.add_argument('--data', default='data/heldout_rendered.jsonl'); ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--max-new', type=int, default=3072); ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--out', required=True); ap.add_argument('--thinking', action='store_true')
    a = ap.parse_args()
    import os
    os.environ['EXPECT_THINKING'] = '1' if a.thinking else '0'
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows = [json.loads(l) for l in open(a.data)]
    llm = LLM(a.model, dtype='bfloat16', max_model_len=a.max_new + 2048, gpu_memory_utilization=0.88, trust_remote_code=True,
              enable_lora=bool(a.lora), max_lora_rank=64, enforce_eager=False, max_num_seqs=128)
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=0.95, max_tokens=a.max_new, seed=1)
    outs = llm.generate([r['prompt'] for r in rows], sp, lora_request=LoRARequest('pol', 1, a.lora) if a.lora else None)
    from emu_reward import emu_score, task_score, words, _answer, spent
    items = []
    for r, o in zip(rows, outs):
        for c in o.outputs:
            items.append({'task_id': r['task_id'], 'kind': r['task'], 'question': r['question'], 'target': r['target_words'],
                          'raw': c.text, 'text': _answer(c.text), 'finish': c.finish_reason})
    json.dump({'items': items}, open(a.out + '.gen.json', 'w'), ensure_ascii=False, indent=1)  # keep generations even if scoring fails
    if not os.environ.get('OPENROUTER_API_KEY'):
        sys.exit('OPENROUTER_API_KEY is not set; generations saved to ' + a.out + '.gen.json')
    def score(it):
        t = it['text']
        it['words'] = words(t)
        it['slop'] = emu_score(t, 'binary') if it['words'] >= 15 else True
        it['score'] = emu_score(t, 'score') if it['words'] >= 15 else 100
        it['task'] = task_score(it['question'], t)
        tw = int(it['target']); it['length_ok'] = 0.6 * tw <= it['words'] <= 1.6 * tw
        it['passed'] = it['slop'] is False and (it['task'] or 0) >= 6 and it['length_ok']
        return it
    with ThreadPoolExecutor(12) as p:
        items = list(p.map(score, items))
    n = len(items)
    summary = {'model': a.model, 'lora': a.lora, 'n': n,
               'pass_rate': sum(i['passed'] for i in items) / n,
               'human_rate': sum(i['slop'] is False for i in items) / n,
               'score_mean': st.mean([i['score'] for i in items if i['score'] is not None] or [float('nan')]),
               'unscored': sum(i['score'] is None for i in items),
               'task_mean': st.mean(i['task'] or 0 for i in items),
               'length_ok_rate': sum(i['length_ok'] for i in items) / n,
               'unfinished': sum(i['finish'] == 'length' for i in items),
               'words_mean': st.mean(i['words'] for i in items), 'or_spent': spent()}
    json.dump({'summary': summary, 'items': items}, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
