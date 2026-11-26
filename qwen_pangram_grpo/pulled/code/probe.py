"""Before training: sample 8 completions for 6 train prompts at T=1 and measure the reward spread GRPO would see,
plus reasoning length. Decides thinking on/off and checks that the emulator gives within-group variance."""
import argparse, json, statistics as st, os
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--thinking', action='store_true'); ap.add_argument('--max-new', type=int, default=3072); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.environ['EXPECT_THINKING'] = '1' if a.thinking else '0'
    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in open(a.data)][:6]
    llm = LLM(a.model, dtype='bfloat16', max_model_len=a.max_new + 2048, gpu_memory_utilization=0.88, trust_remote_code=True, max_num_seqs=128)
    outs = llm.generate([r['prompt'] for r in rows], SamplingParams(n=8, temperature=1.0, top_p=1.0, max_tokens=a.max_new, seed=0))
    from emu_reward import EmuReward, TaskReward, length_reward
    from thinking import split_thinking
    emu, task = EmuReward(), TaskReward()
    report = []
    for r, o in zip(rows, outs):
        comps = [c.text for c in o.outputs]
        kw = {'question': [r['question']] * 8, 'target_words': [r['target_words']] * 8}
        e, t, l = emu([r['prompt']] * 8, comps, **kw), task([r['prompt']] * 8, comps, **kw), length_reward(None, comps, **kw)
        toks = [len(c.token_ids) for c in o.outputs]
        closed = [split_thinking(c)[2] for c in comps] if a.thinking else [True] * 8
        report.append({'kind': r['task'], 'emu_scores': [d['score'] for d in emu.last_details], 'emu_sd': round(st.pstdev(e), 3),
                       'task': task.last, 'len': l, 'tokens_median': st.median(toks), 'closed': sum(closed),
                       'sample': comps[0][-600:]})
        print(json.dumps({k: v for k, v in report[-1].items() if k != 'sample'}), flush=True)
    json.dump(report, open(a.out, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
