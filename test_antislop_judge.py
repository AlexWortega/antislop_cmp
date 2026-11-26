"""Live check of score_antislop on held-out semdiv1 groups: parses, 1 score per text, spread > 0."""
import asyncio, collections, json, os, statistics as st, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_reward as J

rows = [json.loads(l) for l in open("runs/heldout/semdiv1_samples.jsonl")]
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["step"], r["question"])].append(r["text"])
picked = [(k, v) for k, v in groups.items() if len(v) >= 4][:3]

async def main():
    cl = J.client()
    for (step, q), texts in picked:
        t = time.time()
        sc = await cl.score_antislop(q, texts)
        totals = [s["total"] for s in sc]
        print(f"step {step} n={len(texts)} {time.time()-t:.0f}s totals={totals} sd={st.pstdev(totals):.1f} "
              f"neutral={sum(bool(s.get('neutral')) for s in sc)} tells0={sc[0]['tells'][:4]}")
    print("calls", cl.calls, "failures", cl.failures, "truncated", cl.truncated, "model", cl.last_model)
asyncio.run(main())
