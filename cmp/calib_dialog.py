"""Calibrate the Opus antislop judge on short multi-turn chats: pre-LLM human replies vs matched AI replies.

The user side of each chat is kept as written by a human; only the other side ("assistant") is swapped.
Sources: Wizard of Wikipedia (2018, a human answering from Wikipedia) and Blended Skill Talk (2020,
crowdworker chit-chat). AI versions are written turn by turn against the real user turns, plain and
"write like a real person". The judge sees the whole chat and scores the ASSISTANT's turns only, with
the same system text and rulebook as the single-text benchmark.

Steps (resumable): human | ai | judge | report.   JUDGE / JSUF select the judge model like calib.py.
"""
import json
import os
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import JSUF, OUT, append_jsonl, chat, load_jsonl, parse_json, _system, JUDGE

HUMAN = OUT / "dlg_human.jsonl"
AI = OUT / "dlg_ai.jsonl"
ABS = OUT / f"dlg_abs{JSUF}.jsonl"
PAIR = OUT / f"dlg_pair{JSUF}.jsonl"
WRITERS = ["openai/gpt-5.6-luna", "google/gemini-3.8-flash", "deepseek/deepseek-v4.1-flash"]
PER_SOURCE = 25
STYLES = {
    "plain": "You are a helpful assistant chatting with a user. Reply to their last message.",
    "humanlike": ("You are chatting with someone. Reply to their last message the way a real person would in a "
                  "casual chat: natural, specific, no assistant mannerisms, no lists, no headings."),
}

DLG_PROMPT = """A chat between a USER and an ASSISTANT:
<chat>
{chat}
</chat>

Judge only the ASSISTANT's turns (the user's turns are context).
Do this:
1. List every AI-writing tell from the rulebook that appears in the assistant's turns, as {{"pattern": "<rulebook pattern name>", "quote": "<short exact quote>"}}.
2. Scores for the assistant's side of the chat as a whole:
   - "slop": 0-10, how much AI slop by the rulebook (0 = none, 10 = saturated).
   - "human": 1-10, how much it reads like a specific person wrote it (voice, rhythm, concrete detail).
   - "task": 1-10, how well it serves the user across the chat (answers what was asked, follows up sensibly).

Answer with JSON only:
{{"tells": [...], "slop": int, "human": int, "task": int}}"""

PAIR_PROMPT = """Two versions of the same chat. The USER's turns are identical; only the ASSISTANT's turns differ.

Chat 1:
<chat1>
{a}
</chat1>

Chat 2:
<chat2>
{b}
</chat2>

In which chat do the ASSISTANT's turns better satisfy the rulebook: fewer AI-writing tells, a real human voice, and still serving the user? Ignore length unless it causes a rulebook violation.
Answer with JSON only: {{"winner": "1" | "2" | "tie", "reason": "<one sentence>"}}"""


def rows(dataset, split, n, offset=0):
    url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config=default&split={split}"
           f"&offset={offset}&length={n}")
    return [r["row"] for r in json.load(urllib.request.urlopen(url, timeout=60))["rows"]]


def as_list(x):
    return x if isinstance(x, list) else eval(x)  # the WoW mirror stores python-literal strings


def step_human():
    out = []
    rng = random.Random(0)
    wow = rows("chujiezheng/wizard_of_wikipedia", "test", 100)
    rng.shuffle(wow)
    for i, r in enumerate(wow):
        post, resp = as_list(r["post"]), as_list(r["response"])
        turns = []
        for u, a in list(zip(post, resp))[:3]:
            u = u.split("\n")[-1].strip()
            turns += [{"role": "user", "content": u}, {"role": "assistant", "content": a.strip()}]
        if len(turns) == 6 and all(len(t["content"].split()) >= 4 for t in turns):
            out.append({"id": f"wow{i}", "source": "wow", "turns": turns})
        if sum(o["source"] == "wow" for o in out) >= PER_SOURCE:
            break
    bst = rows("ParlAI/blended_skill_talk", "test", 100)
    rng.shuffle(bst)
    for i, r in enumerate(bst):
        a_side = r["free_messages"]
        b_side = r["guided_messages"]
        turns = []
        for u, a in list(zip(a_side, b_side))[:3]:
            turns += [{"role": "user", "content": u.strip()}, {"role": "assistant", "content": a.strip()}]
        if len(turns) == 6 and all(len(t["content"].split()) >= 4 for t in turns):
            out.append({"id": f"bst{i}", "source": "bst", "turns": turns})
        if sum(o["source"] == "bst" for o in out) >= PER_SOURCE:
            break
    HUMAN.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
    print("human chats:", {s: sum(o["source"] == s for o in out) for s in ("wow", "bst")})


def step_ai():
    done = {(r["id"], r["style"]) for r in load_jsonl(AI)}
    jobs = [(h, s) for h in load_jsonl(HUMAN) for s in STYLES if (h["id"], s) not in done]

    def work(job):
        h, style = job
        writer = WRITERS[hash(h["id"] + style) % len(WRITERS)]
        msgs = [{"role": "system", "content": STYLES[style]}]
        turns = []
        for t in h["turns"]:
            if t["role"] == "user":
                msgs.append({"role": "user", "content": t["content"]}); turns.append(t)
            else:
                text, _ = chat(writer, msgs, max_tokens=4000)
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                msgs.append({"role": "assistant", "content": text}); turns.append({"role": "assistant", "content": text})
        append_jsonl(AI, {"id": h["id"], "source": h["source"], "style": style, "writer": writer, "turns": turns})

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(work, jobs))
    print("ai chats:", len(load_jsonl(AI)))


def render(turns):
    return "\n\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)


def judge(prompt):
    for _ in range(3):
        content, usage = chat(JUDGE, [{"role": "system", "content": _system()}, {"role": "user", "content": prompt}],
                              max_tokens=4000)
        try:
            return parse_json(content), usage
        except (AttributeError, ValueError):
            continue
    raise RuntimeError("unparsable")


def step_judge():
    items = [{"key": f"{h['id']}|human", **h, "kind": "human"} for h in load_jsonl(HUMAN)]
    items += [{"key": f"{a['id']}|{a['style']}", **a, "kind": a["style"]} for a in load_jsonl(AI)]
    done = {r["key"] for r in load_jsonl(ABS)}

    def run_abs(t):
        if t["key"] in done:
            return
        try:
            res, usage = judge(DLG_PROMPT.format(chat=render(t["turns"])))
        except Exception as e:
            print("abs failed", t["key"], str(e)[:120], flush=True); return
        append_jsonl(ABS, {"key": t["key"], "id": t["id"], "source": t["source"], "kind": t["kind"], **res, "usage": usage})

    by = {t["key"]: t for t in items}
    pdone = {(r["id"], r["style"], r["order"]) for r in load_jsonl(PAIR)}
    pairs = [(h, by[f"{h['id']}|{s}"], s, o) for h in items if h["kind"] == "human" for s in STYLES
             if f"{h['id']}|{s}" in by for o in ("human_first", "ai_first") if (h["id"], s, o) not in pdone]

    def run_pair(job):
        h, ai, style, order = job
        a, b = (h, ai) if order == "human_first" else (ai, h)
        try:
            res, usage = judge(PAIR_PROMPT.format(a=render(a["turns"]), b=render(b["turns"])))
        except Exception as e:
            print("pair failed", h["id"], str(e)[:120], flush=True); return
        hs = "1" if order == "human_first" else "2"
        picked = "tie" if res.get("winner") == "tie" else ("human" if res.get("winner") == hs else "ai")
        append_jsonl(PAIR, {"id": h["id"], "style": style, "order": order, "picked": picked, **res, "usage": usage})

    with ThreadPoolExecutor(int(os.environ.get("JUDGE_WORKERS", 8))) as pool:
        list(pool.map(run_abs, items))
        list(pool.map(run_pair, pairs))


def auroc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def step_report():
    rs = load_jsonl(ABS); rep = {}
    for style in STYLES:
        hum = [r for r in rs if r["kind"] == "human"]; ai = [r for r in rs if r["kind"] == style]
        if not hum or not ai:
            continue
        rep[f"human_vs_{style}"] = {
            "n": [len(hum), len(ai)],
            "auroc_human": round(auroc([r["human"] for r in hum], [r["human"] for r in ai]), 3),
            "auroc_neg_slop": round(auroc([-r["slop"] for r in hum], [-r["slop"] for r in ai]), 3),
            "mean_human": [round(sum(r["human"] for r in g) / len(g), 2) for g in (hum, ai)],
            "mean_slop": [round(sum(r["slop"] for r in g) / len(g), 2) for g in (hum, ai)],
            "by_source_auroc_human": {s: round(auroc([r["human"] for r in hum if r["source"] == s],
                                                     [r["human"] for r in ai if r["source"] == s]), 3)
                                      for s in ("wow", "bst")},
        }
    ps = load_jsonl(PAIR)
    for style in STYLES:
        p = [r for r in ps if r["style"] == style]
        if not p:
            continue
        both = {}
        for r in p:
            both.setdefault(r["id"], {})[r["order"]] = r["picked"]
        rep[f"pair_{style}"] = {"n": len(p), "acc_picks_human": round(sum(r["picked"] == "human" for r in p) / len(p), 3),
                                "slot1_rate": round(sum(r.get("winner") == "1" for r in p) / len(p), 3),
                                "order_consistent": f"{sum(len(v) == 2 and v['human_first'] == v['ai_first'] for v in both.values())}/{len(both)}"}
    refused = sum(1 for r in rs if not r.get("usage"))
    print(json.dumps(rep, indent=2)); (OUT / f"dlg_report{JSUF}.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    {"human": step_human, "ai": step_ai, "judge": step_judge, "report": step_report}[sys.argv[1]]()
