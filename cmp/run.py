"""humanizer vs Qwen3.8-27B-antislop on 40 writing tasks, judged by Opus.

Outputs per task:
  draft - a plain AI draft (gpt-5.6-luna) of the task
  H     - humanizer rewrite of the draft (exact model-card prompt and sampling)
  QR    - Qwen rewrite of the draft (same instruction, as a chat message, thinking off)
  QD    - Qwen writing the task directly (thinking off)
Track A (rewrite):    H vs QR, facts checked against the draft.
Track B (end result): H vs QD, facts checked against the task.

Steps: drafts | gen-humanizer URL | gen-qwen URL | judge | report
"""
import json
import random
import re
import sys
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from common import JSUF, OUT, ROOT, append_jsonl, chat, judge_abs, judge_pair, load_jsonl
import importlib
import os
TASKS = importlib.import_module(os.environ.get("TASKS", "tasks")).TASKS

DRAFT_MODEL = "openai/gpt-5.6-luna"
DRAFTS = OUT / "drafts.jsonl"
GEN = OUT / "gen.jsonl"
ABS = OUT / f"abs{JSUF}.jsonl"
PAIR = OUT / f"pair{JSUF}.jsonl"
PF = json.loads((Path(os.environ.get("PF_DIR", "/mnt/storage/antislop_cmp/models")) / "prompt_format.json").read_text())


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def step_drafts():
    done = {r["tid"] for r in load_jsonl(DRAFTS)}

    def work(i):
        if i in done:
            return
        text, _ = chat(DRAFT_MODEL, [{"role": "user", "content": TASKS[i][1]}], max_tokens=6000)
        append_jsonl(DRAFTS, {"tid": i, "genre": TASKS[i][0], "task": TASKS[i][1], "text": text.strip()})

    with ThreadPoolExecutor(int(os.environ.get("JUDGE_WORKERS", 8))) as pool:
        list(pool.map(work, range(len(TASKS))))


def humanize(url, draft, seed):
    body = {"prompt": PF["instr"] + "\n\n" + draft.strip() + PF["sep"], "temperature": 1.0, "top_p": 0.95,
            "top_k": 0, "min_p": 0, "repeat_penalty": 1.0, "n_predict": 2048, "seed": seed}
    return post(url + "/completion", body)["content"].strip()


def qwen(url, prompt, seed):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "min_p": 0, "max_tokens": 2048, "seed": seed, "chat_template_kwargs": {"enable_thinking": False}}
    text = post(url + "/v1/chat/completions", body)["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


API_MODELS = {"BQ": "qwen/qwen3.8-27b", "O5": "anthropic/claude-opus-5", "GA": "openai/gpt-6-astra"}
#: extra locally served systems, e.g. EXTRA_SYS=S,A gives SR/SD and AR/AD (rewrite/direct)
EXTRA_SYS = [x for x in os.environ.get("EXTRA_SYS", "").split(",") if x]
REWRITERS = ["H", "QR", "QRt"] + [f"{k}_R" for k in API_MODELS] + [f"{k}R" for k in EXTRA_SYS]
DIRECT = ["QD", "QDt"] + [f"{k}_D" for k in API_MODELS] + [f"{k}D" for k in EXTRA_SYS]
TRACK = {"QR": "A", "QD": "B"}  # tracks of the first run; newer systems get "H~<sys>"


def track_of(name):
    return TRACK.get(name, f"H~{name}")


def api_write(model, prompt):
    extra = {"reasoning": {"enabled": False}, "temperature": 0.7} if model.startswith("qwen/") else {}
    text, _ = chat(model, [{"role": "user", "content": prompt}], max_tokens=8000, **extra)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def step_gen_api():
    drafts = load_jsonl(DRAFTS)
    done = {(r["tid"], r["sys"]) for r in load_jsonl(GEN)}
    jobs = [(d, f"{k}_R", m, PF["instr"] + "\n\n" + d["text"].strip()) for d in drafts for k, m in API_MODELS.items()]
    jobs += [(d, f"{k}_D", m, d["task"]) for d in drafts for k, m in API_MODELS.items()]

    def work(job):
        d, name, model, prompt = job
        if (d["tid"], name) not in done:
            append_jsonl(GEN, {"tid": d["tid"], "sys": name, "text": api_write(model, prompt)})
            print(d["tid"], name, flush=True)

    with ThreadPoolExecutor(int(os.environ.get("GEN_WORKERS", 12))) as pool:
        list(pool.map(work, jobs))


def qwen_think(url, prompt, seed):
    """Qwen-antislop with its default reasoning (effort xhigh); only the final answer is kept."""
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "min_p": 0, "max_tokens": 15000, "seed": seed}
    message = post(url + "/v1/chat/completions", body)["choices"][0]["message"]
    text = re.sub(r"<think>.*?</think>", "", message["content"], flags=re.S).strip()
    return text, len((message.get("reasoning_content") or "").split())


def step_gen_think(url):
    done = {(r["tid"], r["sys"]) for r in load_jsonl(GEN)}
    jobs = [(d, "QRt", PF["instr"] + "\n\n" + d["text"].strip()) for d in load_jsonl(DRAFTS)]
    jobs += [(d, "QDt", d["task"]) for d in load_jsonl(DRAFTS)]

    def work(job):
        d, name, prompt = job
        if (d["tid"], name) not in done:
            text, think_words = qwen_think(url, prompt, d["tid"])
            append_jsonl(GEN, {"tid": d["tid"], "sys": name, "text": text, "think_words": think_words})
            print(d["tid"], name, think_words, flush=True)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work, jobs))


def answer_only(text, think):
    """The part a reader gets. With thinking on, the template opens <think> in the PROMPT, so the
    completion holds only the closing tag: everything up to the last </think> is reasoning, and a
    thought that never closed leaves no answer. Matching <think>...</think> pairs missed all of it and
    sent whole monologues to the judge, which Opus refuses with an empty content_filter reply."""
    if think:
        i = text.rfind("</think>")
        return text[i + len("</think>"):].strip() if i >= 0 else ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def step_gen_vllm(prefix, url, model):
    """Rewrite (<prefix>R) and direct (<prefix>D) outputs from an OpenAI-compatible vLLM server,
    with the same prompts and sampling as the llama.cpp Qwen runs (thinking off).
    BENCH_THINK=1 turns thinking on for models trained with it; only the answer is kept."""
    think = os.environ.get("BENCH_THINK", "0") == "1"

    def ask(prompt, seed):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7,
                "top_p": 0.8, "top_k": 20, "min_p": 0, "max_tokens": 6144 if think else 2048, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": think}}
        text = post(url + "/v1/chat/completions", body)["choices"][0]["message"]["content"] or ""
        return answer_only(text, think)

    done = {(r["tid"], r["sys"]) for r in load_jsonl(GEN)}
    jobs = [(d, f"{prefix}R", PF["instr"] + "\n\n" + d["text"].strip()) for d in load_jsonl(DRAFTS)]
    jobs += [(d, f"{prefix}D", d["task"]) for d in load_jsonl(DRAFTS)]

    def work(job):
        d, name, prompt = job
        if (d["tid"], name) not in done:
            append_jsonl(GEN, {"tid": d["tid"], "sys": name, "text": ask(prompt, d["tid"])})
            print(d["tid"], name, flush=True)

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(work, jobs))


def step_gen(which, url):
    drafts = load_jsonl(DRAFTS)
    done = {(r["tid"], r["sys"]) for r in load_jsonl(GEN)}
    jobs = []
    for d in drafts:
        if which == "humanizer":
            jobs.append((d, "H", lambda d=d: humanize(url, d["text"], d["tid"])))
        else:
            jobs.append((d, "QR", lambda d=d: qwen(url, PF["instr"] + "\n\n" + d["text"].strip(), d["tid"])))
            jobs.append((d, "QD", lambda d=d: qwen(url, d["task"], d["tid"])))

    def work(job):
        d, name, fn = job
        if (d["tid"], name) in done:
            return
        append_jsonl(GEN, {"tid": d["tid"], "sys": name, "text": fn()})
        print(d["tid"], name, flush=True)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work, jobs))


def outputs():
    by = {}
    for d in load_jsonl(DRAFTS):
        by[(d["tid"], "draft")] = d["text"]
    for g in load_jsonl(GEN):
        by[(g["tid"], g["sys"])] = g["text"]
    return by


def step_judge():
    by = outputs()
    drafts = {d["tid"]: d for d in load_jsonl(DRAFTS)}
    done_abs = {(r["tid"], r["sys"]) for r in load_jsonl(ABS)}
    done_pair = {(r["tid"], r["track"], r["order"]) for r in load_jsonl(PAIR)}
    abs_jobs, pair_jobs = [], []
    for tid, d in drafts.items():
        for name in ["draft"] + REWRITERS + DIRECT:
            if (tid, name) in by and (tid, name) not in done_abs:
                source = d["text"] if name in REWRITERS else None
                abs_jobs.append((tid, name, d["task"], by[(tid, name)], source))
        for other in REWRITERS[1:] + DIRECT:
            track, source = track_of(other), (d["text"] if other in REWRITERS else None)
            if (tid, "H") in by and (tid, other) in by:
                for order in ("H_first", "Q_first"):
                    if (tid, track, order) not in done_pair:
                        pair_jobs.append((tid, track, order, d["task"], by[(tid, "H")], by[(tid, other)], source))

    def run_abs(job):
        tid, name, task, text, source = job
        try:
            result, usage = judge_abs(task, text, source)
        except Exception as e:  # one refused or broken call must not take the whole run down
            print(f"abs failed {tid} {name}: {str(e)[:120]}", flush=True)
            return
        append_jsonl(ABS, {"tid": tid, "sys": name, **result, "words": len(text.split()), "usage": usage})

    def run_pair(job):
        tid, track, order, task, h, q, source = job
        a, b = (h, q) if order == "H_first" else (q, h)
        try:
            result, usage = judge_pair(task, a, b, source)
        except Exception as e:
            print(f"pair failed {tid} {track} {order}: {str(e)[:120]}", flush=True)
            return
        h_slot = "1" if order == "H_first" else "2"
        picked = "tie" if result["winner"] == "tie" else ("H" if result["winner"] == h_slot else "Q")
        append_jsonl(PAIR, {"tid": tid, "track": track, "order": order, "picked": picked, **result, "usage": usage})

    with ThreadPoolExecutor(int(os.environ.get("JUDGE_WORKERS", 8))) as pool:
        list(pool.map(run_abs, abs_jobs))
        list(pool.map(run_pair, pair_jobs))


def mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def boot_ci(xs, n=5000):
    rng = random.Random(0)
    stats = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return [round(stats[int(0.025 * n)], 3), round(stats[int(0.975 * n)], 3)]


def step_report():
    report = {"absolute": {}, "pairwise": {}}
    rows = load_jsonl(ABS)
    for name in ["draft"] + REWRITERS + DIRECT:
        g = [r for r in rows if r["sys"] == name]
        if not g:
            continue
        report["absolute"][name] = {
            "n": len(g), "slop": mean([r["slop"] for r in g]), "human": mean([r["human"] for r in g]),
            "task": mean([r["task"] for r in g]), "tells": mean([len(r["tells"]) for r in g]),
            "lost_facts": mean([len(r["lost_facts"]) for r in g]),
            "fabrications": mean([len(r["fabrications"]) for r in g]),
            "texts_with_fact_errors": sum(bool(r["lost_facts"] or r["fabrications"]) for r in g),
            "em_dashes": mean([r["em_dashes"] for r in g]), "words": mean([r["words"] for r in g]),
        }
        counts = {}
        for r in g:
            for t in r["tells"]:
                counts[t["pattern"]] = counts.get(t["pattern"], 0) + 1
        report["absolute"][name]["top_tells"] = sorted(counts.items(), key=lambda x: -x[1])[:6]
    pairs = load_jsonl(PAIR)
    for other in REWRITERS[1:] + DIRECT:
        track = track_of(other)
        p = [r for r in pairs if r["track"] == track]
        if not p:
            continue
        per_task = {}
        for r in p:
            per_task.setdefault(r["tid"], []).append({"H": 1.0, "Q": 0.0, "tie": 0.5}[r["picked"]])
        scores = [sum(v) / len(v) for v in per_task.values()]
        report["pairwise"][other] = {
            "n_tasks": len(per_task), "H_winrate": mean(scores), "H_winrate_ci95": boot_ci(scores) if scores else None,
            "slot1_rate": mean([r["winner"] == "1" for r in p]),
            "order_consistent": sum(len(set(v)) == 1 for v in per_task.values()),
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    (OUT / f"report{JSUF}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "drafts":
        step_drafts()
    elif step == "gen-vllm":
        step_gen_vllm(sys.argv[2], sys.argv[3], sys.argv[4])
    elif step == "gen-think":
        step_gen_think(sys.argv[2])
    elif step == "gen-api":
        step_gen_api()
    elif step.startswith("gen-"):
        step_gen(step[4:], sys.argv[2])
    else:
        {"judge": step_judge, "report": step_report}[step]()
