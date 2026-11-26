"""Calibrate the Opus judge on pre-LLM human texts vs matched AI texts.

Steps (each resumable):
  human  - sample human texts (Enron 2001 emails, Reddit 2017 posts, Yelp reviews)
  ai     - per human text: a writing brief, then an AI text written from it, plain and "humanlike"
  judge  - blind absolute scores for every text, repeats on a subset, pairwise human vs AI both orders
  report - AUROC, pairwise accuracy, position bias, repeat stability
"""
import json
import os
import random
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import JSUF, EXP, OUT, append_jsonl, chat, judge_abs, judge_pair, load_jsonl

HUMAN = OUT / "calib_human.jsonl"
AI = OUT / "calib_ai.jsonl"
ABS = OUT / f"calib_abs{JSUF}.jsonl"
PAIR = OUT / f"calib_pair{JSUF}.jsonl"
PER_SOURCE = 15
WRITERS = ["openai/gpt-5.6-luna", "google/gemini-3.8-flash", "deepseek/deepseek-v4.1-flash"]
BRIEF_MODEL = "openai/gpt-5.6-luna"
SOURCES = {
    "enron": ("corbt/enron-emails", "default", 500_000),
    "reddit": ("webis/tldr-17", "default", 3_800_000),
    "yelp": ("Yelp/yelp_review_full", "yelp_review_full", 650_000),
}


def rows(dataset, config, offset):
    url = ("https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": "train", "offset": offset, "length": 100}))
    with urllib.request.urlopen(url, timeout=120) as r:
        return [x["row"] for x in json.load(r)["rows"]]


def clean(source, row):
    if source == "enron":
        text = row["body"]
        if "Original Message" in text or "Forwarded" in text or "-----" in text or "@" in text:
            return None
    elif source == "reddit":
        text = row["content"]
        if "http" in text or "&gt;" in text:
            return None
    else:
        text = row["text"].replace("\\n", "\n").replace('\\"', '"')
    text = unwrap(text)
    words = len(text.split())
    return text if 90 <= words <= 300 else None


def unwrap(text):
    """Join hard-wrapped lines inside paragraphs so layout does not reveal the source."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return "\n\n".join(re.sub(r"\s+", " ", p).strip() for p in paragraphs if p.strip())


def human_homework():
    import csv
    csv.field_size_limit(10 ** 9)
    rng = random.Random(0)
    essays = [r["full_text"] for r in csv.DictReader(open(OUT.parent.parent.parent / "persuade_full_text.csv"))]
    rng.shuffle(essays)
    picked = [unwrap(t) for t in essays if 150 <= len(t.split()) <= 450][:45]
    for i, text in enumerate(picked):
        append_jsonl(HUMAN, {"id": f"persuade-{i:02d}", "source": "persuade", "text": text})


def step_human():
    if EXP == "homework":
        return human_homework()
    rng = random.Random(0)
    done = load_jsonl(HUMAN)
    for source, (dataset, config, size) in SOURCES.items():
        have = sum(r["source"] == source for r in done)
        while have < PER_SOURCE:
            for row in rows(dataset, config, rng.randrange(size)):
                text = clean(source, row)
                if text and have < PER_SOURCE and rng.random() < 0.3:
                    item = {"id": f"{source}-{have:02d}", "source": source, "text": text}
                    append_jsonl(HUMAN, item)
                    done.append(item)
                    have += 1
            print(source, have, flush=True)


BRIEF = """Below is a piece of writing. Write a writing brief (for a school essay: the assignment prompt as a teacher would give it, and the student's likely grade) that someone could use to produce a similar piece without seeing it: the genre and setting (email at work, Reddit post in some subreddit, restaurant review...), who writes it and to whom, what it must say, every concrete fact it relies on (names, numbers, dates, places), and roughly how many words. Do not quote its sentences and do not describe its style. Output only the brief.

<writing>
{text}
</writing>"""
STYLES = {
    "plain": "{brief}",
    "humanlike": "{brief}\n\nWrite it so it reads like a real person typed it, not an AI: natural voice, uneven rhythm, no polish for its own sake. Output only the text.",
}
if EXP == "homework":
    STYLES["humanlike"] = ("{brief}\n\nWrite it the way a real student in that grade would write it for homework, "
                           "not an AI: their own voice, their level, uneven rhythm. Output only the essay.")


def step_ai():
    human = load_jsonl(HUMAN)
    done = {(r["id"], r["style"]) for r in load_jsonl(AI)}

    def work(i_item):
        i, item = i_item
        brief = None
        for style, template in STYLES.items():
            if (item["id"], style) in done:
                continue
            if brief is None:
                brief, _ = chat(BRIEF_MODEL, [{"role": "user", "content": BRIEF.format(text=item["text"])}])
            writer = WRITERS[i % len(WRITERS)]
            text, _ = chat(writer, [{"role": "user", "content": template.format(brief=brief)}], max_tokens=6000)
            append_jsonl(AI, {"id": item["id"], "style": style, "writer": writer, "brief": brief, "text": text.strip()})

    with ThreadPoolExecutor(int(os.environ.get("JUDGE_WORKERS", 8))) as pool:
        list(pool.map(work, enumerate(human)))


def texts():
    briefs = {r["id"]: r["brief"] for r in load_jsonl(AI)}
    out = [{"key": f"{h['id']}|human", "id": h["id"], "kind": "human", "task": briefs[h["id"]], "text": h["text"]}
           for h in load_jsonl(HUMAN) if h["id"] in briefs]
    out += [{"key": f"{a['id']}|{a['style']}", "id": a["id"], "kind": a["style"], "writer": a["writer"],
             "task": a["brief"], "text": re.sub(r"\A\s*Subject:[^\n]*\n+", "", a["text"])} for a in load_jsonl(AI)]
    return out


def step_judge():
    items = texts()
    rng = random.Random(1)
    repeat_keys = {t["key"] for t in rng.sample(items, 15)}
    jobs = [(t, 0) for t in items] + [(t, r) for t in items if t["key"] in repeat_keys for r in (1, 2)]
    done = {(r["key"], r["rep"]) for r in load_jsonl(ABS)}

    def run_abs(job):
        t, rep = job
        if (t["key"], rep) in done:
            return
        result, usage = judge_abs(t["task"], t["text"])
        append_jsonl(ABS, {"key": t["key"], "id": t["id"], "kind": t["kind"], "rep": rep, **result, "usage": usage})

    by_key = {t["key"]: t for t in items}
    pairs = []
    for h in [t for t in items if t["kind"] == "human"]:
        for style in STYLES:
            ai = by_key.get(f"{h['id']}|{style}")
            if ai:
                pairs += [(h, ai, style, "human_first"), (h, ai, style, "ai_first")]
    pdone = {(r["id"], r["style"], r["order"]) for r in load_jsonl(PAIR)}

    def run_pair(job):
        h, ai, style, order = job
        if (h["id"], style, order) in pdone:
            return
        a, b = (h["text"], ai["text"]) if order == "human_first" else (ai["text"], h["text"])
        result, usage = judge_pair(h["task"], a, b)
        human_slot = "1" if order == "human_first" else "2"
        picked = "tie" if result["winner"] == "tie" else ("human" if result["winner"] == human_slot else "ai")
        append_jsonl(PAIR, {"id": h["id"], "style": style, "order": order, "picked": picked, **result, "usage": usage})

    with ThreadPoolExecutor(int(os.environ.get("JUDGE_WORKERS", 8))) as pool:
        list(pool.map(run_abs, jobs))
        list(pool.map(run_pair, pairs))


def auroc(pos, neg):
    """P(score of a human text > score of an AI text), ties count half."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def step_report():
    rows_ = [r for r in load_jsonl(ABS) if r["rep"] == 0]
    report = {}
    for style in STYLES:
        hum = [r for r in rows_ if r["kind"] == "human"]
        ai = [r for r in rows_ if r["kind"] == style]
        report[f"human_vs_{style}"] = {
            "n_human": len(hum), "n_ai": len(ai),
            "auroc_human_score": round(auroc([r["human"] for r in hum], [r["human"] for r in ai]), 3),
            "auroc_neg_slop": round(auroc([-r["slop"] for r in hum], [-r["slop"] for r in ai]), 3),
            "mean_human": [round(sum(r["human"] for r in g) / len(g), 2) for g in (hum, ai)],
            "mean_slop": [round(sum(r["slop"] for r in g) / len(g), 2) for g in (hum, ai)],
            "mean_tells": [round(sum(len(r["tells"]) for r in g) / len(g), 2) for g in (hum, ai)],
        }
        by_source = {}
        for src in ([] if EXP == "homework" else SOURCES):
            h = [r["human"] for r in hum if r["id"].startswith(src)]
            a = [r["human"] for r in ai if r["id"].startswith(src)]
            by_source[src] = round(auroc(h, a), 3) if h and a else None
        report[f"human_vs_{style}"]["auroc_human_by_source"] = by_source
    pairs = load_jsonl(PAIR)
    for style in STYLES:
        p = [r for r in pairs if r["style"] == style]
        both = {}
        for r in p:
            both.setdefault(r["id"], {})[r["order"]] = r["picked"]
        consistent = [v for v in both.values() if len(v) == 2 and v["human_first"] == v["ai_first"]]
        report[f"pair_{style}"] = {
            "n": len(p),
            "acc_picks_human": round(sum(r["picked"] == "human" for r in p) / len(p), 3),
            "ties": sum(r["picked"] == "tie" for r in p),
            "slot1_rate": round(sum(r["winner"] == "1" for r in p) / len(p), 3),
            "order_consistent": f"{len(consistent)}/{len(both)}",
        }
    reps = {}
    for r in load_jsonl(ABS):
        reps.setdefault(r["key"], []).append(r)
    spread = [max(x["human"] for x in v) - min(x["human"] for x in v) for v in reps.values() if len(v) == 3]
    spread_slop = [max(x["slop"] for x in v) - min(x["slop"] for x in v) for v in reps.values() if len(v) == 3]
    report["repeat_range"] = {"n": len(spread), "human_mean_range": round(sum(spread) / max(len(spread), 1), 2),
                              "slop_mean_range": round(sum(spread_slop) / max(len(spread_slop), 1), 2)}
    print(json.dumps(report, indent=2))
    (OUT / f"calib_report{JSUF}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    {"human": step_human, "ai": step_ai, "judge": step_judge, "report": step_report}[sys.argv[1]]()
