"""OpenRouter client and the Opus judge built on the antislop-copywriting rules."""
import json
import os
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = os.environ.get("EXP", "")
OUT = ROOT / "out" / EXP if EXP else ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)
JUDGE = os.environ.get("JUDGE", "anthropic/claude-opus-5.5")
JSUF = os.environ.get("JSUF", "")  # result-file suffix for a second judge
KEY = os.environ.get("OPENROUTER_API_KEY") or Path.home().joinpath(".or_key").read_text().strip()

RULES = (ROOT / "antislop-copywriting.md").read_text().split("---", 2)[2].strip()


def chat(model, messages, max_tokens=1500, temperature=None, retries=6, **extra):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, **extra}
    if temperature is not None:
        body["temperature"] = temperature
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.load(r)
            if "choices" not in data or not data["choices"][0]["message"].get("content"):
                raise RuntimeError(str(data)[:300])
            return data["choices"][0]["message"]["content"], data.get("usage", {})
        except Exception as e:  # noqa: BLE001 - network/provider errors are retried
            if attempt == retries - 1:
                raise
            print(f"retry {attempt + 1}: {e}", flush=True)
            time.sleep(5 * 2 ** attempt)


def parse_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0))


def _system():
    return [{"type": "text", "cache_control": {"type": "ephemeral"}, "text":
             "You are a strict, calibrated judge of English prose. Judge only by the rulebook below "
             "(the antislop copywriting rules). Look for clusters, not isolated tells, and respect "
             "the 'What NOT to flag' section. Fabricated or lost facts are defects even when the "
             "prose sounds more human.\n\n<rulebook>\n" + RULES + "\n</rulebook>"}]


ABS_PROMPT = """Task the text was written for:
<task>{task}</task>
{source_block}
Text to judge:
<text>{text}</text>

Do this:
1. List every AI-writing tell from the rulebook that appears in the text, as {{"pattern": "<rulebook pattern name>", "quote": "<short exact quote>"}}.
2. Facts: compare against {fact_ref}. List facts (numbers, units, dates, names, quotes) that are missing or changed in "lost_facts", and invented specifics not present there in "fabrications".
3. Scores:
   - "slop": 0-10, how much AI slop by the rulebook (0 = none, 10 = saturated).
   - "human": 1-10, how much it reads like a specific person wrote it (voice, rhythm, concrete detail).
   - "task": 1-10, how well it does the job of the task.

Answer with JSON only:
{{"tells": [...], "lost_facts": [...], "fabrications": [...], "slop": int, "human": int, "task": int}}"""

PAIR_PROMPT = """Task:
<task>{task}</task>
{source_block}
Text 1:
<text1>{a}</text1>

Text 2:
<text2>{b}</text2>

Which text better satisfies the rulebook: fewer AI-writing tells, a real human voice, no lost or fabricated facts (compare against {fact_ref}), and still does the task? Ignore length unless it causes a rulebook violation.
Answer with JSON only: {{"winner": "1" | "2" | "tie", "reason": "<one sentence>"}}"""


def _src(source):
    if source:
        return f"\nSource draft the text was rewritten from:\n<source>{source}</source>\n", "the source draft"
    return "", "the task"


def _judge(prompt, attempts=3):
    for attempt in range(attempts):
        content, usage = chat(JUDGE, [{"role": "system", "content": _system()},
                                      {"role": "user", "content": prompt}], max_tokens=4000)
        try:
            return parse_json(content), usage
        except (AttributeError, ValueError):
            if attempt == attempts - 1:
                raise


def judge_abs(task, text, source=None):
    block, ref = _src(source)
    result, usage = _judge(ABS_PROMPT.format(task=task, source_block=block, fact_ref=ref, text=text))
    result["em_dashes"] = text.count("—")
    return result, usage


def judge_pair(task, a, b, source=None):
    block, ref = _src(source)
    return _judge(PAIR_PROMPT.format(task=task, source_block=block, fact_ref=ref, a=a, b=b))


def load_jsonl(path):
    path = Path(path)
    rows = []
    for line in (path.read_text().splitlines() if path.exists() else []):
        try:
            rows.append(json.loads(line))
        except ValueError:  # a line still being written by a concurrent process
            pass
    return rows


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
