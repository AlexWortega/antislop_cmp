"""Long (500-900 word) rewrite drafts for checking long texts; round-2 hard rewrite drafts: polished AI-style texts in the genres pg120 could not humanize (DeepSeek, ~$0.003 each)."""
import concurrent.futures as cf, hashlib, json, os, random, re, sys, urllib.request
from pathlib import Path
KEY = sorted(Path('~/.config/rawmodels').expanduser().glob('openrouter-*.key'))[-1].read_text().strip()
GENRES = {
 'story': ('a literary fiction short story (third or first person) with vivid imagery, similes, dialogue and a quiet emotional ending', 6, (500, 900)),
 'essay': ('an argumentative or reflective essay with a clear thesis, balanced points and a conclusion', 6, (500, 900)),
 'explainer': ('a clear explainer of how something works or a historical topic, with an intro, ordered points and a summary', 4, (500, 900)),
 'personal': ('a personal blog post about an experience, polished and reflective, with some figurative language', 4, (500, 900)),
}
TOPICS = [l.strip() for l in open(Path(__file__).with_name('topics.txt')) if l.strip()]
def ask(genre, desc, words, topic, seed):
    body = {'model': 'deepseek/deepseek-v4.1-flash', 'temperature': 1.0, 'max_tokens': 8000, 'seed': seed,
            'messages': [{'role': 'user', 'content': f'Write {desc}. Topic: {topic}. Length: about {words} words. Write it the way a capable AI assistant typically writes: polished, well organized, fluent. Plain prose only (no title, headings, lists or markdown). Output only the text.'}]}
    req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    text = (out['choices'][0]['message'].get('content') or '').strip()
    return text, (out.get('usage') or {}).get('cost') or 0
rng = random.Random(1117); jobs = []
for g, (desc, n, (lo, hi)) in GENRES.items():
    for i in range(n):
        jobs.append((g, desc, rng.randrange(lo, hi, 10), rng.choice(TOPICS), rng.randrange(10**6)))
def run(j):
    g, desc, w, topic, seed = j
    for _ in range(3):
        try:
            t, c = ask(g, desc, w, topic, seed)
            if 0.6 * w <= len(t.split()) <= 1.6 * w and not re.search(r'^#|\*\*', t, re.M): return g, t, c
        except Exception as e:
            print('retry', type(e).__name__, file=sys.stderr)
        seed += 1
    return g, None, 0
with cf.ThreadPoolExecutor(16) as p: res = list(p.map(run, jobs))
rows, cost = [], 0
for g, t, c in res:
    cost += c
    if not t: continue
    n = len(re.findall(r"\b[\w'-]+\b", t))
    q = (f'Rewrite the draft below so it reads like a person wrote it. Keep every fact and the meaning, keep it at about {n} words, '
         f'and do not add new claims. Plain prose only.\n\nDraft:\n{t}')
    rows.append({'question': q, 'task': 'rewrite', 'kind': 'rewrite_' + g, 'task_id': 'long_' + hashlib.sha256(t.encode()).hexdigest()[:12],
                 'target_words': str(n), 'allow_markdown': 'False', 'lang': 'en', 'source': 'long_deepseek', 'source_text': ''})
random.Random(3).shuffle(rows)
Path('long_val.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
import collections; print(len(rows), 'drafts', collections.Counter(r['kind'] for r in rows), f'cost ${cost:.3f}')
