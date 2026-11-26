"""Rewriter-prompt harness for the GRPO ck120 NVFP4 model (served as `pg120` on 127.0.0.1:30001 via ssh tunnel).

  python3 rw.py eval  <prompt.txt> <opt|val|final> [--n 2] [--temp 0.7] [--limit K] [--pangram]   -> runs/<stem>_<split>.json
  python3 rw.py spend

Scores per text: emulator ai score 0-100 (Opus Pangram emulator; <50 ~ "human"), DeepSeek task check 0-10, length ok
(0.6-1.6x target words), and with --pangram the REAL Pangram 4 headline (paid; capped). Prompt file = the SYSTEM prompt.
The user message is the task's own rewrite instruction + draft, unchanged. Spending caps are hard (raise BudgetExhausted).
"""
import argparse, concurrent.futures as cf, json, os, statistics as st, sys, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
GRPO = HERE.parent / 'qwen_pangram_grpo'
os.environ.setdefault('EMU_RUN_DIR', str(HERE / 'runs' / 'or'))
os.environ.setdefault('OR_CAP_USD', '12')
if 'OPENROUTER_API_KEY' not in os.environ:
    os.environ['OPENROUTER_API_KEY'] = sorted(Path('~/.config/rawmodels').expanduser().glob('openrouter-*.key'))[-1].read_text().strip()
sys.path.insert(0, str(GRPO / 'code'))
import emu_reward as E  # noqa: E402
PG_CAP = float(os.environ.get('PANGRAM_CAP_USD', '12'))
PG_LEDGER = HERE / 'runs' / 'pangram_ledger.json'
import threading
_PG_LOCK = threading.Lock()

def pg_spent():
    try: return json.loads(PG_LEDGER.read_text())['usd']
    except Exception: return 0.0

def pangram(text):
    import importlib.util, hashlib
    cache = GRPO / 'pangram_real' / 'cache' / (hashlib.sha256(text.encode()).hexdigest()[:32] + '.json')
    with _PG_LOCK:
      if not cache.exists():
        cost = 0.055 * max(1, -(-len(text.split()) // 100))
        if pg_spent() + cost > PG_CAP:
            raise E.BudgetExhausted(f'Pangram cap ${PG_CAP} reached (spent ${pg_spent():.2f})')
        PG_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        PG_LEDGER.write_text(json.dumps({'usd': round(pg_spent() + cost, 4), 'cap': PG_CAP}))
    spec = importlib.util.spec_from_file_location('pc', GRPO / 'pangram_check.py')
    src = (GRPO / 'pangram_check.py').read_text().split('src, label = sys.argv[1]')[0]
    ns = {'__file__': str(GRPO / 'pangram_check.py')}; exec(compile(src, 'pangram_check', 'exec'), ns)
    r = ns['pangram'](text)
    return {'headline': r.get('headline'), 'fraction_ai': r.get('fraction_ai'), 'fraction_human': r.get('fraction_human')}

def generate(system, user, seed, temp):
    msgs = ([{'role': 'system', 'content': system}] if system.strip() else []) + [{'role': 'user', 'content': user}]
    body = {'model': 'pg120', 'messages': msgs, 'max_tokens': 3072, 'temperature': temp, 'top_p': 0.95, 'seed': seed,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request('http://127.0.0.1:30001/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        return (json.load(r)['choices'][0]['message'].get('content') or '').strip()

def one(system, row, k, temp, use_pg):
    text = generate(system, row['question'], 1000 + k, temp)
    w, t = E.words(text), int(row['target_words'])
    emu = E.emu_score(text) if w >= 15 else 100
    task = E.task_score(row['question'], text)
    out = {'task_id': row['task_id'], 'sample': k, 'words': w, 'target': t, 'length_ok': 0.6 * t <= w <= 1.6 * t,
           'emu': emu, 'task': task, 'text': text}
    if use_pg:
        out['pangram'] = pangram(text)
    out['pass_emu'] = emu is not None and emu < 50 and (task or 0) >= 6 and out['length_ok']
    if use_pg:
        out['pass_pangram'] = out['pangram']['headline'] == 'Human Written' and (task or 0) >= 6 and out['length_ok']
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('prompt', nargs='?'); ap.add_argument('split', nargs='?')
    ap.add_argument('--n', type=int, default=2); ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--pangram', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'spend':
        print(json.dumps({'openrouter_usd': E.spent(), 'openrouter_cap': E.CAP, 'pangram_usd': pg_spent(), 'pangram_cap': PG_CAP})); return
    system = Path(a.prompt).read_text()
    rows = [json.loads(l) for l in (HERE / 'data' / f'{a.split}.jsonl').read_text().splitlines() if l.strip()]
    rows = rows[:a.limit] if a.limit else rows
    jobs = [(r, k) for r in rows for k in range(a.n)]
    with cf.ThreadPoolExecutor(8) as p:
        res = list(p.map(lambda j: one(system, j[0], j[1], a.temp, a.pangram), jobs))
    n = len(res)
    summ = {'prompt': a.prompt, 'split': a.split, 'n': n, 'temp': a.temp,
            'pass_emu': sum(r['pass_emu'] for r in res) / n,
            'emu_mean': st.mean(r['emu'] for r in res if r['emu'] is not None),
            'emu_human_rate': sum(r['emu'] is not None and r['emu'] < 50 for r in res) / n,
            'task_mean': st.mean(r['task'] for r in res if r['task'] is not None),
            'length_ok': sum(r['length_ok'] for r in res) / n}
    if a.pangram:
        summ['pangram_human_rate'] = sum(r['pangram']['headline'] == 'Human Written' for r in res) / n
        summ['pass_pangram'] = sum(r['pass_pangram'] for r in res) / n
    out = HERE / 'runs' / f"{Path(a.prompt).stem}_{a.split}{'_pg' if a.pangram else ''}.json"
    out.write_text(json.dumps({'summary': summ, 'items': res}, ensure_ascii=False, indent=1))
    print(json.dumps(summ, indent=1)); print('saved', out)

if __name__ == '__main__':
    main()
