"""Score texts on the REAL Pangram 4 API (paid, ~$0.055/100 words). Cached per text; never resubmits a pending task.
usage: pangram_check.py <results.json with items[]> <label> [sample]   -> pangram_real/<label>.json"""
import hashlib, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path
KEY = Path('~/.config/rawmodels/pangram.key').expanduser().read_text().strip()
URL = 'https://text.external-api.pangram.com/task'
CACHE = Path(__file__).parent / 'pangram_real' / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)

def call(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'x-api-key': KEY})
    for a in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and a < 4: time.sleep(5 * 2 ** a); continue
            raise RuntimeError(f'Pangram HTTP {e.code}: {e.read()[:300]!r}') from None

def pangram(text):
    h = hashlib.sha256(text.encode()).hexdigest()[:32]
    done, task_f = CACHE / f'{h}.json', CACHE / f'{h}.task'
    if done.exists(): return json.loads(done.read_text())
    if not task_f.exists():
        task_f.write_text(call(URL, {'text': text, 'model': 'pangram-4', 'public_dashboard_link': False})['task_id'])
    t0 = time.time()
    while True:
        r = call(URL + '/' + task_f.read_text())
        if r.get('stage') == 'STAGE_SUCCESS': break
        if r.get('stage') == 'STAGE_FAILED': raise RuntimeError('Pangram failed: ' + str(r.get('headline')))
        if time.time() - t0 > 300: raise TimeoutError('pending; rerun to resume')
        time.sleep(2)
    done.write_text(json.dumps(r, ensure_ascii=False)); return r

src, label = sys.argv[1], sys.argv[2]
sample = int(sys.argv[3]) if len(sys.argv) > 3 else 0
items = json.load(open(src)); items = items['items'] if isinstance(items, dict) else items
seen, picked = {}, []
for x in items:  # one text per task: the sample-th occurrence
    k = x.get('task_id') or x['question']; seen[k] = seen.get(k, -1) + 1
    if seen[k] == sample and x['text'].strip(): picked.append(x)
out = []
for x in picked:
    r = pangram(x['text'])
    keep = {k: r.get(k) for k in ('headline', 'prediction', 'prediction_short', 'fraction_ai', 'fraction_ai_assisted', 'fraction_human', 'version')}
    out.append({'kind': x.get('kind') or x.get('task'), 'task_id': x.get('task_id'), 'emu_score': x.get('score'), 'emu_slop': x.get('slop'),
                'task': x.get('task') if isinstance(x.get('task'), int) else None, 'words': len(x['text'].split()), 'pangram': keep, 'text': x['text']})
    print(f"{out[-1]['kind']:8} emu={x.get('score')} words={out[-1]['words']:4} -> {keep.get('headline')} | ai={keep.get('fraction_ai')} asst={keep.get('fraction_ai_assisted')} human={keep.get('fraction_human')}", flush=True)
Path(__file__).parent.joinpath('pangram_real', f'{label}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
w = sum(o['words'] for o in out); print('texts', len(out), 'words', w, f'~${0.055 * sum(max(1, -(-o["words"] // 100)) for o in out):.2f}')
