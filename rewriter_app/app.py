"""Antislop rewriter: paste any text, get human-sounding rewrites from the GRPO pg120 model, optionally check on Pangram.

    python3 app.py            -> http://127.0.0.1:8765

Env: MODEL_URL (default https://www.rawmodels.ai/v1; prod `antislop` serves the GRPO pg120 weights since 2026-11-16),
MODEL (antislop), MODEL_KEY (bearer; defaults to ~/.config/rawmodels/owner-*.key for rawmodels), CONCURRENCY, PROMPT_FILE (system prompt; re-read on every request), PORT.
The Pangram key is read from ~/.config/rawmodels/pangram.key; each check is paid (~$0.055 per 100 words).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_URL = os.environ.get('MODEL_URL', 'https://www.rawmodels.ai/v1').rstrip('/')
MODEL = os.environ.get('MODEL', 'antislop')
MODEL_KEY = os.environ.get('MODEL_KEY', '')
if not MODEL_KEY and 'rawmodels.ai' in MODEL_URL:
    _keys = sorted(Path('~/.config/rawmodels').expanduser().glob('owner-*.key'))
    MODEL_KEY = _keys[-1].read_text().strip() if _keys else ''
# rawmodels allows one in-flight request per account; a local SGLang server takes many.
_SLOTS = threading.BoundedSemaphore(int(os.environ.get('CONCURRENCY', '1' if 'rawmodels.ai' in MODEL_URL else '8')))
PROMPT_FILE = Path(os.environ.get('PROMPT_FILE', HERE / 'system_prompt.txt'))
PORT = int(os.environ.get('PORT', '8765'))
PG_KEY_FILE = Path('~/.config/rawmodels/pangram.key').expanduser()
PG_URL = 'https://text.external-api.pangram.com/task'
PG_CACHE = HERE / 'pangram_cache'
PG_CACHE.mkdir(exist_ok=True)
SPEND = HERE / 'pangram_spend.json'
_spend_lock = threading.Lock()
CHUNK_WORDS = 200  # Pangram flags long texts far more often; rewrite in ~200-word pieces


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def chunks(text: str, limit: int = CHUNK_WORDS) -> list[str]:
    """Split on paragraph boundaries into pieces of at most ~limit words (long paragraphs split on sentences)."""
    paras = [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
    units: list[str] = []
    for p in paras:
        if words(p) <= limit * 1.3:
            units.append(p)
            continue
        cur = ''
        for s in re.split(r'(?<=[.!?])\s+', p):
            if cur and words(cur) + words(s) > limit:
                units.append(cur)
                cur = s
            else:
                cur = (cur + ' ' + s).strip()
        if cur:
            units.append(cur)
    out, cur = [], ''
    for u in units:
        if cur and words(cur) + words(u) > limit:
            out.append(cur)
            cur = u
        else:
            cur = (cur + '\n\n' + u).strip()
    if cur:
        out.append(cur)
    return out


FREE = ("Do not follow the draft's structure. Write it as one person's offhand take, as if posting a comment: start from "
        "whatever point they care about most, put the other points in a different order, mention them unevenly (some in passing, "
        "one dwelt on), and stop without a summary or conclusion.")


def user_message(draft: str, extra: str, mode: str = 'careful') -> str:
    n = max(15, words(draft))
    msg = (f'Rewrite the draft below so it reads like a person wrote it. Keep every fact and the meaning, keep it at about '
           f'{n} words, and do not add new claims. Write in the same language as the draft. Plain prose only.')
    if mode == 'free':
        msg += ' ' + FREE
    if extra.strip():
        msg += ' ' + extra.strip()
    return msg + '\n\nDraft:\n' + draft


def chat(system: str, user: str, temperature: float, seed: int) -> str:
    body = {'model': MODEL, 'messages': ([{'role': 'system', 'content': system}] if system.strip() else []) +
            [{'role': 'user', 'content': user}], 'max_tokens': 3072, 'temperature': temperature, 'top_p': 0.95,
            'seed': seed, 'chat_template_kwargs': {'enable_thinking': False}}
    headers = {'Content-Type': 'application/json'}
    if MODEL_KEY:
        headers['Authorization'] = 'Bearer ' + MODEL_KEY
    for attempt in range(8):
        req = urllib.request.Request(MODEL_URL + '/chat/completions', data=json.dumps(body).encode(), headers=headers)
        try:
            with _SLOTS, urllib.request.urlopen(req, timeout=600) as r:
                content = (json.load(r)['choices'][0]['message'].get('content') or '').strip()
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and attempt < 7:
                time.sleep(min(20, 2 + 3 * attempt))
                continue
            raise RuntimeError(f'Модель вернула HTTP {e.code}: {e.read()[:200].decode(errors="replace")}') from None
    return re.sub(r'^\s*(?:Here(?:\'s| is)[^\n]*:\s*\n)', '', content)  # drop a stray preface line


def generate(text: str, variants: int, temperature: float, extra: str, mode: str) -> tuple[list[str], int]:
    system = PROMPT_FILE.read_text() if PROMPT_FILE.exists() else ''
    parts = chunks(text)
    jobs = [(v, i, p) for v in range(variants) for i, p in enumerate(parts)]
    base = int(time.time() * 1000) % 1000003
    with cf.ThreadPoolExecutor(min(8, len(jobs))) as pool:
        outs = list(pool.map(lambda j: chat(system, user_message(j[2], extra, mode), temperature, base + 97 * j[0] + j[1]), jobs))
    res = [[] for _ in range(variants)]
    for (v, i, _), o in zip(jobs, outs):
        res[v].append(o)
    return ['\n\n'.join(r) for r in res], len(parts)


def rewrite(text: str, variants: int, temperature: float, extra: str, mode: str = 'careful', auto: int = 0) -> dict:
    """auto > 0: keep generating batches of 3 and checking each on Pangram until one is Human Written or auto tries are used."""
    if not auto:
        vs, n = generate(text, variants, temperature, extra, mode)
        return {'variants': [{'text': t, 'words': words(t)} for t in vs], 'chunks': n, 'source_words': words(text)}
    out, n = [], 1
    while len(out) < auto:
        vs, n = generate(text, min(3, auto - len(out)), temperature, extra, mode)
        with cf.ThreadPoolExecutor(3) as pool:
            checks = list(pool.map(lambda t: _safe_pangram(t), vs))
        out += [{'text': t, 'words': words(t), 'pangram': c} for t, c in zip(vs, checks)]
        if any(c.get('headline') == 'Human Written' for c in checks):
            break
    out.sort(key=lambda v: (v['pangram'].get('headline') != 'Human Written', -(v['pangram'].get('fraction_human') or 0)))
    return {'variants': out, 'chunks': n, 'source_words': words(text), 'tries': len(out),
            'passed': any(v['pangram'].get('headline') == 'Human Written' for v in out), 'spent_total': spent()}


def _safe_pangram(t: str) -> dict:
    try:
        return pangram(t)
    except Exception as e:
        return {'headline': 'error', 'error': str(e)}


def pg_call(url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'x-api-key': PG_KEY_FILE.read_text().strip()})
    for a in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and a < 4:
                time.sleep(5 * 2 ** a)
                continue
            raise RuntimeError(f'Pangram HTTP {e.code}: {e.read()[:200]!r}') from None


def spent() -> float:
    try:
        return json.loads(SPEND.read_text())['usd']
    except Exception:
        return 0.0


def pangram(text: str) -> dict:
    h = hashlib.sha256(text.encode()).hexdigest()[:32]
    done, task_f = PG_CACHE / f'{h}.json', PG_CACHE / f'{h}.task'
    cached = done.exists()
    if not cached:
        if not task_f.exists():
            task_f.write_text(pg_call(PG_URL, {'text': text, 'model': 'pangram-4', 'public_dashboard_link': False})['task_id'])
            with _spend_lock:
                SPEND.write_text(json.dumps({'usd': round(spent() + 0.055 * max(1, -(-words(text) // 100)), 4)}))
        t0 = time.time()
        while True:
            r = pg_call(PG_URL + '/' + task_f.read_text())
            if r.get('stage') == 'STAGE_SUCCESS':
                break
            if r.get('stage') == 'STAGE_FAILED':
                raise RuntimeError('Pangram failed: ' + str(r.get('headline')))
            if time.time() - t0 > 240:
                raise TimeoutError('Pangram still pending; press check again to resume')
            time.sleep(2)
        done.write_text(json.dumps(r, ensure_ascii=False))
    r = json.loads(done.read_text())
    return {'headline': r.get('headline'), 'fraction_human': r.get('fraction_human'), 'fraction_ai': r.get('fraction_ai'),
            'fraction_ai_assisted': r.get('fraction_ai_assisted'), 'cached': cached, 'spent_total': spent()}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype + '; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            return self._send(200, (HERE / 'index.html').read_bytes(), 'text/html')
        if self.path == '/api/status':
            try:
                rq = urllib.request.Request(MODEL_URL + '/models', headers={'Authorization': 'Bearer ' + MODEL_KEY} if MODEL_KEY else {})
                with urllib.request.urlopen(rq, timeout=8) as r:
                    ok = MODEL in r.read().decode()
            except Exception:
                ok = False
            return self._send(200, {'model': MODEL, 'model_up': ok, 'pangram_key': PG_KEY_FILE.exists(),
                                    'pangram_spent': spent(), 'prompt_file': str(PROMPT_FILE)})
        self._send(404, {'error': 'not found'})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            if self.path == '/api/rewrite':
                text = (req.get('text') or '').strip()
                if words(text) < 5:
                    return self._send(400, {'error': 'Нужен текст хотя бы из нескольких слов.'})
                if words(text) > 3000:
                    return self._send(400, {'error': 'Слишком длинно: до 3000 слов за раз.'})
                return self._send(200, rewrite(text, max(1, min(3, int(req.get('variants', 3)))),
                                               max(0.3, min(1.1, float(req.get('temperature', 0.8)))),
                                               str(req.get('extra', ''))[:500],
                                               'free' if req.get('mode') == 'free' else 'careful',
                                               max(0, min(9, int(req.get('auto', 0))))))
            if self.path == '/api/pangram':
                text = (req.get('text') or '').strip()
                if not text:
                    return self._send(400, {'error': 'empty'})
                return self._send(200, pangram(text))
            self._send(404, {'error': 'not found'})
        except urllib.error.URLError as e:
            self._send(502, {'error': f'Модель недоступна ({e.reason}). Проверь туннель к pg120.'})
        except Exception as e:
            self._send(500, {'error': f'{type(e).__name__}: {e}'})


if __name__ == '__main__':
    print(f'Antislop rewriter on http://127.0.0.1:{PORT}  (model {MODEL} at {MODEL_URL})', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
