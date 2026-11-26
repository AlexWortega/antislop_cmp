"""Writing-only samples from the live Rawmodels Antislop API, no automatic retries."""
import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'cmp_hemmingway/out'
SECRETS = Path('/Users/alexwortega/Desktop/side_projects/rawmodels/humanizer/.local/secrets.json')
URL = 'https://www.rawmodels.ai/v1/chat/completions'
lock = threading.Lock()
last_start = 0.


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def call(row, mode):
    global last_start
    dest = ROOT / mode / (row['id'] + '.json')
    if dest.exists():
        value = json.loads(dest.read_text())
        assert value['status'] == 'complete', 'Prior failed request requires explicit investigation'
        return value
    key = json.loads(SECRETS.read_text())['RAWMODELS_API_KEY']
    body = {'model': 'antislop', 'messages': [{'role': 'user', 'content': row['assignment']}],
        'stream': True, 'max_tokens': 2048 if mode == 'off' else 8192,
        'temperature': 1., 'top_p': .95, 'top_k': 20, 'seed': 271828}
    if mode == 'off':
        body['reasoning'] = {'enabled': False}
    request_sha = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    record = {'id': row['id'], 'domain': row['domain'], 'language': row['language'],
        'mode': mode, 'request': body, 'request_sha256': request_sha, 'status': 'requested'}
    write(dest, record)
    with lock:
        time.sleep(max(0., 3.2 - (time.monotonic() - last_start)))
        last_start = time.monotonic()
    started = time.monotonic()
    chunks = []
    try:
        with requests.post(URL, headers={'Authorization': 'Bearer ' + key}, json=body,
                stream=True, timeout=(15, 45)) as response:
            if not response.ok:
                record.update(status='http_error', http_status=response.status_code)
                write(dest, record)
                response.raise_for_status()
            response.encoding = 'utf-8'
            record.update(response_content_type=response.headers.get('Content-Type'),
                response_request_id=response.headers.get('X-Request-ID'))
            write(dest, record)
            for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if time.monotonic() - started > 540:
                    raise TimeoutError('Total response deadline exceeded')
                if not line or not line.startswith('data:'):
                    continue
                payload = line[5:].strip()
                if payload == '[DONE]':
                    break
                chunk = json.loads(payload)
                if 'error' in chunk:
                    record.update(status='stream_error', error_code=chunk['error'].get('code'))
                    write(dest, record)
                    raise RuntimeError('Upstream stream error: ' + str(record['error_code']))
                chunks.append(chunk)
        text, reasoning, usage, finish, response_ids = '', '', {}, None, set()
        for chunk in chunks:
            if chunk.get('id'):
                response_ids.add(chunk['id'])
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                delta = choice.get('delta', {})
                text += delta.get('content') or ''
                reasoning += delta.get('reasoning') or delta.get('reasoning_content') or ''
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
        if finish != 'stop' or not text.strip():
            raise ValueError('Truncated or empty answer')
        record.update(status='complete', text=text, reasoning=reasoning,
            finish_reason=finish, usage=usage, seconds=time.monotonic()-started,
            response_ids=sorted(response_ids), raw_chunks=chunks)
        write(dest, record)
        print(mode, row['id'], 'complete', len(text.split()), 'words',
            'reasoning_chars', len(reasoning), 'seconds', round(record['seconds'], 1), flush=True)
        return record
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__,
            seconds=time.monotonic()-started, partial_chunks=chunks)
        write(dest, record)
        raise


def generate():
    rows = [json.loads(s) for s in (SOURCE / 'sbs/examples.jsonl').read_text().splitlines()]
    assert len(rows) == 24 and all(r['domain'] != 'code' for r in rows)
    # First real task also warms the endpoint; it is retained in the fixed sample.
    call(rows[0], 'default')
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda row: call(row, 'default'), rows))
    write(ROOT / 'default_generations.json', {'model': 'rawmodels.ai/antislop',
        'checkpoint_revision': None, 'checkpoint_note': 'Live API does not expose a weight revision',
        'generation': {'temperature': 1., 'top_p': .95, 'top_k': 20, 'seed': 271828,
            'max_tokens': 8192, 'reasoning': 'provider default; enabled in observed responses'},
        'generations': [{k: r[k] for k in ('id', 'domain', 'language', 'text', 'finish_reason', 'usage', 'seconds')} for r in results],
        'source_input_sha256': hashlib.sha256((SOURCE/'sbs/examples.jsonl').read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    selected = [next(r for r in rows if r['language'] == lang and r['domain'] == domain)
                for lang in ('en', 'ru') for domain in ('tweets', 'blogs')]
    defaults = [next(g for g in results if g['id']==r['id']) for r in selected]
    write(ROOT / 'default_diagnostics.json', {'n_tasks': 4,
        'selection': 'First selected tweet and blog in each language; reused from full default run, no extra call',
        'generations': defaults})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['generate'])
    args = p.parse_args()
    globals()[args.action]()
