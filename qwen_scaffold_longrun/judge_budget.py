"""Separate, locked, conservative $120 ledger for this run's Opus calls."""
import fcntl
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT/'opus_budget.json'
CACHE = ROOT/'judge_cache'
LOCK = threading.Lock()

def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(temporary,path)

def transaction(fn):
    with LOCK, open(str(LEDGER)+'.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        b = json.loads(LEDGER.read_text()) if LEDGER.exists() else {'limit':120.,'spent_with_margin':0.,'pending':{},'requests':0}
        assert b['limit']==120.
        result = fn(b)
        atomic_json(LEDGER,b)
        return result

def call(body):
    assert body['model']=='anthropic/claude-opus-5.5'
    CACHE.mkdir(exist_ok=True)
    payload = json.dumps(body,sort_keys=True,ensure_ascii=False)
    key = hashlib.sha256(payload.encode()).hexdigest()
    path = CACHE/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    # UTF-8 byte count bounds token count conservatively. Includes cache-write
    # input price and output/reasoning cap, plus 10% billing margin.
    estimate = (len(payload.encode())*5e-6+body['max_tokens']*25e-6)*1.1
    def reserve(b):
        if b['spent_with_margin']+sum(b['pending'].values())+estimate>b['limit']:
            raise RuntimeError('Opus budget exhausted before request')
        b['pending'][reservation]=estimate
    # Transport errors and 408/429/5xx are retried. Each attempt reserves its own estimate and a failed
    # attempt's reservation is retained, since it may have been billed. 402 and other errors are final.
    for attempt in range(5):
        reservation = key+'-'+str(time.time_ns())
        transaction(reserve)
        try:
            response = requests.post('https://openrouter.ai/api/v1/chat/completions',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']},json=body,timeout=180)
        except requests.RequestException as exc:
            if attempt==4:raise RuntimeError(f'Judge transport failed: {type(exc).__name__}; reservation retained')
            time.sleep(10*2**attempt);continue
        if response.status_code in (408,429,500,502,503,504) and attempt<4:
            time.sleep(10*2**attempt);continue
        break
    if not response.ok:
        raise RuntimeError(f'Judge HTTP {response.status_code}; reservation retained')
    data = response.json()
    cost = data.get('usage',{}).get('cost')
    if not isinstance(cost,(int,float)) or not math.isfinite(cost) or cost<0:
        raise RuntimeError('Judge returned no valid billed cost; reservation retained')
    def settle(b):
        b['spent_with_margin'] += cost*1.1
        b['pending'].pop(reservation)
        b['requests'] += 1
    transaction(settle)
    atomic_json(path,data)
    return data

def content(data):
    choice = data['choices'][0]
    if choice['finish_reason']!='stop':raise RuntimeError('Judge reply incomplete: '+choice['finish_reason'])
    return json.loads(choice['message']['content'])
