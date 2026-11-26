"""Process-safe request reservations; failed/unknown requests retain reservations."""
import fcntl
import json
import uuid
import time
import asyncio
from contextlib import contextmanager
from pathlib import Path

PATH = Path(__file__).resolve().parent / 'cache' / 'budget.json'

@contextmanager
def ledger():
    PATH.parent.mkdir(exist_ok=True)
    with open(PATH.with_suffix('.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(PATH.read_text()) if PATH.exists() else {
            'limits': {'openrouter': 100.0, 'pangram': 100.0},
            'spent': {'openrouter': 0.000132, 'pangram': 0.20}, 'pending': {}}
        yield state
        tmp = PATH.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(PATH)

def reserve(provider, amount):
    with ledger() as state:
        total = state['spent'][provider] + sum(x['amount'] for x in state['pending'].values() if x['provider'] == provider)
        if total + amount > state['limits'][provider]:
            raise RuntimeError(f'{provider} budget exhausted: reserved/spent {total:.4f}; next {amount:.4f}')
        token = uuid.uuid4().hex
        state['pending'][token] = {'provider': provider, 'amount': amount}
    return token

def settle(token, amount):
    with ledger() as state:
        item = state['pending'].pop(token)
        state['spent'][item['provider']] += amount

def settle_once(token, amount):
    with ledger() as state:
        item = state['pending'].pop(token, None)
        if item is not None:
            state['spent'][item['provider']] += amount

def openrouter_reserve(body):
    prices = json.loads((Path(__file__).parent/'pricing_models.json').read_text())
    if body['model'] not in prices:
        raise ValueError('Unbudgeted model')
    price = prices[body['model']]
    # UTF-8 byte count upper-bounds text token count, with framing cushion.
    size = len(json.dumps(body).encode()) + 2048
    output = body.get('max_tokens', body.get('max_completion_tokens', body.get('max_output_tokens')))
    if output is None:
        raise ValueError('Explicit max_tokens required for bounded API calls')
    return reserve('openrouter', 1.10*(size*price['prompt'] + output*price['completion']))

def openrouter_settle(token, result):
    cost = result.get('usage', {}).get('cost')
    if cost is not None:
        settle(token, float(cost)*1.10)

def throttle(model='anthropic/claude-opus-5.5'):
    with ledger() as state:
        now = time.time()
        field = 'next_request' if model == 'anthropic/claude-opus-5.5' else 'next_request_' + model
        slot = max(now, state.get(field, 0))
        state[field] = slot + 3.5
    time.sleep(max(0, slot - now))

def install_shinka_guard():
    """Cover both SDK transports and Chat Completions / Responses endpoints."""
    import importlib
    def patch_transport(httpx):
        original_sync, original_async = httpx.Client.send, httpx.AsyncClient.send
        def reservation(request):
            if request.url.host == 'openrouter.ai' and request.method == 'POST' and request.url.path.endswith(('/chat/completions', '/responses')):
                return openrouter_reserve(json.loads(request.content))
            return None
        def sync(client, request, *args, **kwargs):
            if request.url.host == 'openrouter.ai': throttle()
            token = reservation(request)
            response = original_sync(client, request, *args, **kwargs)
            if token and response.is_success:
                response.read()
                openrouter_settle(token, response.json())
            elif token and response.status_code == 429:
                settle(token, 0)
            return response
        async def asynchronous(client, request, *args, **kwargs):
            if request.url.host == 'openrouter.ai': await asyncio.to_thread(throttle)
            token = reservation(request)
            response = await original_async(client, request, *args, **kwargs)
            if token and response.is_success:
                await response.aread()
                openrouter_settle(token, response.json())
            elif token and response.status_code == 429:
                settle(token, 0)
            return response
        httpx.Client.send, httpx.AsyncClient.send = sync, asynchronous
    for name in ['httpx', 'httpx2']:
        try: module = importlib.import_module(name)
        except ImportError: continue
        patch_transport(module)
