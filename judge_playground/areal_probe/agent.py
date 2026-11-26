"""AReaL agent workflow that runs one Harbor trial per call through our laptop coordinator.

It replaces qwen_verl_native/native_agent.py for an AReaL v2.1.x run (proxy approach, rollout.agent.mode=inline).

How one sample flows:
  1. AReaL's OpenAIProxyWorkflow opens a proxy session and calls run(data, base_url=..., api_key=sk-sess-...).
  2. run() registers that session under a fresh uid in a small local shim on 127.0.0.1:9001, the port the
     agent host already reaches through the to_agents.sh reverse tunnel. The collector keeps calling
     http://host.docker.internal:18974/session/<uid>/v1/..., so the coordinator, collector and tunnels stay as they are.
  3. run() POSTs the usual body {uid, group, task, agent, split, policy, size} to the coordinator
     (127.0.0.1:18975/run on the GPU box, reverse-tunnelled to the laptop) and waits for {'reward', 'infra_failed'}.
  4. The shim forwards each chat completion to AReaL's proxy with the session key. AReaL records the exact
     prompt/output tokens, logprobs and masks (InteractionCache), so there is no re-tokenization here.
  5. run() returns the reward as a float; AReaL puts it on the last completion and exports the session.
     With export_style=concat one linear trajectory becomes one training row.

Infrastructure failures raise InfraFailed. should_reject_failed_sample() then returns True, so
OpenAIProxyWorkflow returns None for that slot: the slot is dropped and its siblings still train
(GroupedRolloutWorkflow keeps a group while at least min_usable_group_size slots survive). This is the
AReaL equivalent of verl's "mask the whole response" for infra_failed trials.

Kept from native_agent.py:
  - the Hermes session-title auxiliary call never reaches the policy proxy, so it is never trained on
    (here it is answered locally with a fixed title, instead of being generated and discarded);
  - a byte-identical repeated request (scaffold retry) gets the cached answer, so it creates no extra branch;
  - group ids get a -cycle<N>-attempt<ID> suffix, as native_agent.py does, so the coordinator groups the 8 slots.
Rewinds (Hermes resending an earlier message list after discarding a response) become retry orphans in
AReaL's conversation tree; rollout.agent.drop_retry_orphans=true removes them, matching native_agent's rewind.
Context rewrites that break the prefix start a new chain; get_episode_reward_for_export() lets AReaL score
every disjoint chain with the episode reward instead of leaving it unscored.

Written against AReaL v2.1.1 (areal/experimental/openai/proxy/workflow.py). Not executed yet.
"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request   # module level: FastAPI resolves string annotations here
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(os.environ.get('ANTISLOP_NATIVE_ROOT', '/workspace/lab/native'))
COORDINATOR = os.environ.get('ANTISLOP_COORDINATOR_URL', 'http://127.0.0.1:18975/run')
SHIM_HOST, SHIM_PORT = '127.0.0.1', int(os.environ.get('ANTISLOP_SHIM_PORT', '9001'))
TITLE_PREFIX = 'You name chat sessions.'
ATTEMPT = uuid.uuid4().hex[:12]

ROUTES: dict[str, dict[str, Any]] = {}     # uid -> {'base': proxy base url, 'key': sk-sess-*, 'last': (signature, response)}
COUNTS: dict[str, int] = {}                # group -> calls so far, for the -cycle<N> suffix
_SHIM: dict[str, Any] = {'server': None, 'lock': None}


class InfraFailed(RuntimeError):
    """The coordinator says the trial broke for infrastructure reasons: drop this slot, keep the group."""


class CoordinatorError(RuntimeError):
    """The coordinator itself failed (budget, circuit open, stop). Not rejected: training fails loudly, as before."""


def _serve_key() -> str:
    return (ROOT / 'serve.key').read_text().strip()


def _proxy_url(base: str, path: str) -> str:
    base = base.rstrip('/')
    if base.endswith('/v1'):
        base = base[:-3]
    return f'{base}/v1/{path}'


def _build_shim():
    app = FastAPI()
    key = _serve_key()

    def check(request: Request, uid: str) -> dict[str, Any]:
        if request.headers.get('authorization') != 'Bearer ' + key:
            raise HTTPException(401)
        if uid not in ROUTES:
            raise HTTPException(404)
        return ROUTES[uid]

    @app.get('/v1/models')
    @app.get('/session/{uid}/v1/models')
    @app.get('/session/{uid}/trial/{trial}/v1/models')
    async def models(uid: str = '', trial: str = ''):
        return {'object': 'list', 'data': [{'id': 'antislop-scaffold', 'object': 'model', 'max_model_len': 65536}]}

    @app.post('/session/{uid}/v1/chat/completions')
    @app.post('/session/{uid}/trial/{trial}/v1/chat/completions')
    async def completion(uid: str, request: Request, trial: str = ''):
        route = check(request, uid)
        body = await request.json()
        messages = body.get('messages') or []
        if messages and messages[0].get('role') == 'system' and str(messages[0].get('content', '')).startswith(TITLE_PREFIX):
            # Hermes' session-title call: answered here, never sent to the policy, never trained.
            return JSONResponse(_title_response(body))
        signature = json.dumps(body, sort_keys=True, ensure_ascii=False)
        async with route['lock']:
            last = route.get('last')
            if last and last[0] == signature:
                return _replay(last[1], body)
            client: httpx.AsyncClient = route['client']
            headers = {'Authorization': 'Bearer ' + route['key'], 'Content-Type': 'application/json'}
            if body.get('stream'):
                # Ask the proxy for a whole response, cache it, and stream it back ourselves.
                upstream = dict(body, stream=False)
                upstream.pop('stream_options', None)
            else:
                upstream = body
            reply = await client.post(_proxy_url(route['base'], 'chat/completions'), json=upstream, headers=headers, timeout=1800)
            if reply.status_code != 200:
                return JSONResponse(status_code=reply.status_code, content=_safe_json(reply))
            data = reply.json()
            route['last'] = (signature, data)
        return _replay(data, body)

    def _replay(data: dict[str, Any], body: dict[str, Any]):
        if not body.get('stream'):
            return JSONResponse(data)
        return StreamingResponse(_chunks(data), media_type='text/event-stream')

    return app


def _safe_json(reply: httpx.Response) -> Any:
    try:
        return reply.json()
    except ValueError:
        return {'error': reply.text[:2000]}


def _title_response(body: dict[str, Any]) -> dict[str, Any]:
    return {'id': 'chatcmpl-title-' + uuid.uuid4().hex, 'object': 'chat.completion', 'created': int(time.time()),
            'model': body.get('model', 'antislop-scaffold'),
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Harbor task'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}


async def _chunks(data: dict[str, Any]):
    choice = data['choices'][0]
    delta = copy.deepcopy(choice.get('message') or {})
    for i, call in enumerate(delta.get('tool_calls') or []):
        call['index'] = i
    for d, reason in ((delta, None), ({}, choice.get('finish_reason'))):
        chunk = {'id': data.get('id'), 'object': 'chat.completion.chunk', 'created': data.get('created', int(time.time())),
                 'model': data.get('model'), 'choices': [{'index': 0, 'delta': d, 'finish_reason': reason}]}
        yield 'data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\n'
    yield 'data: [DONE]\n\n'


async def _ensure_shim() -> None:
    if _SHIM['lock'] is None:
        _SHIM['lock'] = asyncio.Lock()
    async with _SHIM['lock']:
        if _SHIM['server'] is not None:
            return
        import uvicorn
        server = uvicorn.Server(uvicorn.Config(_build_shim(), host=SHIM_HOST, port=SHIM_PORT, log_level='warning'))
        asyncio.get_running_loop().create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.05)
        _SHIM['server'] = server


class HarborCoordinatorAgent:
    """AReaL agent: async run(data, **extra_kwargs) -> float. One call = one Harbor trial = one group slot."""

    def __init__(self, group_size: int = 8, eval_group_size: int = 2, coordinator_url: str = COORDINATOR,
                 timeout_s: float = 2400.0, **_: Any) -> None:
        self.group_size, self.eval_group_size = int(group_size), int(eval_group_size)
        self.coordinator_url, self.timeout_s = coordinator_url, float(timeout_s)

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        base_url, api_key = extra_kwargs.get('base_url'), extra_kwargs.get('api_key')
        if not base_url or not api_key:
            raise CoordinatorError('AReaL did not inject base_url/api_key; rollout.agent.mode must be inline')
        await _ensure_shim()
        meta = data.get('extra_info') or data
        size = self.group_size if meta['split'] == 'train' else self.eval_group_size
        counter = COUNTS.get(meta['group'], 0)
        COUNTS[meta['group']] = counter + 1
        group = f"{meta['group']}-cycle{counter // size}-attempt{ATTEMPT}"
        uid = uuid.uuid4().hex
        client = extra_kwargs.get('http_client') or httpx.AsyncClient(timeout=1800)
        ROUTES[uid] = {'base': base_url, 'key': api_key, 'client': client, 'lock': asyncio.Lock(), 'last': None}
        body = {'uid': uid, 'group': group, 'task': meta['task'], 'agent': meta['agent'], 'split': meta['split'],
                'policy': meta.get('policy', 'lab'), 'size': size}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as coordinator:
                reply = await coordinator.post(self.coordinator_url, json=body, headers={'Authorization': 'Bearer ' + _serve_key()})
            if not reply.is_success:
                raise CoordinatorError('Harbor callback failed: ' + reply.text[:500])
            result = reply.json()
        except httpx.HTTPError as exc:
            raise CoordinatorError(f'coordinator unreachable: {type(exc).__name__}: {exc}') from exc
        finally:
            ROUTES.pop(uid, None)
        if result.get('infra_failed'):
            raise InfraFailed(f'trial {group}/{uid} broke for infrastructure reasons')
        reward = float(result['reward'])           # AReaL accepts float or dict only; an int would be rejected
        if not math.isfinite(reward):
            raise CoordinatorError(f'non-finite reward for {group}/{uid}: {reward!r}')
        return reward

    # ---- hooks OpenAIProxyWorkflow looks up with getattr (AReaL v2.1.x) ----

    @staticmethod
    def classify_proxy_failure(error: Exception, *, context_overflow: bool, interaction_count: int,
                               invalid_tool_json: bool = False) -> str:
        if isinstance(error, InfraFailed):
            return 'system_failure_reject'
        if (context_overflow or invalid_tool_json) and interaction_count > 0 and not isinstance(error, CoordinatorError):
            return 'model_failure_zero'
        return 'unknown_failure_reject'

    @staticmethod
    def should_reject_failed_sample(error: Exception, disposition: str) -> bool:
        # True -> the workflow returns None: this slot is dropped, the rest of the group still trains.
        return isinstance(error, InfraFailed)

    @staticmethod
    def get_episode_reward_for_export(data: dict[str, Any], reward: float, *, export_style: str) -> float | None:
        # Hermes context rewrites start a new chain; score every disjoint chain with the episode reward.
        return reward if export_style == 'concat' and math.isfinite(reward) else None
