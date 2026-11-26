"""Training run on rented Vast machines, driven by the lab config.

Machines:
  GPU box     verl trains and serves the policy. In async mode (verl fully_async_policy) one GPU keeps generating
              while the other trains, and weights are synced without stopping generation. In sync mode verl
              main_ppo runs on one GPU.
  agent host  a cheap Vast VM with Docker where Hermes and Pi run Harbor tasks, many at a time. It reaches the
              policy through an SSH tunnel the GPU box opens to it, so model traffic does not pass through this
              laptop. With agents.where = local the containers run here in colima instead.
  this laptop the runner: rents, uploads, starts, watches, judges. The agent loop on the GPU asks the coordinator
              here for each trial's reward through a reverse tunnel; the coordinator starts the trial on the agent
              host and scores finished groups with lab_core.score_group, the judges the UI tests.

If this laptop disappears, a guard on each rented machine destroys it after `dead_man_minutes` without a
heartbeat, so money stops. Adapters are exported from verl checkpoints and backed up to a private Hugging Face
repository; a full Hub never stops training: old backups are deleted, history is squashed, and if that is not
enough backups pause with a warning while training continues. Publishing (adapter or merged model) is a
request from the UI.

One run lives in lab_state/vast/<run_id>/: config.json, state.json, events.jsonl (everything the UI shows),
rollouts/, checkpoints/. The runner is a separate process: python3 vast_core.py <run_dir>.
A failed stage does not kill the run: the runner writes the error with the log tail that explains it, keeps the
machines, and waits for RESUME (retry with the current config; verl continues from its last checkpoint), STOP,
or the idle deadline. Remote work runs under nohup and is polled, so a dropped SSH session does not lose it.
"""
import collections
import concurrent.futures
import json
import math
import os
import random
import re
import secrets
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import lab_core as core

ROOT = Path(__file__).resolve().parent
RUNS = core.STATE / 'vast'
REMOTE = '/workspace/lab'            # on the GPU box
NATIVE = REMOTE + '/native'
VENV = REMOTE + '/vendor/verl/.venv/bin'
AG = '/root/lab'                     # on the agent host
AGENT_PORT = 18974                   # agent host: the policy proxy, forwarded from the GPU box
API = 'https://console.vast.ai/api/v0'

DEFAULT_VAST = {
    'gpu': {
        'num_gpus': 2,               # async needs 2 (one trains, one generates); sync runs on 1
        'name_contains': 'H200',     # '' = any GPU that fits the memory floor
        'min_gpu_ram_gb': 140,       # per GPU
        'min_cpu_ram_gb': 200,       # merging a 27B model for publishing needs about 60 GB
        'max_price_per_hour': 9.5,   # for the whole machine
        'min_reliability': 0.98,
        'min_inet_mbps': 500,
        'min_cuda': 13,
        'offer_id': 0,               # non-zero: rent exactly this offer
        'image': 'nvidia/cuda:13.0.2-devel-ubuntu24.04',
        'disk_gb': 300,
    },
    'agents': {
        'where': 'vast',             # vast: a separate Vast VM with Docker; local: colima on this machine
        'min_cpu': 32, 'min_ram_gb': 64, 'max_price_per_hour': 0.8, 'min_reliability': 0.97, 'min_inet_mbps': 300,
        'disk_gb': 150, 'offer_id': 0, 'image': 'docker.io/vastai/kvm:ubuntu_terminal', 'setup_timeout_min': 60,
    },
    'limits': {
        'max_hours': 8,
        'gpu_budget_usd': 60,        # GPU box and agent host together
        'idle_destroy_minutes': 45,  # a run paused on an error destroys its machines after this long
        'dead_man_minutes': 30,      # machines destroy themselves when this laptop is silent this long
        'destroy_on_finish': True,
    },
    'model': {
        'hf_id': 'rawmodels/Qwen3.8-27B-antislop',
        'revision': '395c2c057ac4d7d6fa333dd9774a652039ebddb6',
    },
    'serve': {
        'context': 81920,            # at least prompt + response length
        'max_seqs': 64, 'max_batched_tokens': 8192,
        'gpu_util': 0.85,            # share of the generation GPU for vLLM; on one shared GPU (sync) use 0.5
        'enforce_eager': False, 'prefix_caching': True,
        'mtp_tokens': 0,             # draft tokens from the model's own MTP head (speculative decoding); 0 = off
    },
    'train': {
        'mode': 'async',             # async: fully_async_policy, 2 GPUs. sync: main_ppo, 1 GPU
        'steps': 150,                # optimizer steps
        'prompts_per_step': 8, 'samples_per_group': 8, 'agents': ['hermes', 'pi'],
        'lr': 3e-5, 'lora_rank': 32, 'lora_alpha': 64, 'clip_ratio': 0.2,
        'clip_ratio_high': 0.28,     # DAPO clip-higher: rare good tokens may grow more, keeps entropy from collapsing
        'temperature': 1.0,          # sampling temperature of training rollouts (eval stays at 1.0)
        'loss_agg_mode': 'token-mean',  # or seq-mean-token-sum-norm (Dr. GRPO); both weigh every token alike within a batch
        'kl_coef': 0.0,              # 0 = no KL loss and no reference pass (-14 s a step); at 0.01 the KL term was 1% of the loss
        'updates_per_step': 2,       # optimizer steps per batch of answers; with 1 the PPO clip never acts (clipfrac was 0)
        'std_normalize': False,      # False = Dr. GRPO: near-tied groups, where the judge is least repeatable, are not blown up
        'bypass_old_logprob': True,  # old log-probs from vLLM instead of a trainer pass (-16 s a step; mismatch was 0.017)
        'max_prompt_length': 8192,
        'max_response_length': 65536,  # whole multi-turn trajectory; 8192 cut off long Hermes edit loops
        # Hermes exits 1 once it uses more turns than this, and the trial is dropped: with 4, 121 of 259 Hermes trials
        # (all with 5 turns, 115 with a finished answer) never reached training on run 1111-164531.
        'max_turns': 8, 'trial_timeout_sec': 480,
        'trial_slots': 64,           # containers at once; below prompts x samples a step runs in two waves (twice the time)
        'agent_schedule': 'by_step', # by_step: one agent per step (sync waits for the slowest); mixed: both in every step
        'save_freq': 10, 'test_freq': 25, 'eval_tasks': 20, 'eval_samples': 2,
        # async: share of a step that may come from older weights. It also caps generation: at most
        # prompts x (1 + threshold) prompts run at once (0.5 gave 3 of 4 prompts in flight, 28-38% rollout use).
        'staleness_threshold': 1.0,
        'remove_padding': True,
        # Pack trajectories into micro-batches of up to this many real tokens (0 = one trajectory per micro-batch).
        # Packing was broken for Qwen3.5 until the qwen3_5.py patch: transformers 5.16 wraps causal_conv1d_fn in a
        # hub-kernel shim that silently drops kwargs its torch fallback lacks, so seq_idx never reached the conv and the
        # first 3 tokens of every packed trajectory saw the tail of the previous one (ppo_kl 0.06-0.18 vs 0.0004-0.002
        # without packing, run 1111-200356). Packed == single to 1e-6 on CPU (qwen_verl_pilot/pack_logprob_check.py).
        # max_ppo_kl below pauses the run if it ever regresses on the GPU.
        'micro_tokens': 32768,       # 81920 OOMed in lm_head: logits for the whole packed batch (28 GiB) on top of 133 GiB, run 1111-235058
        # Pause the run when the median of the last 3 actor/ppo_kl exceeds this (0 = off). Here ppo_kl is the gap
        # between vLLM's and the trainer's logprobs on the same weights: healthy ~0.001, broken packing gave 0.06-0.18.
        'max_ppo_kl': 0.02,
        'sync_every': 1,             # async: optimizer steps between weight syncs to vLLM
        # async: verl cancels in-flight agent loops at a weight sync to resume them later; an external agent (Harbor on
        # the VM) cannot be resumed, the cancel crashed the rollouter (run 1111-164531). Keep False.
        'partial_rollout': False,
        'cpu_offload': False,        # sync on one GPU needs True to fit trainer and vLLM together (slow)
        'auto_retries': 2,           # restarts from the last checkpoint before the run pauses for you
        'download_every': 50,        # steps between adapter downloads to this machine
        'seed': 20261110,
        'extra_overrides': '',       # extra hydra overrides for verl, space-separated; they win over everything above
    },
    'hf': {
        'token_env': 'HF_BACKUP_TOKEN',
        'backup': True,
        'backup_repo': '',           # '' = <your account>/antislop-lab-<run id>, private
        'backup_every': 10,          # steps; a backup is the exported adapter, small
        'keep': 2,                   # newest backups kept on the Hub; older ones are deleted and history squashed
    },
    'local': {
        'ssh_key': '~/.ssh/qwen_verl_pilot_20261101',
        'python': 'qwen_scaffold_longrun/.venv/bin/python',     # needs harbor (local agents, local publishing)
        'collector': 'qwen_scaffold_longrun/repo/scaffold_training/hermes_collect.py',
        'requirements': 'qwen_scaffold_longrun/local_environment.txt',
        'agent_images': 'qwen_verl_native/cloud/agent_images',
        'docker_plugins': 'qwen_verl_native/cloud/bin',
        'native': 'qwen_verl_native',                           # native_agent.py, agent.yaml, native_compat.py, cap_grade.py
        'verl': 'qwen_verl_pilot/vendor/verl',
        'tasks': 'tasks_v2/rows.jsonl',
        'task_root': 'tasks_v2/harbor',
        'docker_host': 'unix://~/.colima/antislop-pilot/docker.sock',
        'tunnel_port': 18974, 'coordinator_port': 18975,
        'agent_versions': {'pi': '1.4.0', 'hermes': 'v0.22.5'},
        'bootstrap_timeout_min': 90,
    },
}

# What a known failure means and what to do about it. First match wins.
HINTS = [
    (r'CUDA out of memory|OutOfMemoryError', 'Не хватило памяти GPU. Уменьшите max_response_length, max_seqs или gpu_util и нажмите «Продолжить».'),
    (r'SIGKILL|OOM killer|Raylet is terminated|Cannot allocate memory', 'Процесс убит системой: кончилась оперативная память. Возьмите оффер с большей RAM или выключите cpu_offload.'),
    (r'No space left on device', 'На инстансе кончился диск. Нужен инстанс с большим disk_gb или реже чекпойнты (save_freq).'),
    (r'Budget of \$', 'Исчерпан бюджет судьи. Поднимите «Бюджет, $» на вкладке «Судья» и нажмите «Продолжить».'),
    (r'HTTP 402', 'У провайдера судьи кончились деньги (402). Пополните баланс и нажмите «Продолжить».'),
    (r'Logprob mismatch', 'Тренер и vLLM считают разные logprobs на одних весах. Чаще всего это упаковка: поставьте «токенов в микробатче» = 0 и нажмите «Продолжить».'),
    (r'Circuit open', 'Больше половины последних попыток агента упали: проверьте агент-хост, Docker и туннель. Причины — в строках coordinator выше.'),
    (r'Could not override|Key .* is not in struct|ConfigCompositionException|MissingMandatoryValue|hydra', 'verl не принял конфиг. Проверьте поле «Доп. overrides» и параметры обучения.'),
    (r'429|Too Many Requests|rate.?limit', 'Hugging Face ограничил скорость скачивания. «Продолжить» повторит; укажите токен HF, с ним лимит выше.'),
    (r'Permission denied \(publickey', 'Vast не принял SSH-ключ. Добавьте публичный ключ в аккаунт Vast (Account → SSH Keys) или укажите другой ключ.'),
    (r'Connection refused|Connection timed out|Operation timed out|No route to host|Connection closed', 'Нет связи с инстансом. Если он жив в списке инстансов, нажмите «Продолжить»; если пропал, «Продолжить» возьмёт новый.'),
    (r'Cannot connect to the Docker daemon|docker\.sock', 'Docker не отвечает. Для локальных агентов запустите colima (colima start antislop-pilot); для агент-хоста — «Продолжить» перезапустит настройку.'),
]
HF_HINT = ('Hugging Face: кончилось приватное хранилище. Старые бэкапы удалены и история сжата, но места всё равно нет. '
           'Бэкапы на HF приостановлены, обучение идёт, адаптеры скачиваются на этот компьютер. Освободите место на HF '
           'или укажите другой репозиторий и нажмите «Возобновить бэкапы».')
FATAL = ('Budget of $', 'HTTP 402', 'Circuit open', 'Stop requested', 'Logprob mismatch')
METRIC_KEYS = ('rewards/mean', 'score/mean', 'pg_loss', 'kl_loss', 'ppo_kl', 'grad_norm', 'pg_clipfrac', 'actor/lr', 'timing_s/', 'timing_per_token',
               'perf/', 'response_length/', 'prompt_length/', 'num_turns/', 'staleness', 'partial_ratio', 'val-', 'memory')


def explain(text):
    for pattern, hint in HINTS:
        if re.search(pattern, text):
            return hint
    return ''


def expand(path):
    """Absolute and ~ paths as they are; anything else is relative to the project root."""
    path = os.path.expanduser(str(path))
    return Path(path) if os.path.isabs(path) else core.PROJECT / path


def rollout_prompts(steps, per_step):
    """Prompts the async rollouter may start. verl stops it after exactly this many, but stale and failed groups never
    reach the trainer: with steps x per_step the trainer ran dry at step 91 of 100 (run 1111-235058). The trainer
    still stops at total_training_steps, so the spare 15% only costs the groups in flight at the end."""
    return math.ceil(steps * per_step * 1.15)


def packing_overrides(train):
    tokens = int(train.get('micro_tokens', 0))
    if tokens <= 0:
        return ['actor_rollout_ref.actor.use_dynamic_bsz=False']
    return ['actor_rollout_ref.actor.use_dynamic_bsz=True', f'actor_rollout_ref.actor.ppo_max_token_len_per_gpu={tokens}',
            'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True', f'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={tokens}',
            'actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True', f'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={tokens}']


def mtp_overrides(serve):
    """Speculative decoding with the checkpoint's MTP head. Rollout only: the trainer never trains the head, and the
    weight sync sends the main model, so vLLM keeps the base drafter. Without mtp.enable verl zeroes
    mtp_num_hidden_layers in the config and vLLM starts with speculative_config=None (run 1111-164531)."""
    tokens = int(serve.get('mtp_tokens', 0))
    if tokens <= 0:
        return []
    return ['actor_rollout_ref.model.mtp.enable=True', 'actor_rollout_ref.model.mtp.enable_rollout=True',
            'actor_rollout_ref.model.mtp.enable_train=False', 'actor_rollout_ref.model.mtp.method=mtp',
            f'actor_rollout_ref.model.mtp.num_speculative_tokens={tokens}']


def validate_vast(v):
    train, gpu = v['train'], v['gpu']
    if train['mode'] not in ('async', 'sync'):
        raise ValueError('vast.train.mode must be async or sync')
    if train['mode'] == 'async' and gpu['num_gpus'] < 2:
        raise ValueError('Async-режим требует 2 GPU: один обучает, второй генерирует. Поставьте num_gpus = 2 или режим sync.')
    if train['mode'] == 'sync' and gpu['num_gpus'] == 1 and v['serve']['gpu_util'] > 0.6:
        raise ValueError('В sync на одном GPU vLLM и тренер делят память: поставьте gpu_util не больше 0.5.')
    if v['agents']['where'] not in ('vast', 'local'):
        raise ValueError('vast.agents.where must be vast or local')
    if not train['agents'] or not set(train['agents']) <= {'hermes', 'pi'}:
        raise ValueError('vast.train.agents: choose from hermes, pi')
    for key in ('steps', 'prompts_per_step', 'samples_per_group', 'lora_rank', 'lora_alpha', 'trial_slots', 'save_freq', 'max_turns'):
        if not isinstance(train[key], (int, float)) or train[key] < 1:
            raise ValueError(f'vast.train.{key} must be at least 1')
    if train['samples_per_group'] < 2:
        raise ValueError('vast.train.samples_per_group: GRPO needs at least 2 answers per group')
    updates = train.get('updates_per_step', 1)
    if not isinstance(updates, int) or updates < 1 or train['prompts_per_step'] % updates:
        raise ValueError('vast.train.updates_per_step: целое от 1, и «Задач на шаг» должно на него делиться')
    if v['serve']['context'] < train['max_prompt_length'] + train['max_response_length']:
        raise ValueError(f'Контекст vLLM ({v["serve"]["context"]}) меньше промпта + ответа '
                         f'({train["max_prompt_length"] + train["max_response_length"]}): поднимите «Контекст» в «Модель и vLLM».')
    if train['mode'] == 'async' and train.get('partial_rollout'):
        raise ValueError('partial_rollout для агентов Harbor не работает: verl отменяет идущие траектории при синхронизации весов, '
                         'а внешнего агента нельзя продолжить. Выключите «partial rollout».')
    if train.get('agent_schedule', 'mixed') not in ('mixed', 'by_step'):
        raise ValueError('vast.train.agent_schedule: mixed или by_step')
    if train['mode'] == 'sync' and train['trial_slots'] < train['prompts_per_step'] * train['samples_per_group']:
        raise ValueError(f'«Контейнеров одновременно» ({train["trial_slots"]}) меньше ответов на шаг '
                         f'({train["prompts_per_step"] * train["samples_per_group"]}): генерация пойдёт в две волны и шаг станет вдвое дольше. '
                         'Поднимите число контейнеров (и агент-хост побольше) или уменьшите задачи × ответы.')
    if train.get('loss_agg_mode', 'token-mean') not in ('token-mean', 'seq-mean-token-sum-norm', 'seq-mean-token-mean', 'seq-mean-token-sum'):
        raise ValueError('vast.train.loss_agg_mode: token-mean, seq-mean-token-sum-norm, seq-mean-token-mean или seq-mean-token-sum')
    if train.get('kl_coef', 0) < 0:
        raise ValueError('vast.train.kl_coef must be 0 or more')
    if not 0 < train['lr'] < 1:
        raise ValueError('vast.train.lr must be between 0 and 1')
    if v['limits']['max_hours'] <= 0 or v['limits']['gpu_budget_usd'] <= 0 or gpu['max_price_per_hour'] <= 0:
        raise ValueError('vast limits and the price cap must be positive')
    if v['limits']['dead_man_minutes'] < 10:
        raise ValueError('vast.limits.dead_man_minutes: at least 10, or a slow network destroys a healthy run')
    if v['hf']['keep'] < 1:
        raise ValueError('vast.hf.keep must be at least 1')
    return v


core.DEFAULT_CONFIG['vast'] = DEFAULT_VAST


# ---------------------------------------------------------------- Vast API

def vast(method, path, body=None, params=None):
    key = os.environ.get('VAST_API_KEY')
    if not key:
        raise RuntimeError('VAST_API_KEY is not set')
    url = API + path + ('?' + urllib.parse.urlencode(params) if params else '')
    for attempt in range(6):
        request = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None,
                                         {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors='replace')[:300]
            if exc.code in (429, 500, 502, 503, 504) and attempt < 5 and method == 'GET':
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f'Vast HTTP {exc.code} for {method} {path}: {detail}') from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < 5 and method in ('GET', 'DELETE'):
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f'Vast is unreachable: {type(exc).__name__}') from exc


def hourly(offer, disk_gb):
    """Upper bound of the hourly price with our disk size, with a 10% margin."""
    return max(offer['dph_total'], offer['dph_base'] + offer.get('storage_cost', 0) * disk_gb / 720) * 1.1


OFFER_KEYS = ('id', 'gpu_name', 'num_gpus', 'gpu_ram', 'dph_total', 'reliability2', 'geolocation', 'inet_down', 'inet_up',
              'cpu_cores_effective', 'cpu_ram', 'disk_space', 'cuda_max_good', 'machine_id')


def pick_offers(offers, gpu):
    """GPU offers that satisfy the config, cheapest first."""
    good = []
    for offer in offers:
        if offer.get('num_gpus') != gpu['num_gpus'] or offer.get('gpu_ram', 0) < gpu['min_gpu_ram_gb'] * 1000:
            continue
        if gpu['name_contains'] and gpu['name_contains'].lower() not in str(offer.get('gpu_name', '')).lower():
            continue
        if offer.get('cuda_max_good', 0) < gpu['min_cuda'] or offer.get('reliability2', 0) < gpu['min_reliability']:
            continue
        if min(offer.get('inet_up', 0), offer.get('inet_down', 0)) < gpu['min_inet_mbps'] or offer.get('disk_space', 0) < gpu['disk_gb']:
            continue
        if offer.get('cpu_ram', 0) < gpu.get('min_cpu_ram_gb', 0) * 1000:
            continue
        price = hourly(offer, gpu['disk_gb'])
        if price <= gpu['max_price_per_hour']:
            good.append({**{k: offer.get(k) for k in OFFER_KEYS}, 'hourly_bound': price})
    return sorted(good, key=lambda o: o['hourly_bound'])


def pick_host_offers(offers, agents, exclude=()):
    """VM offers for the agent host: many CPUs, RAM, Docker inside a VM. Cheapest first."""
    good = []
    for offer in offers:
        if not offer.get('vms_enabled') or offer.get('machine_id') in exclude:
            continue
        if offer.get('cpu_cores_effective', 0) < agents['min_cpu'] or offer.get('cpu_ram', 0) < agents['min_ram_gb'] * 1000:
            continue
        if offer.get('reliability2', 0) < agents['min_reliability'] or offer.get('disk_space', 0) < agents['disk_gb']:
            continue
        if min(offer.get('inet_up', 0), offer.get('inet_down', 0)) < agents['min_inet_mbps']:
            continue
        price = hourly(offer, agents['disk_gb'])
        if price <= agents['max_price_per_hour']:
            good.append({**{k: offer.get(k) for k in OFFER_KEYS}, 'hourly_bound': price})
    return sorted(good, key=lambda o: o['hourly_bound'])


def search_offers(gpu):
    query = {'num_gpus': {'eq': gpu['num_gpus']}, 'gpu_ram': {'gte': gpu['min_gpu_ram_gb'] * 1000}, 'rentable': {'eq': True},
             'rented': {'eq': False}, 'disk_space': {'gte': gpu['disk_gb']}, 'reliability2': {'gte': gpu['min_reliability']},
             'type': 'on-demand', 'limit': 200}
    return pick_offers(vast('GET', '/bundles/', params={'q': json.dumps(query)})['offers'], gpu)


def search_host_offers(agents, exclude=()):
    query = {'vms_enabled': {'eq': True}, 'verified': {'eq': True}, 'rentable': {'eq': True}, 'rented': {'eq': False},
             'cpu_cores_effective': {'gte': agents['min_cpu']}, 'cpu_ram': {'gte': agents['min_ram_gb'] * 1000},
             'disk_space': {'gte': agents['disk_gb']}, 'reliability2': {'gte': agents['min_reliability']}, 'type': 'on-demand',
             'order': [['dph_total', 'asc']], 'limit': 300}
    return pick_host_offers(vast('GET', '/bundles/', params={'q': json.dumps(query)})['offers'], agents, exclude)


def instances():
    keys = ('id', 'label', 'actual_status', 'status_msg', 'gpu_name', 'num_gpus', 'dph_total', 'ssh_host', 'ssh_port', 'start_date',
            'public_ipaddr', 'ports', 'machine_id')
    return [{k: x.get(k) for k in keys} for x in vast('GET', '/instances/')['instances']]


def endpoints(info):
    """SSH addresses of an instance: Vast's proxy, then the direct port (VMs are not always served by the proxy)."""
    found = [(info['ssh_host'], int(info['ssh_port']))] if info.get('ssh_host') and info.get('ssh_port') else []
    direct = (info.get('ports') or {}).get('22/tcp') or []
    if info.get('public_ipaddr') and direct:
        found.append((info['public_ipaddr'].strip(), int(direct[0]['HostPort'])))
    return found


def destroy(instance_id):
    result = vast('DELETE', f'/instances/{instance_id}/')
    if not result.get('success'):
        raise RuntimeError('Vast did not confirm the destroy')
    return True


# ---------------------------------------------------------------- what the machines run

def data_rows(rows, train, offset=0):
    """Prompts in the order verl will take them: (train rows, validation rows). offset: steps already trained before a restore."""
    tasks = sorted(r['task'] for r in rows if r['split'] == 'train')
    random.Random(train['seed']).shuffle(tasks)
    per_step, agents = int(train['prompts_per_step']), train['agents']

    def row(task, agent, split, group):
        return {'data_source': 'harbor', 'prompt': [{'role': 'user', 'content': 'Run the frozen Harbor task ' + task}], 'agent_name': 'harbor_native',
                'extra_info': {'task': task, 'agent': agent, 'split': split, 'policy': 'lab', 'group': group},
                'reward_model': {'style': 'rule', 'ground_truth': ''}}

    train_rows = []
    for step in range(offset + 1, int(train['steps']) + 1):
        for i in range(per_step):
            # by_step: every group of a step uses one agent, so in sync a Pi step does not wait for the slower Hermes
            # (Hermes trials took 100-125 s, Pi 41-85 s, on the test run).
            pick = (step - 1) if train.get('agent_schedule', 'mixed') == 'by_step' else (step + i)
            task, agent = tasks[((step - 1) * per_step + i) % len(tasks)], agents[pick % len(agents)]
            train_rows.append(row(task, agent, 'train', f'train-s{step:04d}-{agent}-{task}'))
    heldout = sorted(r['task'] for r in rows if r['split'] == 'heldout')[:int(train['eval_tasks'])]
    val_rows = [row(task, agents[i % len(agents)], 'dev', f'dev-{agents[i % len(agents)]}-{task}') for i, task in enumerate(heldout)]
    return train_rows, val_rows


GUARD = '''"""Destroys this rented machine when the lab runner goes silent or the deadline passes. Standard library only."""
import json, os, sys, time, urllib.request
from pathlib import Path
root = Path(sys.argv[1]); config = json.loads((root / 'guard.json').read_text())
def creds():
    env = dict(os.environ)
    for source in ('/proc/1/environ', '/etc/environment'):
        try:
            raw = Path(source).read_bytes()
        except OSError:
            continue
        for item in raw.replace(b'\\n', b'\\0').split(b'\\0'):
            key, _, value = item.decode(errors='replace').partition('=')
            env.setdefault(key.strip(), value.strip().strip('"'))
    return env.get('CONTAINER_ID'), env.get('CONTAINER_API_KEY')
def log(**event):
    with (root / 'guard.jsonl').open('a') as handle:
        handle.write(json.dumps({'time': time.time(), **event}) + '\\n')
ident, key = creds()
if not ident or not key:
    log(event='no_instance_key'); sys.exit(2)
started = time.time(); (root / 'guard.pid').write_text(str(os.getpid())); log(event='started', instance=ident)
while True:
    beat = root / 'heartbeat'
    last = beat.stat().st_mtime if beat.exists() else None
    now = time.time(); reason = None
    if (root / 'DESTROY').exists():
        reason = 'requested'
    elif now > config['deadline']:
        reason = 'deadline'
    elif last is None and now - started > config['first_grace']:
        reason = 'runner_never_connected'
    elif last is not None and now - last > config['heartbeat_timeout']:
        reason = 'runner_silent'
    if reason:
        break
    time.sleep(30)
log(event='destroying', reason=reason)
while True:
    try:
        request = urllib.request.Request(f'https://console.vast.ai/api/v0/instances/{ident}/', method='DELETE', headers={'Authorization': 'Bearer ' + key})
        log(event='destroyed', response=urllib.request.urlopen(request, timeout=40).read().decode()[:200]); break
    except Exception as exc:
        log(event='retry', error=str(exc)[:200]); time.sleep(15)
'''


def scripts(v, offset=0):
    """Scripts the GPU box runs, rendered from the config. offset: steps already trained before a restore."""
    model, serve, train, gpus = v['model'], v['serve'], v['train'], int(v['gpu']['num_gpus'])
    left = max(1, int(train['steps']) - offset)
    bootstrap = f'''#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-dev python3-venv git curl rsync openssh-client iproute2 libnuma1 libnuma-dev ninja-build
(curl -LsSf https://astral.sh/uv/0.11.16/install.sh | sh) || {{ python3 -m venv /opt/uvpip && /opt/uvpip/bin/pip install -q uv==0.11.16 && mkdir -p /root/.local/bin && ln -sf /opt/uvpip/bin/uv /root/.local/bin/uv; }}
export PATH=/root/.local/bin:$PATH UV_CACHE_DIR=/workspace/uvcache HF_HOME=/workspace/hf
[ -s /root/.hf_token ] && export HF_TOKEN="$(cat /root/.hf_token)"
cd {REMOTE}/vendor/verl
uv sync --frozen --all-packages --extra vllm --extra fsdp
.venv/bin/python - <<'PY'
import json, time, torch, transformers, vllm, peft
from huggingface_hub import snapshot_download
print(json.dumps({{'torch': torch.__version__, 'transformers': transformers.__version__, 'vllm': vllm.__version__, 'peft': peft.__version__,
                  'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}}), flush=True)
for attempt in range(8):
    try:
        snapshot_download({model['hf_id']!r}, revision={model['revision']!r}, local_dir='/workspace/model', max_workers=8)
        break
    except Exception as exc:   # rate limits and dropped connections: resume, it skips finished files
        if attempt == 7:
            raise
        print(f'download retry {{attempt + 1}}: {{type(exc).__name__}}: {{str(exc)[:200]}}', flush=True)
        time.sleep(30 * (attempt + 1))
PY
rm -rf /workspace/model/.cache
touch {REMOTE}/BOOTSTRAP_DONE
'''
    make_data = f'''import json
import pandas as pd
data = json.load(open('{NATIVE}/data.json'))
pd.DataFrame(data['train']).to_parquet('{NATIVE}/train.parquet', index=False)
pd.DataFrame(data['val']).to_parquet('{NATIVE}/val.parquet', index=False)
print({{'train_prompts': len(data['train']), 'val_prompts': len(data['val'])}})
'''
    to_agents = f'''#!/usr/bin/env bash
# The agent host reaches the policy proxy (port 9001 here) at $3:{AGENT_PORT}; this machine dials it, so traffic skips the laptop.
HOST=$1; PORT=$2; BIND=$3
echo $$ > {REMOTE}/to_agents.pid
while [ ! -f {REMOTE}/STOP_TUNNEL ]; do
  ssh -N -i /root/.ssh/to_agents -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \\
      -o ServerAliveCountMax=3 -p "$PORT" -R "$BIND:{AGENT_PORT}:127.0.0.1:9001" "root@$HOST"
  sleep 5
done
'''
    asynchronous = train['mode'] == 'async'
    offload = str(bool(train['cpu_offload']))
    per_step = int(train['prompts_per_step'])
    common = [
        'algorithm.adv_estimator=grpo', 'algorithm.use_kl_in_reward=False',
        f'data.train_files={NATIVE}/train.parquet', f'data.val_files={NATIVE}/val.parquet',
        f'data.max_prompt_length={int(train["max_prompt_length"])}', f'data.max_response_length={int(train["max_response_length"])}',
        'data.filter_overlong_prompts=True', 'data.truncation=error', 'data.shuffle=False',
        "'+data.apply_chat_template_kwargs.enable_thinking=False'",
        'actor_rollout_ref.model.path=/workspace/model',
        f'actor_rollout_ref.model.lora_rank={int(train["lora_rank"])}', f'actor_rollout_ref.model.lora_alpha={int(train["lora_alpha"])}',
        "'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,out_proj]'",
        "'actor_rollout_ref.model.exclude_modules=.*visual.*'",
        # Packed sequences (verl patches Qwen3.5 GatedDeltaNet with cu_seqlens). Padded, every sample costs prompt + response
        # length: with 64k responses an update of 16 trajectories took 144 s at 6% MFU (run 1111-164531, step 2).
        f'actor_rollout_ref.model.use_remove_padding={bool(train.get("remove_padding", True))}',
        'actor_rollout_ref.model.enable_gradient_checkpointing=True',
        'actor_rollout_ref.model.external_lib=native_compat',
        '+actor_rollout_ref.actor.checkpoint.save_lora_only=True', 'actor_rollout_ref.actor.strategy=fsdp2',
        f'actor_rollout_ref.actor.optim.lr={float(train["lr"])!r}',
        f'actor_rollout_ref.actor.ppo_mini_batch_size={per_step // int(train.get("updates_per_step", 1))}',
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1',
        'actor_rollout_ref.actor.ppo_epochs=1', 'actor_rollout_ref.actor.use_torch_compile=False',
        *packing_overrides(train),
        f'actor_rollout_ref.actor.clip_ratio_low={float(train["clip_ratio"])}',
        f'actor_rollout_ref.actor.clip_ratio_high={float(train.get("clip_ratio_high", train["clip_ratio"]))}',
        f'actor_rollout_ref.rollout.temperature={float(train.get("temperature", 1.0))}',
        f'actor_rollout_ref.actor.use_kl_loss={float(train["kl_coef"]) > 0}', f'actor_rollout_ref.actor.kl_loss_coef={float(train["kl_coef"])}',
        f'algorithm.norm_adv_by_std_in_grpo={bool(train.get("std_normalize", True))}',
        f'actor_rollout_ref.actor.loss_agg_mode={train.get("loss_agg_mode", "token-mean")}',
        f'algorithm.rollout_correction.bypass_mode={bool(train.get("bypass_old_logprob", False))}',
        'actor_rollout_ref.actor.kl_loss_type=low_var_kl',
        'actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16', 'actor_rollout_ref.actor.fsdp_config.fsdp_size=1',
        f'actor_rollout_ref.actor.fsdp_config.param_offload={offload}', f'actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}',
        'actor_rollout_ref.actor.fsdp_config.offload_policy=False', 'actor_rollout_ref.actor.entropy_from_logits_with_chunking=True',
        'actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16', 'actor_rollout_ref.ref.strategy=fsdp2',
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1', f'actor_rollout_ref.ref.fsdp_config.param_offload={offload}',
        'actor_rollout_ref.ref.use_torch_compile=False',
        'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.tensor_model_parallel_size=1',
        f'actor_rollout_ref.rollout.gpu_memory_utilization={float(serve["gpu_util"])}', f'actor_rollout_ref.rollout.n={int(train["samples_per_group"])}',
        f'actor_rollout_ref.rollout.max_model_len={int(serve["context"])}', f'actor_rollout_ref.rollout.max_num_batched_tokens={int(serve["max_batched_tokens"])}',
        f'actor_rollout_ref.rollout.max_num_seqs={int(serve["max_seqs"])}', f'actor_rollout_ref.rollout.enforce_eager={bool(serve["enforce_eager"])}',
        f'actor_rollout_ref.rollout.enable_prefix_caching={bool(serve["prefix_caching"])}',
        *mtp_overrides(serve),
        'actor_rollout_ref.rollout.load_format=safetensors', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1',
        'actor_rollout_ref.rollout.calculate_log_probs=True',
        'actor_rollout_ref.rollout.val_kwargs.do_sample=True', 'actor_rollout_ref.rollout.val_kwargs.temperature=1.0',
        f'actor_rollout_ref.rollout.val_kwargs.n={int(train["eval_samples"])}',
        'actor_rollout_ref.rollout.agent.num_workers=1', 'actor_rollout_ref.rollout.agent.default_agent_loop=harbor_native',
        f'actor_rollout_ref.rollout.agent.agent_loop_config_path={NATIVE}/agent.yaml',
        'reward.num_workers=1', f'reward.custom_reward_function.path={NATIVE}/reward_fn.py', 'reward.custom_reward_function.name=compute_score',
        'trainer.nnodes=1', 'trainer.total_epochs=1', 'trainer.val_before_train=True',
        f'trainer.test_freq={int(train["test_freq"])}', f'trainer.save_freq={int(train["save_freq"])}', 'trainer.max_actor_ckpt_to_keep=2',
        "'trainer.logger=[console,file]'", 'trainer.project_name=antislop_lab', 'trainer.experiment_name=lab',
        f'trainer.default_local_dir={NATIVE}/output/checkpoints', 'ray_kwargs.ray_init.runtime_env.py_executable=null',
    ]
    if asynchronous:
        entry = 'verl.experimental.fully_async_policy.fully_async_main'
        # Disaggregated weight sync sends full tensors only: verl's async path drops the PEFT config, so an unmerged
        # LoRA model arrives with '.base_layer.' names vLLM rejects (seen on run 1111-160639). Merge before each sync;
        # vLLM then runs without LoRA. Checkpoints stay LoRA-only.
        common += ['actor_rollout_ref.model.lora.merge=True']
        # Without this the async rollouter replaces every agent with single_turn_agent (detach_utils.py), so training
        # samples skip Harbor and reach the no-fallback reward function (seen on run 1111-162628; validation was fine).
        common += ['actor_rollout_ref.rollout.multi_turn.enable=True']
        common += ['actor_rollout_ref.hybrid_engine=False', 'actor_rollout_ref.rollout.mode=async', 'data.train_batch_size=0', 'data.gen_batch_size=1',
                   f'trainer.n_gpus_per_node={max(1, gpus - 1)}', 'rollout.nnodes=1', 'rollout.n_gpus_per_node=1',
                   f'rollout.n={int(train["samples_per_group"])}', f'rollout.total_rollout_steps={rollout_prompts(left, per_step)}', f'trainer.total_training_steps={left}',
                   f'async_training.staleness_threshold={float(train["staleness_threshold"])}',
                   # The trainer takes require_batches mini-batches per step: with updates_per_step mini-batches it trains on
                   # the whole step's prompts, not on one mini-batch (2 of 4 prompts on run 1111-164531).
                   f'async_training.trigger_parameter_sync_step={int(train["sync_every"])}',
                   f'async_training.require_batches={int(train.get("updates_per_step", 1))}',
                   f'async_training.partial_rollout={bool(train["partial_rollout"])}']
    else:
        entry = 'verl.trainer.main_ppo'
        common += [f'data.train_batch_size={per_step}', f'data.val_batch_size={max(1, min(4, int(train["eval_tasks"])))}',
                   f'trainer.n_gpus_per_node={gpus}', f'trainer.total_training_steps={left}', 'trainer.use_v1=False',
                   'actor_rollout_ref.rollout.free_cache_engine=True', 'actor_rollout_ref.rollout.layered_summon=True']
    run = f'''#!/usr/bin/env bash
set -euo pipefail
cd {REMOTE}/vendor/verl
export PYTHONPATH={NATIVE}:$PWD ANTISLOP_NATIVE_ROOT={NATIVE}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_USE_V1=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME=/workspace/hf VERL_FILE_LOGGER_PATH={NATIVE}/output/metrics.jsonl
exec {VENV}/python -m {entry} \\
 ''' + ' \\\n '.join(common) + (' \\\n ' + train['extra_overrides'].strip() if train['extra_overrides'].strip() else '') + ' \\\n "$@"\n'
    return {'bootstrap.sh': bootstrap, 'native/make_data.py': make_data, 'native/run_verl.sh': run, 'to_agents.sh': to_agents, 'guard.py': GUARD}


def agent_setup_script():
    """One-time setup of the agent host: Docker plugins, sshd for many sessions and the GPU's tunnel, Python, images."""
    return f'''#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
command -v rsync >/dev/null && command -v python3 >/dev/null || {{ apt-get update -qq && apt-get install -y -qq rsync git curl python3 python3-venv iproute2; }}
docker info >/dev/null
mkdir -p /etc/ssh/sshd_config.d
# Many trials share one multiplexed connection; the GPU's tunnel binds on the Docker bridge, not on the public address.
printf 'ClientAliveInterval 15\\nClientAliveCountMax 3\\nMaxSessions 256\\nMaxStartups 200:30:400\\nGatewayPorts clientspecified\\n' > /etc/ssh/sshd_config.d/10-lab.conf
grep -q '^Include /etc/ssh/sshd_config.d' /etc/ssh/sshd_config || sed -i '1i Include /etc/ssh/sshd_config.d/*.conf' /etc/ssh/sshd_config
sshd -t && {{ systemctl reload ssh 2>/dev/null || systemctl reload sshd 2>/dev/null || true; }}
P=/usr/local/lib/docker/cli-plugins; mkdir -p $P
for name in compose buildx; do
  docker $name version >/dev/null 2>&1 || {{ [ -x {AG}/bin/docker-$name ] && install -m 755 {AG}/bin/docker-$name $P/docker-$name; }}
  docker $name version >/dev/null
done
# Parallel trials each create a compose network; the default pools run out at about 30.
if ! grep -q default-address-pools /etc/docker/daemon.json 2>/dev/null; then
  echo '{{"default-address-pools":[{{"base":"10.210.0.0/16","size":24}}]}}' > /etc/docker/daemon.json
  systemctl restart docker; sleep 3; docker info >/dev/null
fi
[ -x /root/.local/bin/uv ] || (curl -LsSf https://astral.sh/uv/0.11.16/install.sh | sh) || {{ python3 -m venv /opt/uvpip && /opt/uvpip/bin/pip install -q uv==0.11.16 && mkdir -p /root/.local/bin && ln -sf /opt/uvpip/bin/uv /root/.local/bin/uv; }}
export PATH=/root/.local/bin:$PATH
cd {AG}
grep -vE '^-e |@ file:' requirements.txt > /tmp/requirements.txt
[ -x venv/bin/python ] || uv venv --python 3.12 venv
uv pip install --python venv/bin/python -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
venv/bin/python -c "import harbor"
docker image inspect antislop-pilot-hermes:local >/dev/null 2>&1 || docker build -q -t antislop-pilot-hermes:local -f images/hermes.Dockerfile images
docker image inspect antislop-pilot-pi:local >/dev/null 2>&1 || docker build -q -t antislop-pilot-pi:local -f images/pi.Dockerfile images
mkdir -p docker-config && echo '{{}}' > docker-config/config.json
mkdir -p rollouts rollouts_failed
touch {AG}/SETUP_DONE
'''


HOST_GATEWAY = '''# Docker on Linux does not define host.docker.internal; agents reach the policy tunnel through it.
services:
  main:
    extra_hosts:
      - "host.docker.internal:host-gateway"
'''


def model_card(cfg, state, step, merged):
    """README for a published model: what it is, how it was trained, what the judges measured."""
    v, train = cfg['vast'], cfg['vast']['train']
    groups = [g for g in state.get('groups', []) if g['source'] == 'eval']
    lines = ['---', f'base_model: {v["model"]["hf_id"]}'] + ([] if merged else ['library_name: peft']) + [
        'tags:', '- antislop', '- grpo', '- verl', '- lora', '- agents', '---', '',
        f'# {v["model"]["hf_id"].split("/")[-1]}: antislop GRPO, step {step}', '',
        ('Merged weights' if merged else 'A LoRA adapter') + f' for `{v["model"]["hf_id"]}` (revision `{v["model"]["revision"][:12]}`), '
        'trained to write text that reads as written by a person, inside the Hermes and Pi agent harnesses.', '',
        '## Training', '',
        f'- Method: GRPO with verl ({"fully async" if train["mode"] == "async" else "synchronous"}), LoRA rank {train["lora_rank"]}, alpha {train["lora_alpha"]}, LR {train["lr"]}.',
        f'- {step} optimizer steps, {train["prompts_per_step"]} writing tasks per step, {train["samples_per_group"]} agent runs per task.',
        f'- Rewards from LLM judges: style by `{cfg["judge"]["model"]}`'
        + (f', task completion by `{cfg["task_judge"]["model"]}`' if cfg['task_judge']['enabled'] else '') + '.',
        f'- Reward formula: `{cfg["reward"]["formula"]}`.', '']
    if groups:
        lines += ['## Held-out tasks', '', f'Mean reward over {len(groups)} held-out groups: {sum(g["reward"] for g in groups[-20:]) / len(groups[-20:]):.3f} '
                  '(last evaluation). These are judge scores, not a detector result.', '']
    lines += ['## Limits', '', 'Experimental. Scores come from LLM judges, which can be gamed; check outputs yourself.', '']
    return '\n'.join(lines)


def split_log(buffer, chunk):
    """Complete lines of buffer + chunk (bytes), decoded, and the unfinished tail to keep."""
    lines = (buffer + chunk).split(b'\n')
    return [line.decode(errors='replace') for line in lines[:-1] if line.strip()], lines[-1]


def pick_metrics(data):
    return {k: v for k, v in data.items() if isinstance(v, (int, float)) and any(part in k for part in METRIC_KEYS)}


class Paused(Exception):
    """Stop requested; not an error."""


# ---------------------------------------------------------------- rollouts and rewards

class Coordinator:
    """Answers the GPU's agent loop: runs one agent trial and returns its reward.

    A reward exists only once the whole group of answers to a task is in, because the judges score a group.
    A trial that broke for infrastructure reasons gets the group mean (zero advantage) and a flag that masks
    its tokens, so one bad container does not crash a training step.
    """

    def __init__(self, runner):
        self.runner, self.key = runner, (runner.dir / 'serve.key').read_text().strip()
        self.groups, self.lock = {}, threading.Lock()
        self.slots = threading.Semaphore(int(runner.v['train']['trial_slots']))
        self.recent = collections.deque(maxlen=32)
        self.circuit = False
        # Judges start on each answer as soon as its trial ends; the group waits only for the last one.
        self.judges = concurrent.futures.ThreadPoolExecutor(max_workers=32, thread_name_prefix='prefetch')
        sys.path.insert(0, str(expand(runner.v['local']['native'])))
        import cap_grade
        self.cap = cap_grade

    def fatal_reason(self):
        if self.runner.stop_requested():
            return 'Stop requested'
        if self.circuit:
            return 'Circuit open: most recent agent trials failed'
        return None

    def note(self, failed, reason=''):
        with self.lock:
            self.recent.append(bool(failed))
            if len(self.recent) == self.recent.maxlen and sum(self.recent) * 2 > len(self.recent):
                self.circuit = True
        if failed:
            self.runner.emit('warn', reason[:1500], source='coordinator')

    def collector_args(self, body, out, remote):
        r, train, local = self.runner, self.runner.v['train'], self.runner.v['local']
        agent, uid = body['agent'], body['uid']
        if remote:
            head, protocol, tasks = [f'{AG}/venv/bin/python', f'{AG}/collector/hermes_collect.py'], f'{AG}/protocol.json', f'{AG}/tasks/{agent}'
            api = f'http://host.docker.internal:{AGENT_PORT}/session/{uid}/v1'
        else:
            head, protocol, tasks = [str(expand(local['python'])), str(expand(local['collector']))], str(r.dir / 'protocol.json'), str(expand(local['task_root']) / agent)
            api = f'http://host.docker.internal:{int(local["tunnel_port"])}/session/{uid}/v1'
        return head + ['--agent', agent, '--agent-version', local['agent_versions'][agent], '--preinstalled', '--terminal-only',
                       '--protocol', protocol, '--task-root', tasks, '--tasks', body['task'], '--split', body['split'], '--api-base', api,
                       '--policy-id', body['policy'], '--model', 'antislop-scaffold', '--samples', '1', '--max-turns', str(train['max_turns']),
                       '--timeout', str(train['trial_timeout_sec']), '--setup-timeout', '300', '--out', str(out)]

    def execute(self, body, out, attempt):
        """One collector run. Locally it is a subprocess; on the agent host it runs detached and is polled, then copied back."""
        r, timeout = self.runner, self.runner.v['train']['trial_timeout_sec'] + 900
        if r.v['agents']['where'] == 'local':
            out.mkdir(parents=True, exist_ok=True)
            with (out / 'collector.log').open('w') as log:
                return subprocess.run(self.collector_args(body, out, False), env=r.local_env(), stdout=log, stderr=subprocess.STDOUT,
                                      timeout=timeout).returncode
        rout = f'{AG}/rollouts/{body["group"]}/{body["uid"]}'
        inner = f'set -a; . {AG}/run.env; set +a; {shlex.join(self.collector_args(body, rout, True))} > {rout}/collector.log 2>&1; echo $? > {rout}/exit'
        r.ssh(f'mkdir -p {rout} && rm -f {rout}/exit && (setsid nohup bash -c {shlex.quote(inner)} > /dev/null 2>&1 < /dev/null &)', who='agents', timeout=60)
        started, failures, code = time.time(), 0, ''
        while not code:
            time.sleep(10)
            if time.time() - started > timeout:
                raise ValueError(f'trial did not finish in {timeout} s on the agent host')
            try:
                code = r.ssh(f'cat {rout}/exit 2>/dev/null || true', who='agents', timeout=60).strip()
                failures = 0
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                failures += 1
                if failures > 30:
                    raise ValueError(f'agent host unreachable for 5 minutes: {exc}') from exc
        out.mkdir(parents=True, exist_ok=True)
        r.rsync(str(out) + '/', rout + '/', upload=False, who='agents')
        return int(code)

    def discard(self, body, out, attempt):
        """A trial with no verifier result is moved aside and repeated; the collector would otherwise resume it."""
        r = self.runner
        (r.dir / 'rollouts_failed').mkdir(exist_ok=True)
        out.rename(r.dir / 'rollouts_failed' / f'{body["uid"]}-{attempt}')
        if r.v['agents']['where'] == 'vast':
            r.ssh(f'mv {AG}/rollouts/{body["group"]}/{body["uid"]} {AG}/rollouts_failed/{body["uid"]}-{attempt} 2>/dev/null || true', who='agents', timeout=60)

    def run_trial(self, body):
        r, agent = self.runner, body['agent']
        out = r.dir / 'rollouts' / body['group'] / body['uid']
        for attempt, backoff in enumerate((30, 60, 120, None)):
            with self.slots:
                code = self.execute(body, out, attempt)
            rows = [json.loads(x) for x in (out / 'outcomes.jsonl').read_text().splitlines()] if (out / 'outcomes.jsonl').exists() else []
            if code or len(rows) != 1:
                log = (out / 'collector.log').read_text(errors='replace')[-600:] if (out / 'collector.log').exists() else ''
                raise ValueError(f'collector failed (exit {code}): ' + log)
            row = rows[0]
            if row['grade_outcome'].get('valid'):
                return row, out
            raw = json.loads(next(out.glob('*-result.json')).read_text())
            session = next(iter(out.glob('trials/*/agent/hermes-session.jsonl')), None)
            text = session.read_text() if session else ''
            agent_log = next(iter(out.glob('trials/*/agent/hermes.txt')), None)
            startup = agent == 'hermes' and agent_log and self.cap.hermes_startup_failed(raw, text, agent_log.read_text(errors='replace'))
            if self.cap.docker_command_timed_out(raw) or startup:
                # No verifier ran and no sample was produced: repeat the trial instead of grading it.
                self.discard(body, out, attempt)
                if backoff is None:
                    raise ValueError('Docker or agent startup failed on every attempt')
                time.sleep(backoff)
                continue
            grade = None
            if agent == 'hermes' and text.strip():
                grade = self.cap.confirmed_cap_grade(raw, json.loads(text.splitlines()[0]))
            grade = grade or self.cap.confirmed_timeout_grade(raw) or self.cap.confirmed_rate_limit_label_grade(raw)
            if grade is None:
                raise ValueError('invalid trial grade: ' + str(row['grade_outcome'].get('reason')))
            row['grade_outcome'] = grade
            return row, out

    @staticmethod
    def answer(row, out):
        files = list(out.glob(f'**/{row["trial_id"]}/verifier/answer.txt'))
        return files[0].read_text(errors='replace') if len(files) == 1 and row['grade_outcome'].get('reward') == 1 else ''

    def prefetch(self, body, row, out):
        task = self.runner.by_task[body['task']]
        return self.judges.submit(core.prefetch, self.runner.cfg, task['prompt'], self.answer(row, out), task.get('reference'))

    def score(self, body, group, valid, failed, early=()):
        r = self.runner
        task = r.by_task[body['task']]
        for future in early:      # errors are not fatal here: score_group repeats the call and raises properly
            try:
                future.result(timeout=900)
            except Exception:  # noqa: BLE001
                pass
        answers = [self.answer(row, out) for row, out in valid.values()]
        result = core.score_group(r.cfg, task['prompt'], answers, task['target_words'], task['lang'], task['allow_markdown'],
                                  reference=task.get('reference'), length_tolerance=task.get('length_tolerance'))
        step = re.match(r'train-s(\d+)-', group)
        record = {'ts': time.time(), 'step': int(step.group(1)) if step else None, 'source': 'train' if body['split'] == 'train' else 'eval',
                  'task': task['prompt'], 'agent': body['agent'], 'answers': answers, 'items': result['items'], 'stats': result['stats'],
                  **({'reference': task['reference']} if task.get('reference') else {}),
                  'cost_usd': result['cost_usd']}
        with self.lock, (core.STATE / 'rewards.jsonl').open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        stats = result['stats']
        r.emit('metric', f'{record["source"]} {"шаг " + str(record["step"]) + " " if step else ""}{body["agent"]} {body["task"]}: награда {stats["mean"]:.3f}, '
                         f'разброс {stats["std"]:.3f}{" — ничья" if stats["tied"] else ""}, упало попыток {len(failed)}, судья ${result["cost_usd"]:.2f}',
               source='coordinator')
        r.count_group(record)
        out = {uid: {'reward': reward, 'metrics': {'infra_failed': 0}} for uid, reward in zip(valid, result['rewards'])}
        out.update({uid: {'reward': stats['mean'], 'metrics': {'infra_failed': 1}, 'infra_failed': True} for uid in failed})
        return out

    def process(self, body):
        reason = self.fatal_reason()
        if reason:
            raise RuntimeError(reason)
        group, uid, size = body['group'], body['uid'], int(body['size'])
        with self.lock:
            state = self.groups.setdefault(group, {'rows': {}, 'future': concurrent.futures.Future(), 'early': []})
        try:
            row, out = self.run_trial(body)
            self.note(False)
        except Exception as exc:  # noqa: BLE001 - one broken trial must not end the training step
            self.note(True, f'Попытка {group}/{uid} не удалась: {type(exc).__name__}: {exc}')
            row, out = None, None
        early = None
        if row is not None:
            try:
                early = self.prefetch(body, row, out)
            except Exception:  # noqa: BLE001 - only an optimisation
                early = None
        with self.lock:
            state['rows'][uid] = (row, out)
            if early is not None:
                state['early'].append(early)
            ready = len(state['rows']) == size
        if ready:
            try:
                valid = {u: pair for u, pair in state['rows'].items() if pair[0] is not None}
                failed = [u for u, pair in state['rows'].items() if pair[0] is None]
                if len(valid) >= 2:
                    result = self.score(body, group, valid, failed, state['early'])
                else:
                    self.runner.emit('warn', f'Группа {group}: годных попыток {len(valid)} из {size}, группа пропущена.', source='coordinator')
                    result = {u: {'reward': 0.0, 'metrics': {'infra_failed': 1}, 'infra_failed': True} for u in state['rows']}
                state['future'].set_result(result)
            except Exception as exc:  # noqa: BLE001 - delivered to every waiting request of the group
                state['future'].set_exception(exc)
        return state['future'].result(timeout=2400)[uid]

    def serve(self):
        coordinator = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != '/run' or self.headers.get('Authorization') != 'Bearer ' + coordinator.key:
                    return self.send_error(401)
                try:
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    if not all(isinstance(body.get(k), str) and re.fullmatch(r'[A-Za-z0-9_-]+', body[k]) for k in ('uid', 'group', 'task', 'agent', 'split', 'policy')):
                        raise ValueError('Invalid identifiers')
                    payload, code = json.dumps(coordinator.process(body)).encode(), 200
                except Exception as exc:  # noqa: BLE001 - the GPU side gets the reason and fails loudly
                    text = f'{type(exc).__name__}: {exc}'
                    payload, code = json.dumps({'error': text}).encode(), 500
                    if any(mark in text for mark in FATAL):
                        coordinator.runner.fatal = text
                    coordinator.runner.emit('error' if any(mark in text for mark in FATAL) else 'warn', 'Награда не выдана: ' + text[:1500], source='coordinator')
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', int(self.runner.v['local']['coordinator_port'])), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server


# ---------------------------------------------------------------- the runner

class Runner:
    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.state = json.loads((self.dir / 'state.json').read_text())
        self.lock, self.group_lock = threading.Lock(), threading.Lock()
        self.reload()
        self.tunnel, self.coordinator, self.server, self.fatal = None, None, None, None
        self.background = threading.Event()   # set: heartbeat and HF threads stop
        self.threads_started = False

    # -- bookkeeping

    def reload(self):
        self.cfg = core.validate_config(core.merged(core.DEFAULT_CONFIG, json.loads((self.dir / 'config.json').read_text())))
        self.v = validate_vast(self.cfg['vast'])
        self.rows = [json.loads(line) for line in expand(self.v['local']['tasks']).read_text().splitlines()]
        self.by_task = {r['task']: r for r in self.rows}

    def save(self, **updates):
        with self.lock:
            self.state.update(updates, updated=time.time())
            core.atomic_json(self.dir / 'state.json', self.state)

    def emit(self, level, message, source='runner', **extra):
        event = {'ts': time.time(), 'level': level, 'stage': self.state.get('stage'), 'source': source, 'msg': message, **extra}
        with self.lock, (self.dir / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + '\n')

    def stage(self, name, **extra):
        self.save(stage=name, **extra)
        self.emit('stage', name)

    def stop_requested(self):
        return (self.dir / 'STOP').exists()

    def count_group(self, record):
        """Running totals of judged groups, for the header and the curves."""
        items = [i for i in record['items'] if not i['empty']]
        with self.group_lock:
            self.save(groups=(self.state.get('groups', []) + [self.group_row(record, items)])[-4000:])

    @staticmethod
    def group_row(record, items):
        return {
            'ts': record['ts'], 'step': record['step'], 'source': record['source'], 'reward': record['stats']['mean'], 'std': record['stats']['std'],
            'tied': record['stats']['tied'], 'empty': len(record['items']) - len(items), 'usd': record['cost_usd'],
            'words': sum(i['words'] for i in items) / len(items) if items else 0,
            'scores': {k: sum(i['scores'][k] for i in items) / len(items) for k in (items[0]['scores'] if items else {})}}

    def offset(self):
        return int(self.state.get('step_offset', 0))

    def machines(self):
        return [(who, self.state[key]) for who, key in (('gpu', 'instance'), ('agents', 'agents')) if self.state.get(key)]

    def spent(self):
        return sum((time.time() - inst['created']) / 3600 * inst.get('hourly_bound', 0) for _, inst in self.machines())

    def over_limits(self):
        limits, inst = self.v['limits'], self.state.get('instance') or {}
        hours = (time.time() - inst.get('created', time.time())) / 3600
        if hours > limits['max_hours'] or self.spent() > limits['gpu_budget_usd']:
            return f'Лимит аренды достигнут: {hours:.1f} ч, около ${self.spent():.0f}.'
        return None

    # -- ssh to either machine

    def ssh_base(self, who='gpu'):
        inst = self.state['instance' if who == 'gpu' else 'agents']
        base = ['ssh', '-i', str(expand(self.v['local']['ssh_key'])), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
                '-o', 'UserKnownHostsFile=' + str(self.dir / 'known_hosts'), '-o', 'ServerAliveInterval=20', '-o', 'ConnectTimeout=20',
                '-p', str(inst['ssh_port'])]
        # Many threads poll both machines: one multiplexed connection each instead of a connection storm.
        base[1:1] = ['-o', 'ControlMaster=auto', '-o', 'ControlPath=~/.ssh/lab-%C', '-o', 'ControlPersist=600']
        return base

    def tunnel_base(self):
        """A long-lived forward must own its connection: when it dies, it must not take the shared one with it."""
        base = self.ssh_base()
        base[1:7] = ['-o', 'ControlMaster=no', '-o', 'ControlPath=none']
        return base

    def ssh(self, command, timeout=120, raw=False, who='gpu', stdin=None):
        host = self.state['instance' if who == 'gpu' else 'agents']['ssh_host']
        result = subprocess.run(self.ssh_base(who) + ['root@' + host, command], capture_output=True, timeout=timeout, input=stdin)
        if result.returncode:
            raise RuntimeError(f'Команда на {"GPU" if who == "gpu" else "агент-хосте"} завершилась с кодом {result.returncode}: '
                               + (result.stderr or result.stdout).decode(errors='replace')[-800:].strip())
        return result.stdout if raw else result.stdout.decode(errors='replace')

    def rsync(self, local, remote, upload=True, exclude=(), who='gpu'):
        spec = 'root@' + self.state['instance' if who == 'gpu' else 'agents']['ssh_host'] + ':' + remote
        command = ['rsync', '-az', '--partial', '--timeout=600', '-e', shlex.join(self.ssh_base(who))]
        for pattern in exclude:
            command += ['--exclude', pattern]
        command += [str(local), spec] if upload else [spec, str(local)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600, errors='replace')
        if result.returncode:
            raise RuntimeError('rsync не прошёл: ' + result.stderr[-800:].strip())

    def job(self, name, command, timeout_min, tick=None, who='gpu', quiet=False, guarded=True):
        """Run a command on a machine under nohup and stream its log here. Survives dropped SSH sessions."""
        root = REMOTE if who == 'gpu' else AG
        log, done = f'{root}/output/{name}.log', f'{root}/output/{name}.exit'
        script, pidfile = f'{root}/jobs/{name}.sh', f'{root}/jobs/{name}.pid'
        # Matching by name would also match the shell that runs it (its command line contains the path) and kill our own SSH
        # session; the job writes its pid instead and is stopped as a process group.
        stop = f'[ -s {pidfile} ] && kill -- -$(cat {pidfile}) 2>/dev/null; true'
        body = f'echo $$ > {pidfile}\n{command} > {log} 2>&1; echo $? > {done}\n'
        self.ssh(f'mkdir -p {root}/jobs {root}/output; {stop}; rm -f {done}; '
                 f'printf %s {shlex.quote(body)} > {script}; (setsid nohup bash {script} > /dev/null 2>&1 < /dev/null &) ; sleep 1', who=who)
        offset, buffer, failures, started, tail = 0, b'', 0, time.time(), []
        while True:
            reason = ('остановка по запросу' if self.stop_requested() else self.fatal or self.over_limits()) if guarded else None
            if reason or time.time() - started > timeout_min * 60:
                self.ssh(stop, timeout=60, who=who)
                if self.stop_requested() or (reason and reason == self.over_limits()):
                    self.emit('warn', f'{name} прерван: {reason}')
                    (self.dir / 'STOP').touch()
                    raise Paused()
                raise RuntimeError(reason or f'{name}: не завершилось за {timeout_min} мин.\n' + '\n'.join(tail[-40:]))
            try:
                out = self.ssh(f'tail -c +{offset + 1} {log} 2>/dev/null | head -c 400000; printf "\\n__EXIT__%s" "$(cat {done} 2>/dev/null)"',
                               timeout=60, raw=True, who=who)
                failures = 0
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                failures += 1
                if failures == 1:
                    self.emit('warn', f'SSH пропал во время «{name}», жду восстановления: {str(exc)[:160]}')
                if failures > 60:
                    raise RuntimeError(f'{name}: связь с машиной не восстановилась за 10 минут. {exc}') from exc
                time.sleep(10)
                continue
            chunk, _, code = out.rpartition(b'\n__EXIT__')
            offset += len(chunk)
            lines, buffer = split_log(buffer, chunk)
            for line in lines:
                tail.append(line)
                del tail[:-80]
                if not quiet:
                    self.emit('log', line[:2000], source=name)
            if tick:
                tick()
            code = code.decode(errors='replace').strip()
            if code:
                if code != '0':
                    raise RuntimeError(f'{name}: завершилось с кодом {code}.\n' + '\n'.join(tail[-40:]))
                return tail
            time.sleep(5)

    # -- stages

    def local_env(self):
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'SSL_CERT_FILE')}
        env['PATH'] = '/opt/homebrew/bin:/usr/local/bin:' + env.get('PATH', '')
        env['DOCKER_HOST'] = self.v['local']['docker_host'].replace('~', str(Path.home()))
        env['ANTISLOP_SERVE_KEY'] = (self.dir / 'serve.key').read_text().strip()
        return env

    def preflight(self):
        self.stage('preflight')
        local, train, problems = self.v['local'], self.v['train'], []
        for name in ('VAST_API_KEY', self.cfg['judge']['api_key_env'], self.cfg['task_judge']['api_key_env']):
            if name and not os.environ.get(name):
                problems.append(f'Не задана переменная {name}')
        paths = [('SSH-ключ', local['ssh_key']), ('сборщик траекторий', local['collector']), ('agent loop для verl', Path(local['native']) / 'native_agent.py'),
                 ('verl', local['verl']), ('задачи', local['tasks'])]
        if self.v['agents']['where'] == 'vast':
            paths += [('зависимости агент-хоста', local['requirements']), ('Dockerfile агентов', local['agent_images']), ('плагины Docker', local['docker_plugins'])]
        else:
            paths += [('Python с harbor', local['python'])]
        for label, path in paths:
            if not expand(path).exists():
                problems.append(f'Не найден {label}: {expand(path)}')
        for agent in train['agents']:
            if not (expand(local['task_root']) / agent).is_dir():
                problems.append(f'Нет каталога задач для агента {agent}: {expand(local["task_root"]) / agent}')
        if self.v['agents']['where'] == 'local':
            try:
                images = subprocess.run(['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'], env=self.local_env(), capture_output=True,
                                        text=True, timeout=30)
                if images.returncode:
                    problems.append('Docker не отвечает: ' + images.stderr.strip()[-200:] + ' ' + explain('docker.sock'))
                else:
                    for agent in train['agents']:
                        if f'antislop-pilot-{agent}:local' not in images.stdout:
                            problems.append(f'Нет Docker-образа antislop-pilot-{agent}:local')
            except (OSError, subprocess.TimeoutExpired) as exc:
                problems.append(f'Docker не запускается: {exc}')
        hf = self.v['hf']
        if hf['backup'] and not os.environ.get(hf['token_env']):
            self.emit('warn', f'Нет токена Hugging Face ({hf["token_env"]}): бэкапов на HF не будет, только скачивание адаптеров сюда.')
        if problems:
            raise RuntimeError('Проверка перед запуском не прошла:\n- ' + '\n- '.join(problems))
        calls = train['steps'] * train['prompts_per_step'] * train['samples_per_group']
        self.emit('info', f'Проверка пройдена. Режим {train["mode"]}, агенты {"на отдельной VM" if self.v["agents"]["where"] == "vast" else "здесь"}: '
                          f'{train["steps"]} шагов × {train["prompts_per_step"]} задач × {train["samples_per_group"]} ответов = {calls} оценок судьи. '
                          f'Бюджет судьи ${self.cfg["budget_usd"]:.0f}, потрачено ${core.ledger()["spent_usd"]:.2f}.')
        log = core.STATE / 'rewards.jsonl'
        recent = [json.loads(line) for line in log.read_text().splitlines()[-40:]] if log.exists() else []
        paid = [(r['cost_usd'], len(r['answers'])) for r in recent if r.get('cost_usd')]
        if paid:
            estimate = calls * sum(c for c, _ in paid) / sum(n for _, n in paid)
            left = self.cfg['budget_usd'] - core.ledger()['spent_usd']
            if estimate > left:
                self.emit('warn', f'Судья на весь прогон обойдётся примерно в ${estimate:.0f} (по последним оценкам), а в бюджете осталось ${left:.0f}. '
                                  'Когда бюджет кончится, прогон встанет на паузу; поднимите «Бюджет, $» на вкладке «Судья» заранее.')

    def wait_ssh(self, key, label, minutes=20):
        """Wait until the machine answers on one of its SSH addresses; remember the one that works."""
        inst, deadline, last = self.state[key], time.time() + minutes * 60, None
        while True:
            own = next((x for x in instances() if x['id'] == inst['id']), None)
            if own is None:
                raise RuntimeError(f'Vast удалил {label} до запуска (оффер стал недоступен). Нажмите «Продолжить», чтобы взять другой.')
            status = f'{own.get("actual_status")}: {own.get("status_msg") or ""}'.strip()
            if status != last:
                self.emit('info', f'{label}: {status}')
                last = status
            if own.get('actual_status') == 'running':
                for host, port in endpoints(own):
                    inst.update(ssh_host=host, ssh_port=port, endpoints=endpoints(own))
                    self.save(**{key: inst})
                    try:
                        if self.ssh('echo ready', timeout=30, who='gpu' if key == 'instance' else 'agents').strip().endswith('ready'):
                            if key == 'agents':
                                self.settle_host_key(label)
                            return
                    except (RuntimeError, subprocess.TimeoutExpired) as exc:
                        if 'publickey' in str(exc) and host == endpoints(own)[-1][0]:
                            raise
            if time.time() > deadline:
                raise RuntimeError(f'{label} не стал доступен по SSH за {minutes} минут. Последний статус: {status}')
            time.sleep(10)

    def settle_host_key(self, label, minutes=10):
        """The agent VM (vastai/kvm image) answers SSH before cloud-init regenerates its host keys: the key learned
        first stops matching ~2 minutes later and every new connection fails with REMOTE HOST IDENTIFICATION HAS
        CHANGED (run 1111-200356, after bootstrap had passed on the multiplexed connection). Until cloud-init is done
        a changed key is expected and re-learned; after this the key stays pinned."""
        inst = self.state['agents']
        entry = f'[{inst["ssh_host"]}]:{inst["ssh_port"]}'
        deadline = time.time() + minutes * 60
        while True:
            try:
                out = self.ssh('if command -v cloud-init >/dev/null; then cloud-init status --wait >/dev/null 2>&1; fi; echo settled',
                               timeout=minutes * 60, who='agents')
                if out.strip().endswith('settled'):
                    break
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                if 'IDENTIFICATION HAS CHANGED' not in str(exc) or time.time() > deadline:
                    raise
                self.emit('info', f'{label}: VM сменила ssh-ключ при первой загрузке, запоминаю новый.')
                subprocess.run(self.ssh_base('agents') + ['-O', 'exit', 'root@' + inst['ssh_host']], capture_output=True, timeout=30)
                subprocess.run(['ssh-keygen', '-R', entry, '-f', str(self.dir / 'known_hosts')], capture_output=True, timeout=30)
            time.sleep(10)
        # The multiplexed master may predate the final key; a fresh connection must pass with the pinned one.
        subprocess.run(self.ssh_base('agents') + ['-O', 'exit', 'root@' + inst['ssh_host']], capture_output=True, timeout=30)
        self.ssh('true', timeout=60, who='agents')

    def rent_one(self, key, label, offers, body):
        """Rent the cheapest offer, or keep the machine already rented for this run."""
        inst = self.state.get(key)
        if inst:
            if any(x['id'] == inst['id'] for x in instances()):
                self.emit('info', f'{label}: использую уже арендованный {inst["id"]}.')
                return inst
            with self.lock:
                self.state.setdefault('destroyed', []).append({**self.state.pop(key), 'key': key, 'vanished': time.time()})
            self.save(bootstrapped=False) if key == 'instance' else self.save()
            raise RuntimeError(f'{label} {inst["id"]} исчез из аккаунта Vast. Нажмите «Продолжить», чтобы взять новый'
                               + (': обучение продолжится с последнего сохранённого адаптера.' if key == 'instance' else '.'))
        if not offers:
            raise RuntimeError(f'{label}: нет подходящих офферов Vast под эти фильтры. Ослабьте цену, надёжность или требования.')
        offer = offers[0]
        body = {**body, 'label': f'antislop-lab-{key}-{self.state["run_id"]}', 'runtype': 'ssh', 'target_state': 'running', 'cancel_unavail': True}
        self.emit('info', f'{label}: арендую оффер {offer["id"]} ({offer.get("num_gpus") or 0}× {offer.get("gpu_name") or "CPU"}, '
                          f'{(offer.get("cpu_cores_effective") or 0):.0f} CPU, {(offer.get("cpu_ram") or 0) / 1000:.0f} ГБ RAM) за ≤ ${offer["hourly_bound"]:.2f}/ч, {offer.get("geolocation")}.')
        try:
            ident = vast('PUT', f'/asks/{offer["id"]}/', body)['new_contract']
        except (RuntimeError, KeyError):
            # The request can succeed although the reply was lost; the label tells.
            found = [x for x in instances() if x.get('label') == body['label']]
            if len(found) != 1:
                raise
            ident = found[0]['id']
        inst = {'id': ident, 'created': time.time(), 'hourly_bound': offer['hourly_bound'], 'machine_id': offer.get('machine_id'),
                'gpu_name': f'{offer.get("num_gpus") or 0}× {offer.get("gpu_name") or "CPU"}'}
        self.save(**{key: inst})
        public = Path(str(expand(self.v['local']['ssh_key'])) + '.pub')
        if public.exists():
            try:
                vast('POST', f'/instances/{ident}/ssh/', {'ssh_key': public.read_text().strip()})
            except RuntimeError as exc:
                self.emit('warn', f'{label}: SSH-ключ не привязан через API ({str(exc)[:120]}); рассчитываю на ключ из аккаунта.')
        return inst

    def start_guard(self, who):
        """Self-destruct on the machine itself: from now on it stops its own billing if this laptop goes silent."""
        root, limits, inst = (REMOTE, self.v['limits'], self.state['instance']) if who == 'gpu' else (AG, self.v['limits'], self.state['agents'])
        config = {'deadline': inst['created'] + limits['max_hours'] * 3600 + 1800, 'heartbeat_timeout': limits['dead_man_minutes'] * 60,
                  'first_grace': 3 * 3600}
        self.ssh(f'mkdir -p {root} && (command -v python3 >/dev/null || (apt-get update -qq && apt-get install -y -qq python3)) >/dev/null 2>&1; '
                 f'cat > {root}/guard.json', timeout=600, who=who, stdin=json.dumps(config).encode())
        self.ssh(f'cat > {root}/guard.py', who=who, stdin=GUARD.encode())
        self.ssh(f'touch {root}/heartbeat; ([ -s {root}/guard.pid ] && kill -0 $(cat {root}/guard.pid) 2>/dev/null) || '
                 f'(setsid nohup python3 {root}/guard.py {root} > {root}/guard.out 2>&1 < /dev/null &); sleep 3', who=who, timeout=60)
        log = self.ssh(f'tail -2 {root}/guard.jsonl 2>/dev/null; [ -s {root}/guard.pid ] && kill -0 $(cat {root}/guard.pid) 2>/dev/null && echo GUARD_ALIVE || echo GUARD_DEAD', who=who)
        if 'GUARD_DEAD' in log and 'no_instance_key' not in log:
            raise RuntimeError(f'Страж на {"GPU" if who == "gpu" else "агент-хосте"} не запустился: ' + self.ssh(f'tail -5 {root}/guard.out 2>/dev/null || true', who=who))
        if 'no_instance_key' in log:
            self.emit('warn', f'{"GPU" if who == "gpu" else "Агент-хост"}: Vast не выдал ключ инстанса, самоуничтожение без ноутбука недоступно. '
                              'Если этот компьютер выключится, машину придётся удалить вручную.')
        else:
            self.emit('info', f'{"GPU" if who == "gpu" else "Агент-хост"}: страж запущен, машина удалит себя после {limits["dead_man_minutes"]} мин без связи с ноутбуком.')
        inst['guard'] = 'no_instance_key' not in log
        self.save()

    def rent(self):
        self.stage('rent')
        agents_cfg = self.v['agents']
        if agents_cfg['where'] == 'vast':
            # The cheap machine first: if it cannot boot, no GPU money is spent.
            bad = set(self.state.get('bad_hosts', []))
            for _ in range(3):
                offers = [] if self.state.get('agents') else search_host_offers(agents_cfg, bad)
                if agents_cfg['offer_id'] and not self.state.get('agents'):
                    offers = [o for o in offers if o['id'] == agents_cfg['offer_id']]
                self.rent_one('agents', 'Агент-хост', offers, {'image': agents_cfg['image'], 'disk': agents_cfg['disk_gb']})
                try:
                    self.wait_ssh('agents', 'Агент-хост', minutes=15)
                    break
                except RuntimeError as exc:
                    inst = self.state.get('agents') or {}
                    self.emit('warn', f'Агент-хост не загрузился, беру другой: {str(exc)[:200]}')
                    bad.add(inst.get('machine_id'))
                    self.save(bad_hosts=sorted(x for x in bad if x))
                    self.destroy_one('agents', 'не загрузился')
            else:
                raise RuntimeError('Ни один агент-хост не загрузился за три попытки.')
            self.start_guard('agents')
        gpu = self.v['gpu']
        offers = [] if self.state.get('instance') else search_offers(gpu)
        if gpu['offer_id'] and not self.state.get('instance'):
            offers = [o for o in offers if o['id'] == gpu['offer_id']]
        self.rent_one('instance', 'GPU', offers, {'image': gpu['image'], 'disk': gpu['disk_gb'], 'onstart': f'mkdir -p {REMOTE}/output'})
        self.wait_ssh('instance', 'GPU', minutes=20)
        self.start_guard('gpu')
        self.start_background()

    def upload(self):
        self.stage('upload')
        local, train = self.v['local'], self.v['train']
        self.ssh(f'mkdir -p {REMOTE}/vendor/verl {REMOTE}/output {NATIVE}/audit {NATIVE}/output/checkpoints {NATIVE}/export')
        self.rsync(str(expand(local['verl'])) + '/', f'{REMOTE}/vendor/verl/', exclude=('.git', '.venv', '__pycache__'))
        staging = self.dir / 'remote'
        (staging / 'native').mkdir(parents=True, exist_ok=True)
        (staging / 'hf_sync.py').write_bytes((ROOT / 'hf_sync.py').read_bytes())
        for name in ('native_agent.py', 'native_compat.py', 'agent.yaml', 'reward_fn.py'):
            (staging / 'native' / name).write_bytes((expand(local['native']) / name).read_bytes())
        (staging / 'native/serve.key').write_bytes((self.dir / 'serve.key').read_bytes())
        (staging / 'native/group_size').write_text(str(int(train['samples_per_group'])))
        (staging / 'native/val_group_size').write_text(str(int(train['eval_samples'])))
        self.write_plan(staging)
        self.rsync(str(staging) + '/', f'{REMOTE}/', exclude=('guard.py',))
        self.ssh(f'chmod 600 {NATIVE}/serve.key')
        token = os.environ.get(self.v['hf']['token_env'])
        if token:
            self.ssh('umask 077; cat > /root/.hf_token', stdin=token.encode())
        self.emit('info', 'GPU: загружены verl, agent loop, скрипты и план данных.')
        if self.v['agents']['where'] == 'vast':
            self.upload_agents()

    def write_plan(self, staging):
        """Scripts and the data order, from the current config and restore point."""
        for name, text in scripts(self.v, self.offset()).items():
            (staging / name).write_text(text)
        train_rows, val_rows = data_rows(self.rows, self.v['train'], self.offset())
        (staging / 'native/data.json').write_text(json.dumps({'train': train_rows, 'val': val_rows}))
        return len(train_rows), len(val_rows)

    def upload_agents(self):
        local = self.v['local']
        self.ssh(f'mkdir -p {AG}/collector {AG}/images {AG}/bin {AG}/tasks {AG}/output', who='agents')
        collector = expand(local['collector'])
        self.rsync(str(collector.parent) + '/', f'{AG}/collector/', exclude=('__pycache__',), who='agents')
        self.rsync(str(expand(local['agent_images'])) + '/', f'{AG}/images/', who='agents')
        self.rsync(str(expand(local['docker_plugins'])) + '/', f'{AG}/bin/', who='agents')
        self.rsync(str(expand(local['task_root'])) + '/', f'{AG}/tasks/', who='agents')
        self.rsync(str(expand(local['requirements'])), f'{AG}/requirements.txt', who='agents')
        self.rsync(str(self.dir / 'protocol.json'), f'{AG}/protocol.json', who='agents')
        self.ssh(f'cat > {AG}/setup.sh', who='agents', stdin=agent_setup_script().encode())
        self.ssh(f'cat > {AG}/host_gateway.compose.yaml', who='agents', stdin=HOST_GATEWAY.encode())
        env = (f'ANTISLOP_SERVE_KEY={(self.dir / "serve.key").read_text().strip()}\nDOCKER_CONFIG={AG}/docker-config\n'
               f'ANTISLOP_EXTRA_COMPOSE={AG}/host_gateway.compose.yaml\nPATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n')
        self.ssh(f'umask 077; cat > {AG}/run.env', who='agents', stdin=env.encode())
        self.emit('info', 'Агент-хост: загружены сборщик, задачи Harbor, Dockerfile агентов и зависимости.')

    def bootstrap(self):
        """GPU environment and model, and agent host setup, in parallel: both take tens of minutes."""
        self.stage('bootstrap')
        jobs = {}
        if self.ssh(f'test -f {REMOTE}/BOOTSTRAP_DONE && echo yes || true').strip() != 'yes':
            self.emit('info', 'GPU: ставлю окружение и скачиваю модель, обычно 20–40 минут.')
            jobs['gpu'] = lambda: self.job('bootstrap', f'bash {REMOTE}/bootstrap.sh', self.v['local']['bootstrap_timeout_min'])
        if self.v['agents']['where'] == 'vast' and self.ssh(f'test -f {AG}/SETUP_DONE && echo yes || true', who='agents').strip() != 'yes':
            self.emit('info', 'Агент-хост: ставлю Docker-плагины, Python и собираю образы Hermes и Pi.')
            jobs['agents'] = lambda: self.job('agents-setup', f'bash {AG}/setup.sh', self.v['agents']['setup_timeout_min'], who='agents')
        if jobs:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = {name: pool.submit(fn) for name, fn in jobs.items()}
                errors = []
                for name, future in futures.items():
                    try:
                        future.result()
                    except Paused:
                        raise
                    except Exception as exc:  # noqa: BLE001 - both are reported together
                        errors.append(f'{"GPU" if name == "gpu" else "Агент-хост"}: {exc}')
                if errors:
                    raise RuntimeError('\n\n'.join(errors))
        self.save(bootstrapped=True)
        self.plan_restore()
        staging = self.dir / 'remote'
        rows = self.write_plan(staging)
        # The config may have changed on «Продолжить», and a restore moves the start: always send the current plan.
        self.rsync(str(staging / 'native/run_verl.sh'), f'{NATIVE}/run_verl.sh')
        self.rsync(str(staging / 'native/data.json'), f'{NATIVE}/data.json')
        self.emit('info', f'План: {rows[0]} обучающих задач с шага {self.offset() + 1}, {rows[1]} отложенных. '
                          + self.ssh(f'{VENV}/python {NATIVE}/make_data.py', timeout=300).strip())
        self.stage('config')
        try:
            resolved = self.ssh(f'bash {NATIVE}/run_verl.sh --cfg job 2>&1', timeout=300)
        except RuntimeError as exc:
            raise RuntimeError('verl не принял конфиг (hydra):\n' + str(exc)) from exc
        (self.dir / 'resolved.yaml').write_text(resolved)
        self.emit('info', f'verl принял конфиг; полный вид сохранён в {self.dir / "resolved.yaml"}.')

    # -- tunnels

    def agents_bridge(self):
        return self.ssh("ip -4 -o addr show docker0 | awk '{print $4}' | cut -d/ -f1", who='agents').strip() or '172.17.0.1'

    def tunnel_to_agents(self):
        """The GPU dials the agent host and exposes the policy proxy on its Docker bridge."""
        agents = self.state['agents']
        bridge = agents.get('bridge') or self.agents_bridge()
        if not agents.get('gpu_key'):
            public = self.ssh('mkdir -p ~/.ssh && ([ -f ~/.ssh/to_agents ] || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/to_agents) && cat ~/.ssh/to_agents.pub').strip()
            self.ssh('mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys', who='agents', stdin=(public + '\n').encode())
            agents.update(gpu_key=True, bridge=bridge)
            self.save(agents=agents)
        # The direct address first: the proxy may not forward non-account keys.
        host, port = (agents.get('endpoints') or [(agents['ssh_host'], agents['ssh_port'])])[-1]
        self.ssh(f'rm -f {REMOTE}/STOP_TUNNEL; ([ -s {REMOTE}/to_agents.pid ] && kill -0 $(cat {REMOTE}/to_agents.pid) 2>/dev/null) || '
                 f'(setsid nohup bash {REMOTE}/to_agents.sh {host} {port} {bridge} > {REMOTE}/output/to_agents.log 2>&1 < /dev/null &)')
        for _ in range(24):
            time.sleep(5)
            if self.ssh(f"ss -ltn | grep -q '{bridge}:{AGENT_PORT} ' && echo up || echo down", who='agents').strip() == 'up':
                self.emit('info', f'Туннель GPU → агент-хост поднят: агенты видят политику на {bridge}:{AGENT_PORT}.')
                return
        raise RuntimeError('GPU не смог открыть туннель к агент-хосту.\n' + self.ssh(f'tail -5 {REMOTE}/output/to_agents.log || true'))

    def keep_tunnel(self):
        """This laptop: -R lets the GPU reach the coordinator here; -L lets local agents reach the GPU (local mode only)."""
        local = self.v['local']
        forward = ['-o', 'ExitOnForwardFailure=yes', '-N', '-R', f'127.0.0.1:18975:127.0.0.1:{int(local["coordinator_port"])}']
        if self.v['agents']['where'] == 'local':
            forward += ['-L', f'0.0.0.0:{int(local["tunnel_port"])}:127.0.0.1:9001']

        def loop():
            while not self.tunnel_stop.is_set():
                if self.tunnel is None or self.tunnel.poll() is not None:
                    if self.tunnel is not None:
                        self.emit('warn', 'SSH-туннель ноутбук ↔ GPU упал, поднимаю заново.')
                    try:
                        self.tunnel = subprocess.Popen(self.tunnel_base() + forward + ['root@' + self.state['instance']['ssh_host']],
                                                       stdout=(self.dir / 'tunnel.log').open('a'), stderr=subprocess.STDOUT)
                    except (OSError, KeyError) as exc:
                        self.emit('warn', f'Туннель не запускается: {exc}')
                self.tunnel_stop.wait(5)

        self.tunnel_stop = threading.Event()
        threading.Thread(target=loop, daemon=True).start()
        time.sleep(6)
        if self.tunnel is None or self.tunnel.poll() is not None:
            raise RuntimeError('SSH-туннель не поднялся: ' + (self.dir / 'tunnel.log').read_text(errors='replace')[-400:])
        if self.v['agents']['where'] == 'vast':
            self.tunnel_to_agents()

    # -- background: heartbeat, disk, HF

    def start_background(self):
        if self.threads_started:
            return
        self.threads_started = True
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        threading.Thread(target=self.hf_loop, daemon=True).start()

    def heartbeat_loop(self):
        """Keeps the guards from destroying healthy machines, checks the GPU disk and the agent tunnel."""
        while not self.background.is_set():
            for who, inst in self.machines():
                if not inst.get('ssh_host'):
                    continue
                root = REMOTE if who == 'gpu' else AG
                try:
                    out = self.ssh(f'touch {root}/heartbeat; df -BG --output=avail {"/workspace" if who == "gpu" else "/"} | tail -1', who=who, timeout=40)
                    free = int(out.strip().rstrip('G') or 0)
                    if free < 20:
                        self.emit('warn', f'{"GPU" if who == "gpu" else "Агент-хост"}: свободно {free} ГБ диска. Удаляю старые экспорты и прогоны.')
                        self.ssh(f'ls -dt {NATIVE}/export/step_* 2>/dev/null | tail -n +3 | xargs -r rm -rf' if who == 'gpu' else
                                 f'find {AG}/rollouts -mindepth 1 -maxdepth 1 -mmin +60 -exec rm -rf {{}} +; docker system prune -f >/dev/null', who=who)
                except (RuntimeError, subprocess.TimeoutExpired, ValueError, KeyError) as exc:
                    self.emit('warn', f'Пульс до {"GPU" if who == "gpu" else "агент-хоста"} не прошёл: {str(exc)[:160]}')
            if self.state.get('stage') == 'train' and self.v['agents']['where'] == 'vast' and self.state.get('agents', {}).get('bridge'):
                try:
                    bridge = self.state['agents']['bridge']
                    if self.ssh(f"ss -ltn | grep -q '{bridge}:{AGENT_PORT} ' && echo up || echo down", who='agents').strip() != 'up':
                        self.emit('warn', 'Туннель GPU → агент-хост пропал, поднимаю заново.')
                        self.tunnel_to_agents()
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    self.emit('warn', f'Проверка туннеля к агент-хосту не прошла: {str(exc)[:160]}')
            self.background.wait(60)

    def hf_repo(self):
        hf = self.v['hf']
        if hf['backup_repo']:
            return hf['backup_repo']
        if not self.state.get('hf_user'):
            request = urllib.request.Request('https://huggingface.co/api/whoami-v2', headers={'Authorization': 'Bearer ' + os.environ[hf['token_env']]})
            with urllib.request.urlopen(request, timeout=30) as response:
                self.save(hf_user=json.load(response)['name'])
        return f'{self.state["hf_user"]}/antislop-lab-{self.state["run_id"]}'

    def export(self, step):
        """verl checkpoint of absolute step `step` -> PEFT adapter on the GPU box. Small, directly usable by vLLM and PEFT."""
        own = step - self.offset()
        if own < 1:
            raise RuntimeError(f'Шаг {step} обучался на прошлом GPU; на этом есть шаги после {self.offset()}.')
        out = f'{NATIVE}/export/step_{step:04d}'
        if self.ssh(f'test -f {out}/adapter_model.safetensors && echo yes || true').strip() == 'yes':
            return out
        self.job(f'export-{step}', f'{VENV}/python {REMOTE}/hf_sync.py export --checkpoint {NATIVE}/output/checkpoints/global_step_{own}/actor '
                                   f'--out {out} --base-id {shlex.quote(self.v["model"]["hf_id"])} --revision {shlex.quote(self.v["model"]["revision"])}',
                 20, quiet=True, guarded=False)
        self.ssh(f'ls -dt {NATIVE}/export/step_* | tail -n +4 | xargs -r rm -rf')
        return out

    def hf_loop(self):
        """Backups and publish requests. Never pauses the run: a full Hub only pauses backups."""
        failures = 0
        while not self.background.is_set():
            try:
                if (self.dir / 'HF_RESUME').exists():
                    (self.dir / 'HF_RESUME').unlink()
                    self.cfg['vast']['hf'] = core.merged(core.DEFAULT_CONFIG, json.loads((self.dir / 'config.json').read_text()))['vast']['hf']
                    self.v = self.cfg['vast']
                    failures = 0
                    self.save(hf_paused=None)
                    self.emit('info', f'Бэкапы на Hugging Face возобновлены: {self.hf_repo()}.')
                self.handle_publish()
                hf = self.v['hf']
                if (hf['backup'] and os.environ.get(hf['token_env']) and not self.state.get('hf_paused') and self.state.get('bootstrapped')
                        and self.state.get('instance', {}).get('ssh_host')):
                    own = self.latest_checkpoint()
                    step = own + self.offset() if own else 0
                    last = self.state.get('hf_backed', 0)
                    if step and step > last and (not last or step - last >= hf['backup_every']):
                        folder, repo = self.export(step), self.hf_repo()
                        try:
                            self.job(f'hf-backup-{step}', f'{VENV}/python {REMOTE}/hf_sync.py backup --folder {folder} --repo {repo} '
                                                          f'--name step_{step:04d} --keep {int(hf["keep"])}', 60, quiet=True, guarded=False)
                            self.save(hf_backed=step, hf_repo=repo)
                            self.emit('info', f'Адаптер шага {step} сохранён на Hugging Face: {repo} (приватно, хранятся последние {hf["keep"]}).')
                            failures = 0
                        except RuntimeError as exc:
                            if 'завершилось с кодом 3' in str(exc):
                                self.save(hf_paused=HF_HINT)
                                self.emit('error', 'Бэкап на Hugging Face остановлен: нет места.', hint=HF_HINT)
                            else:
                                raise
            except Paused:
                pass
            except Exception as exc:  # noqa: BLE001 - backups must never break training
                failures += 1
                self.emit('warn', f'Бэкап на Hugging Face не удался ({failures} подряд): {str(exc)[-400:]}')
                if failures >= 3:
                    self.save(hf_paused='Три ошибки бэкапа подряд: ' + str(exc)[-300:])
                    self.emit('error', 'Бэкапы на HF приостановлены после трёх ошибок подряд; обучение продолжается.',
                              hint='Проверьте токен и репозиторий на вкладке Vast и нажмите «Возобновить бэкапы».')
            self.background.wait(120)

    def handle_publish(self):
        request_path = self.dir / 'PUBLISH.json'
        if not request_path.exists():
            return
        request = json.loads(request_path.read_text())
        request_path.unlink()
        try:
            own = self.latest_checkpoint()
            step = int(request.get('step') or (own + self.offset() if own else 0))
            if not step:
                raise RuntimeError('Ещё нет ни одного чекпойнта.')
            self.emit('info', f'Публикую шаг {step} в {request["repo"]} ({"публично" if request["public"] else "приватно"}, '
                              f'{"слитая модель" if request["merged"] else "адаптер"}).')
            folder = self.export(step)
            card = model_card(self.cfg, self.state, step, request['merged'])
            self.ssh(f'cat > {NATIVE}/export/README-{step}.md', stdin=card.encode())
            flags = (' --public' if request['public'] else '') + (' --merged --base /workspace/model' if request['merged'] else '')
            self.job(f'publish-{step}', f'{VENV}/python {REMOTE}/hf_sync.py publish --folder {folder} --repo {shlex.quote(request["repo"])}'
                                        f'{flags} --card {NATIVE}/export/README-{step}.md', 240 if request['merged'] else 30, guarded=False)
            entry = {'repo': request['repo'], 'step': step, 'public': request['public'], 'merged': request['merged'], 'ts': time.time(),
                     'url': f'https://huggingface.co/{request["repo"]}'}
            self.save(published=self.state.get('published', []) + [entry])
            self.emit('info', f'Опубликовано: {entry["url"]}')
        except Exception as exc:  # noqa: BLE001 - shown in the UI, the run goes on
            self.emit('error', f'Публикация не удалась: {str(exc)[-1500:]}', hint=explain(str(exc)))

    # -- training

    def remote_cleanup(self):
        """Nothing from an earlier attempt may hold the GPUs."""
        self.ssh(f"for f in {REMOTE}/jobs/train-*.pid; do [ -s $f ] && kill -9 -- -$(cat $f) 2>/dev/null; done; "
                 f"{VENV}/ray stop --force > /dev/null 2>&1; pkill -9 -f '[V]LLM::'; true", timeout=180)
        if self.v['agents']['where'] == 'local':
            subprocess.run(['pkill', '-f', '[h]ermes_collect.py'], check=False)
        elif self.state.get('agents'):
            self.ssh("pkill -f '[h]ermes_collect.py'; docker ps -q | xargs -r docker rm -f >/dev/null 2>&1; true", who='agents', timeout=180)

    def read_metrics(self):
        """New lines of verl's metrics file, folded into the state for the curves."""
        try:
            out = self.ssh(f'tail -c +{self.metrics_offset + 1} {NATIVE}/output/metrics.jsonl 2>/dev/null | head -c 2000000', timeout=60, raw=True)
        except (RuntimeError, subprocess.TimeoutExpired):
            return
        whole = out[:out.rfind(b'\n') + 1]
        if not whole:
            return
        self.metrics_offset += len(whole)
        history = self.state.get('history', [])
        for line in whole.splitlines():
            try:
                row = json.loads(line)
                history.append({'step': row['step'] + self.offset(), **pick_metrics(row['data'])})
            except (ValueError, KeyError, TypeError):
                continue
        self.check_logprob_gap(history)
        step = max((h['step'] for h in history), default=0)
        self.save(history=history[-3000:], completed=max(step, self.state.get('completed', 0)))
        every = int(self.v['train']['download_every'])
        if every and step // every > self.state.get('last_downloaded', 0) // every:
            try:
                self.download_checkpoint()
            except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
                self.emit('warn', f'Плановое скачивание адаптера не удалось, обучение продолжается: {str(exc)[:200]}')

    def check_logprob_gap(self, history):
        """Generator and trainer must see the same model; stop before a whole run trains on wrong logprobs."""
        limit = float(self.v['train'].get('max_ppo_kl', 0) or 0)
        kls = [h['actor/ppo_kl'] for h in history if isinstance(h.get('actor/ppo_kl'), (int, float))][-3:]
        if not limit or len(kls) < 3 or sorted(kls)[1] <= limit:
            return
        self.fatal = (f'Logprob mismatch: actor/ppo_kl {", ".join(f"{k:.4f}" for k in kls)} > {limit} '
                      '(норма ~0.001). Генератор и тренер видят разные модели; обучать дальше нельзя.')

    def latest_checkpoint(self):
        """verl's own step of its newest checkpoint on this GPU box (0 = none)."""
        return int(self.ssh(f'cat {NATIVE}/output/checkpoints/latest_checkpointed_iteration.txt 2>/dev/null || echo 0').strip() or 0)

    def plan_restore(self):
        """A new GPU box after a lost one: start LoRA from the newest saved adapter instead of from zero."""
        if self.state.get('restore_for') == self.state['instance']['id'] or self.latest_checkpoint():
            return
        if not any(d.get('key') == 'instance' for d in self.state.get('destroyed', [])):
            return
        local, hf = self.state.get('last_downloaded', 0), self.state.get('hf_backed', 0)
        best = max(local, hf)
        if not best:
            self.emit('warn', 'Новый GPU, а сохранённых адаптеров нет: обучение начнётся с нуля.')
            self.save(restore_for=self.state['instance']['id'], step_offset=0, restore_path='')
            return
        target = f'{NATIVE}/restore'
        if local >= hf:
            self.ssh(f'rm -rf {target} && mkdir -p {target}')
            self.rsync(str(self.dir / 'checkpoints' / f'step_{local:04d}') + '/', target + '/')
            source = 'с этого компьютера'
        else:
            self.job('hf-restore', f'{VENV}/python {REMOTE}/hf_sync.py fetch --repo {self.state["hf_repo"]} --name step_{hf:04d} --out {target}',
                     30, guarded=False)
            source = 'с Hugging Face'
        self.save(restore_for=self.state['instance']['id'], step_offset=best, restore_path=target, completed=best)
        self.emit('info', f'Продолжаю с адаптера шага {best} ({source}). Состояние оптимизатора не сохранялось, поэтому первые шаги после восстановления шумнее.')

    def download_checkpoint(self):
        own = self.latest_checkpoint()
        step = own + self.offset() if own else 0
        if not step or step == self.state.get('last_downloaded'):
            return
        folder = self.export(step)
        path = self.dir / 'checkpoints' / f'step_{step:04d}'
        path.mkdir(parents=True, exist_ok=True)
        self.rsync(str(path) + '/', folder + '/', upload=False)
        self.save(last_downloaded=step)
        self.emit('info', f'Адаптер шага {step} скачан: {path}')

    def train(self):
        train = self.v['train']
        if self.coordinator is None:
            self.coordinator = Coordinator(self)
            self.server = self.coordinator.serve()
        self.coordinator.circuit, self.fatal = False, None
        self.coordinator.recent.clear()
        self.keep_tunnel()
        attempt = 0
        while True:
            self.stage('train')
            self.remote_cleanup()
            latest = self.latest_checkpoint()
            resume = (f'trainer.resume_mode=resume_path trainer.resume_from_path={NATIVE}/output/checkpoints/global_step_{latest} trainer.val_before_train=False'
                      if latest else 'trainer.resume_mode=disable')
            if not latest and self.state.get('restore_path'):
                resume += f' actor_rollout_ref.model.lora_adapter_path={self.state["restore_path"]}'
            self.metrics_offset = int(self.ssh(f'stat -c %s {NATIVE}/output/metrics.jsonl 2>/dev/null || echo 0').strip() or 0) if latest else 0
            start = (f', продолжение с шага {latest + self.offset()}.' if latest else
                     f' с адаптера шага {self.offset()}.' if self.offset() else ' с нуля. Сначала базовая проверка на отложенных задачах.')
            self.emit('info', f'Запускаю verl ({train["mode"]})' + start)
            try:
                self.job(f'train-{int(time.time())}', f'bash {NATIVE}/run_verl.sh {resume}', self.v['limits']['max_hours'] * 60 + 30, tick=self.read_metrics)
                self.read_metrics()
                return
            except RuntimeError as exc:
                text = str(exc)
                if self.latest_checkpoint() + self.offset() >= int(train['steps']):
                    # Seen on the first real run: all steps saved, then a DataLoader worker killed during interpreter exit.
                    self.emit('warn', 'verl сохранил последний шаг, но вышел с ошибкой при завершении; считаю обучение законченным.\n' + text[-800:])
                    self.read_metrics()
                    return
                progressed = self.latest_checkpoint() > latest
                attempt = 0 if progressed else attempt + 1
                if any(mark in text for mark in FATAL) or attempt > int(train['auto_retries']):
                    raise
                self.emit('warn', f'verl упал, перезапускаю с последнего чекпойнта (попытка {attempt} из {train["auto_retries"]}).\n{text[-2500:]}',
                          hint=explain(text))

    def work(self):
        if self.stop_requested():
            raise Paused()
        self.preflight()
        self.rent()
        self.upload()
        self.bootstrap()
        self.train()

    # -- the end

    def finish(self):
        """Save what the GPU holds, then give the machines back."""
        self.stage('finish')
        if getattr(self, 'tunnel_stop', None):
            self.tunnel_stop.set()
        while True:
            inst = self.state.get('instance')
            if not inst or not inst.get('ssh_host'):
                break
            try:
                self.remote_cleanup()
                if self.state.get('bootstrapped'):
                    self.download_checkpoint()
                    self.handle_publish()
                    (self.dir / 'gpu_output').mkdir(exist_ok=True)
                    self.rsync(str(self.dir / 'gpu_output') + '/', f'{NATIVE}/output/', upload=False, exclude=('checkpoints',))
                break
            except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
                hint = 'Машины пока живы. «Продолжить» повторит скачивание; иначе они будут уничтожены по таймеру простоя.'
                self.emit('error', f'Не удалось забрать результаты с GPU: {exc}', hint=hint)
                self.save(status='paused', error={'stage': 'finish', 'message': str(exc), 'hint': hint, 'ts': time.time()})
                (self.dir / 'STOP').unlink(missing_ok=True)
                if not self.wait_while_paused():
                    break
        self.background.set()
        if self.tunnel is not None:
            self.tunnel.terminate()
        if self.v['limits']['destroy_on_finish'] or self.state.get('status') == 'paused':
            for key, label in (('instance', 'GPU'), ('agents', 'агент-хост')):
                if self.state.get(key):
                    self.destroy_one(key, 'прогон завершён')
        left = [inst['id'] for _, inst in self.machines()]
        self.save(status='finished', stage='finished', finished=time.time(), error=None)
        self.emit('warn' if left else 'info', f'Прогон завершён. Оставлены включёнными: {left}; они продолжают стоить денег.' if left else 'Прогон завершён.')

    def destroy_one(self, key, reason):
        inst = self.state.get(key)
        if not inst:
            return
        label = 'GPU' if key == 'instance' else 'Агент-хост'
        for attempt in range(20):
            try:
                destroy(inst['id'])
                break
            except RuntimeError as exc:
                if '404' in str(exc):
                    break
                self.emit('warn', f'{label}: уничтожение, попытка {attempt + 1}: {str(exc)[:160]}')
                time.sleep(15)
        else:
            self.emit('error', f'{label} {inst["id"]} НЕ уничтожен. Удалите его вручную в консоли Vast.')
            return
        self.emit('info', f'{label} {inst["id"]} уничтожен ({reason}).')
        with self.lock:
            self.state.setdefault('destroyed', []).append({**self.state.pop(key), 'key': key, 'destroyed': time.time()})
        self.save()

    def wait_while_paused(self):
        """After an error: keep the machines and wait for the user, but not forever."""
        deadline = time.time() + self.v['limits']['idle_destroy_minutes'] * 60
        while True:
            if (self.dir / 'RESUME').exists():
                (self.dir / 'RESUME').unlink()
                (self.dir / 'STOP').unlink(missing_ok=True)
                self.reload()
                self.emit('info', 'Продолжаю с текущим конфигом.')
                return True
            if self.stop_requested():
                return False
            if time.time() > deadline:
                self.emit('warn', f'Пауза дольше {self.v["limits"]["idle_destroy_minutes"]} мин: завершаю прогон, чтобы машины не простаивали.')
                return False
            self.handle_publish()
            time.sleep(3)

    def run(self):
        self.save(status='running', pid=os.getpid(), error=None)
        self.emit('info', f'Раннер запущен, прогон {self.state["run_id"]}.')
        if self.machines():
            self.start_background()      # a restarted runner resumes the heartbeat before anything else
        while True:
            try:
                self.work()
                break
            except Paused:
                self.emit('info', 'Остановка по запросу.')
                break
            except Exception as exc:  # noqa: BLE001 - every failure becomes a visible, resumable pause
                text = str(exc) if isinstance(exc, RuntimeError) else f'{type(exc).__name__}: {exc}'
                hint = explain(text)
                for key, word in (('instance', 'GPU'), ('agents', 'Агент-хост')):
                    if text.startswith(f'Vast удалил {word}') and self.state.get(key):
                        with self.lock:
                            self.state.setdefault('destroyed', []).append({**self.state.pop(key), 'key': key, 'vanished': time.time()})
                        self.save(bootstrapped=False) if key == 'instance' else self.save()
                self.emit('error', text[:6000], hint=hint)
                self.save(status='paused', error={'stage': self.state.get('stage'), 'message': text[:6000], 'hint': hint, 'ts': time.time()})
                if getattr(self, 'tunnel_stop', None):
                    self.tunnel_stop.set()
                if not self.wait_while_paused():
                    break
                self.save(status='running', error=None)
        self.finish()


def new_run(cfg):
    """Create the run directory with everything that must not change during the run."""
    run_id = time.strftime('%m%d-%H%M%S')
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True)
    core.atomic_json(run_dir / 'config.json', cfg)
    rows = [json.loads(line) for line in expand(cfg['vast']['local']['tasks']).read_text().splitlines()]
    protocol = {'name': 'antislop-lab-' + run_id, 'base_revision': cfg['vast']['model']['revision'], 'source_commit': 'lab',
                'tasks': [{'task': r['task'], 'slug': r['task'], 'split': 'dev' if r['split'] == 'heldout' else 'train'} for r in rows]}
    (run_dir / 'protocol.json').write_text(json.dumps(protocol, indent=1) + '\n')
    key = run_dir / 'serve.key'
    key.write_text(secrets.token_hex(24) + '\n')
    key.chmod(0o600)
    core.atomic_json(run_dir / 'state.json', {'run_id': run_id, 'status': 'starting', 'stage': 'starting', 'completed': 0, 'history': [],
                                              'groups': [], 'created': time.time()})
    core.atomic_json(RUNS / 'current.json', {'run_id': run_id})
    return run_dir


def publish_local(run_dir, request):
    """Publish an adapter that is already downloaded, when no GPU box is left. Uses the local Python with huggingface_hub."""
    run_dir = Path(run_dir)
    cfg = core.validate_config(core.merged(core.DEFAULT_CONFIG, json.loads((run_dir / 'config.json').read_text())))
    state = json.loads((run_dir / 'state.json').read_text())
    if request['merged']:
        raise ValueError('Слитую модель можно собрать только на GPU-машине; она уже возвращена. Опубликуйте адаптер.')
    folders = sorted((run_dir / 'checkpoints').glob('step_*'))
    if request.get('step'):
        folders = [f for f in folders if f.name == f'step_{int(request["step"]):04d}']
    if not folders:
        raise ValueError('Нет скачанного адаптера этого прогона.')
    folder, step = folders[-1], int(folders[-1].name.split('_')[1])
    token = os.environ.get(cfg['vast']['hf']['token_env'])
    if not token:
        raise ValueError(f'Нет токена Hugging Face в {cfg["vast"]["hf"]["token_env"]}')
    card = run_dir / f'README-{step}.md'
    card.write_text(model_card(cfg, state, step, False))
    token_file = run_dir / '.hf_token'
    token_file.write_text(token)
    token_file.chmod(0o600)
    try:
        result = subprocess.run([str(expand(cfg['vast']['local']['python'])), str(ROOT / 'hf_sync.py'), 'publish', '--folder', str(folder),
                                 '--repo', request['repo'], '--card', str(card)] + (['--public'] if request['public'] else []),
                                env={**os.environ, 'HF_TOKEN_FILE': str(token_file)}, capture_output=True, text=True, timeout=3600)
    finally:
        token_file.unlink(missing_ok=True)
    if result.returncode:
        raise RuntimeError('Публикация не удалась:\n' + (result.stderr or result.stdout)[-1500:])
    entry = {'repo': request['repo'], 'step': step, 'public': request['public'], 'merged': False, 'ts': time.time(),
             'url': f'https://huggingface.co/{request["repo"]}'}
    state.setdefault('published', []).append(entry)
    core.atomic_json(run_dir / 'state.json', state)
    return entry


if __name__ == '__main__':
    core.load_env()
    Runner(sys.argv[1]).run()
