"""Loopback-only lab: tune the judge, check it against Pangram labels, compare generators, launch training.

Run: python3 judge_playground/lab_server.py   (LAB_PORT, default 8770)
Training code gets rewards from POST /api/reward.
"""
import json
import re
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import lab_core as core
import blind
import vast_core as vast

ROOT, PROJECT, STATE = core.ROOT, core.PROJECT, core.STATE
RUNS = STATE / 'runs'
REWARD_LOG = STATE / 'rewards.jsonl'
TRAIN_LOG = STATE / 'train.log'
JOBS, MODELS, TRAIN = {}, {}, {'process': None, 'started': None, 'command': None}
LOG_LOCK = threading.Lock()

# Prompt files that can be loaded into the judge prompt from the UI.
PROMPT_FILES = {
    'Pangram-эмулятор (Opus, калиброванный)': 'pangram_opus/inference_system.txt',
    'Opus antislop по рулбуку (калиброванный, cmp)': 'cmp/antislop-copywriting.md',
    'Промпт по умолчанию': None,
}
DATASETS = ['tasks_v2/opus_calib.jsonl', 'tasks_v2/task_calib.jsonl', 'pangram_opus/round5/test.jsonl', 'pangram_opus/round6/train.jsonl', 'pangram_opus/round6/dev.jsonl', 'pangram_opus/data/train.jsonl',
            'pangram_opus/data/test.jsonl']


def inside_project(relative):
    path = (PROJECT / relative).resolve()
    if PROJECT.resolve() not in path.parents:
        raise ValueError('Path must be inside the project')
    return path


def start_job(fn):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {'status': 'running', 'done': 0, 'total': 0, 'result': None, 'error': None}

    def progress(done, total):
        JOBS[job_id].update(done=done, total=total)

    def run():
        try:
            JOBS[job_id].update(result=fn(progress), status='done')
        except Exception as exc:  # shown to the local user; keys are never part of these messages
            JOBS[job_id].update(error=f'{type(exc).__name__}: {exc}', status='failed')

    threading.Thread(target=run, daemon=True).start()
    return {'job': job_id}


def score_request(data, cfg):
    answers = data.get('answers')
    if not isinstance(answers, list) or not answers or any(not isinstance(a, str) for a in answers):
        raise ValueError('answers must be a non-empty list of strings')
    result = core.score_group(cfg, data.get('task'), answers, data.get('target_words') or None, data.get('lang') or None,
                              bool(data.get('allow_markdown')), data.get('repeat', 0))
    record = {'ts': time.time(), 'step': data.get('step'), 'source': data.get('source', 'trainer'), 'task': data.get('task'),
              'answers': answers, 'items': result['items'], 'stats': result['stats'], 'cost_usd': result['cost_usd']}
    with LOG_LOCK:
        STATE.mkdir(exist_ok=True)
        with REWARD_LOG.open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return result


def dataset_run(data, cfg, progress):
    rows = core.load_labeled(inside_project(data['path']), int(data.get('limit', 100)), int(data.get('min_words', 0)),
                             int(data.get('max_words', 100000)), int(data.get('seed', 0)))
    if not rows:
        raise ValueError('No labeled rows match these filters')
    full = cfg
    if any(r['task_set'] for r in rows):   # rows labeled ok / not ok are a task-judge set
        cfg = core.task_view(cfg)
    result = core.evaluate_labeled(cfg, rows, int(data.get('repeats', 1)), progress)
    result.update(name=data.get('name') or time.strftime('%m-%d %H:%M'), ts=time.time(), path=data['path'],
                  model=cfg['judge']['model'], mode=cfg['judge']['mode'], reasoning=cfg['judge']['reasoning_effort'],
                  ai_when=cfg['calibration']['ai_when'], prompt_sha=core.digest(cfg['judge']['system'])[:12], config=full, task_set=cfg is not full)
    core.atomic_json(RUNS / (time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6] + '.json'), result)
    return result


def generate_run(data, cfg, progress):
    chosen = [g for g in cfg['generators'] if g['name'] in data.get('generators', [])]
    if not chosen:
        raise ValueError('Pick at least one generator')
    n = max(1, min(16, int(data.get('n', 1))))
    out = []
    for index, endpoint in enumerate(chosen):
        texts, cost = core.generate(endpoint, data.get('system', ''), data['prompt'], n, float(data.get('temperature', 1.0)),
                                    int(data.get('max_tokens', 1500)), cfg['budget_usd'])
        entry = {'name': endpoint['name'], 'model': endpoint['model'], 'texts': texts, 'gen_cost_usd': cost}
        if data.get('judge', True):
            entry['score'] = score_request({'task': data['prompt'], 'answers': texts, 'target_words': data.get('target_words'),
                                            'lang': data.get('lang'), 'allow_markdown': data.get('allow_markdown'),
                                            'source': 'generator:' + endpoint['name']}, cfg)
        out.append(entry)
        progress(index + 1, len(chosen))
    return out


def list_runs():
    runs = []
    for path in sorted(RUNS.glob('*.json'), reverse=True)[:50]:
        run = json.loads(path.read_text())
        run.pop('config', None)
        run['file'] = path.name
        runs.append(run)
    return runs


def openrouter_models():
    if 'ids' not in MODELS:
        with urllib.request.urlopen('https://openrouter.ai/api/v1/models', timeout=20) as response:
            MODELS['ids'] = sorted(m['id'] for m in json.load(response)['data'])
    return MODELS['ids']


# ---------------------------------------------------------------- training

def train_running():
    return TRAIN['process'] is not None and TRAIN['process'].poll() is None


def train_start(cfg, port):
    if train_running():
        raise ValueError('A training run is already going')
    train = cfg['train']
    if not train['command'].strip():
        raise ValueError('Set the launch command first')
    params = {**train['params'], 'config': str(core.CONFIG_PATH), 'reward_url': f'http://127.0.0.1:{port}/api/reward'}
    try:
        command = train['command'].format_map(params)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f'Unknown placeholder in the command: {exc}') from exc
    env = {**os.environ, 'JUDGE_CONFIG': params['config'], 'REWARD_URL': params['reward_url'],
           **{'TRAIN_' + k.upper(): str(v) for k, v in train['params'].items()}}
    STATE.mkdir(exist_ok=True)
    log = TRAIN_LOG.open('w')
    log.write('$ ' + command + '\n')
    log.flush()
    TRAIN.update(process=subprocess.Popen(['bash', '-c', command], cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True), started=time.time(), command=command)
    return train_status(cfg)


def train_stop():
    if not train_running():
        raise ValueError('No training run is going')
    os.killpg(TRAIN['process'].pid, signal.SIGTERM)
    return {'stopping': True}


def tail(path, size=6000):
    if not path.exists():
        return ''
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - size))
        return handle.read().decode('utf-8', 'replace')


def reward_records(limit=4000):
    if not REWARD_LOG.exists():
        return []
    return [json.loads(line) for line in REWARD_LOG.read_text().splitlines()[-limit:]]


def reward_curve(records):
    """Per-step averages from the reward log, so curves exist whatever the training backend is."""
    steps = {}
    for index, record in enumerate(r for r in records if r.get('source') == 'trainer'):
        steps.setdefault(record['step'] if record.get('step') is not None else index, []).append(record)
    curve = []
    for step in sorted(steps, key=lambda s: (isinstance(s, str), s)):
        groups = steps[step]
        items = [i for g in groups for i in g['items'] if not i.get('empty')]
        point = {'step': step, 'groups': len(groups),
                 'reward_mean': sum(g['stats']['mean'] for g in groups) / len(groups),
                 'group_std': sum(g['stats']['std'] for g in groups) / len(groups),
                 'tied_share': sum(g['stats']['tied'] for g in groups) / len(groups)}
        if items:
            point['words'] = sum(i['words'] for i in items) / len(items)
            for flag in items[0]['flags']:
                point[flag + '_rate'] = sum(i['flags'][flag] for i in items) / len(items)
            for field in items[0]['scores']:
                point[field] = sum(i['scores'][field] for i in items) / len(items)
        curve.append(point)
    return curve


def metrics_file(cfg):
    relative = cfg['train'].get('metrics_path', '').strip()
    if not relative:
        return []
    path = inside_project(relative)
    rows = []
    if path.exists():
        for line in path.read_text().splitlines()[-3000:]:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append({k: v for k, v in row.items() if type(v) in (int, float)})
    return rows


def train_status(cfg):
    records = reward_records()
    process = TRAIN['process']
    return {'running': train_running(), 'exit_code': None if process is None else process.poll(), 'started': TRAIN['started'],
            'command': TRAIN['command'], 'log': tail(TRAIN_LOG), 'curve': reward_curve(records), 'metrics': metrics_file(cfg),
            'rollouts': [{k: r[k] for k in ('ts', 'step', 'source', 'task', 'answers', 'items', 'stats')} for r in records[-12:]][::-1]}


def status(cfg):
    spent = core.ledger()
    return {'judge_model': cfg['judge']['model'], 'spent_usd': round(spent['spent_usd'], 4), 'requests': spent['requests'],
            'budget_usd': cfg['budget_usd'], 'judge_key': bool(os.environ.get(cfg['judge']['api_key_env'] or '')) or not cfg['judge']['api_key_env'],
            'pangram_key': bool(os.environ.get('PANGRAM_API_KEY')), 'train_running': train_running()}


# ---------------------------------------------------------------- Vast runs

def vast_run_dir():
    current = vast.RUNS / 'current.json'
    return vast.RUNS / json.loads(current.read_text())['run_id'] if current.exists() else None


def pid_alive(pid):
    try:
        pid = int(pid)
        # A runner this server spawned stays a zombie until reaped, and a zombie still answers kill(0).
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return False
        except ChildProcessError:
            pass                      # not our child: kill(0) below decides
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def vast_state():
    run = vast_run_dir()
    if run is None:
        return {'run': None, 'alive': False, 'judge_spent': core.ledger()['spent_usd']}
    state = json.loads((run / 'state.json').read_text())
    alive = pid_alive(state.get('pid'))
    # Every machine this run rented, alive or gone, at its hourly upper bound.
    machines = [state[k] for k in ('instance', 'agents') if state.get(k)] + state.get('destroyed', [])
    usd = sum(max(0.0, (m.get('destroyed') or m.get('vanished') or time.time()) - m['created']) / 3600 * m.get('hourly_bound', 0) for m in machines)
    gpus = [m for m in machines if m is state.get('instance') or m.get('key') == 'instance']
    hours = sum(max(0.0, (m.get('destroyed') or m.get('vanished') or time.time()) - m['created']) / 3600 for m in gpus)
    out = {'run': state, 'alive': alive, 'gpu_hours': hours, 'gpu_usd': usd,
           'judge_spent': core.ledger()['spent_usd'], 'dir': str(run),
           'local_adapters': sorted(p.name for p in (run / 'checkpoints').glob('step_*')) if (run / 'checkpoints').exists() else []}
    if not alive and state.get('status') != 'finished' and (run / 'runner.log').exists():
        out['runner_log'] = (run / 'runner.log').read_text(errors='replace')[-3000:]
    return out


def vast_events(after):
    """Events from a byte offset: complete lines only, and where to read next."""
    run = vast_run_dir()
    path = run / 'events.jsonl' if run else None
    if not path or not path.exists():
        return {'events': [], 'next': 0, 'run_id': run.name if run else None}
    with path.open('rb') as handle:
        handle.seek(after if after <= path.stat().st_size else 0)
        start = handle.tell()
        data = handle.read(2_000_000)
    data = data[:data.rfind(b'\n') + 1]
    return {'events': [json.loads(line) for line in data.splitlines() if line.strip()], 'next': start + len(data), 'run_id': run.name}


def vast_spawn(run):
    with (run / 'runner.log').open('a') as log:
        subprocess.Popen([sys.executable, '-u', str(ROOT / 'vast_core.py'), str(run)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)


def vast_start(cfg):
    vast.validate_vast(cfg['vast'])
    run = vast_run_dir()
    if run is not None:
        state = json.loads((run / 'state.json').read_text())
        if pid_alive(state.get('pid')) or (state.get('status') != 'finished' and state.get('instance')):
            raise ValueError('Есть незавершённый прогон. Продолжите его или остановите, прежде чем начинать новый.')
    run = vast.new_run(cfg)
    vast_spawn(run)
    return vast_state()


def vast_control(action, cfg):
    run = vast_run_dir()
    if run is None:
        raise ValueError('Прогонов ещё не было')
    state = json.loads((run / 'state.json').read_text())
    alive = pid_alive(state.get('pid'))
    if action == 'resume':
        vast.validate_vast(cfg['vast'])
        if state.get('status') == 'finished':
            raise ValueError('Прогон уже завершён; начните новый')
        core.atomic_json(run / 'config.json', cfg)
        if alive:
            (run / 'RESUME').touch()
        else:               # the runner died (reboot, crash): start it again on the same run
            (run / 'STOP').unlink(missing_ok=True)
            vast_spawn(run)
    elif action == 'stop':
        (run / 'STOP').write_text('requested in the UI\n')
        if not alive and state.get('status') != 'finished':
            vast_spawn(run)  # a dead runner cannot clean up; a new one sees STOP and goes straight to finish
    return vast_state()


def vast_publish(data, cfg):
    """Through the runner while a GPU box exists (it can also merge); otherwise the downloaded adapter goes up from here."""
    run = vast_run_dir()
    if run is None:
        raise ValueError('Прогонов ещё не было')
    repo = str(data.get('repo', '')).strip()
    if not re.fullmatch(r'[A-Za-z0-9][\w.-]*/[\w.-]+', repo):
        raise ValueError('Укажите репозиторий как владелец/имя')
    request = {'repo': repo, 'public': bool(data.get('public')), 'merged': bool(data.get('merged')), 'step': int(data['step']) if data.get('step') else None}
    state = json.loads((run / 'state.json').read_text())
    if pid_alive(state.get('pid')) and (state.get('instance') or {}).get('ssh_host') and state.get('bootstrapped'):
        core.atomic_json(run / 'PUBLISH.json', request)
        return {'queued': True}
    return {'job': start_job(lambda progress: vast.publish_local(run, request))['job']}


def vast_hf_resume():
    run = vast_run_dir()
    if run is None:
        raise ValueError('Прогонов ещё не было')
    core.atomic_json(run / 'config.json', core.load_config())
    (run / 'HF_RESUME').touch()
    return {'ok': True}


def vast_destroy(instance_id):
    vast.destroy(int(instance_id))
    run = vast_run_dir()
    if run is not None:
        state = json.loads((run / 'state.json').read_text())
        if any((state.get(k) or {}).get('id') == int(instance_id) for k in ('instance', 'agents')):
            (run / 'STOP').write_text('instance destroyed from the UI\n')
    return {'destroyed': int(instance_id)}


def vast_account():
    return {'instances': vast.instances(), 'credit': vast.vast('GET', '/users/current/').get('credit')}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, code, data):
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        url = urlsplit(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ('/', '/index.html'):
                raw = (ROOT / 'lab.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                return self.wfile.write(raw)
            cfg = core.load_config()
            routes = {
                '/api/status': lambda: status(cfg),
                '/api/config': lambda: cfg,
                '/api/models': openrouter_models,
                '/api/prompt_files': lambda: list(PROMPT_FILES),
                '/api/datasets': lambda: [d for d in DATASETS if (PROJECT / d).exists()],
                '/api/runs': list_runs,
                '/api/train/status': lambda: train_status(cfg),
                '/api/vast/state': vast_state,
                '/api/vast/events': lambda: vast_events(int(query.get('after', 0))),
                '/api/vast/offers': lambda: vast.search_offers(cfg['vast']['gpu']),
                '/api/vast/host_offers': lambda: vast.search_host_offers(cfg['vast']['agents']),
                '/api/vast/account': vast_account,
                '/api/blind/next': lambda: blind.next_pair(query.get('mode', 'human')),
                '/api/blind/summary': blind.summary,
            }
            if url.path == '/api/job':
                return self.send_json(200, JOBS[query['id']]) if query.get('id') in JOBS else self.send_json(404, {'error': 'Unknown job'})
            if url.path == '/api/prompt_file':
                relative = PROMPT_FILES[query['name']]
                text = core.DEFAULT_SYSTEM if relative is None else (PROJECT / relative).read_text()
                if relative is None:
                    patch = {k: core.DEFAULT_CONFIG[k] for k in ('judge', 'reward', 'calibration')}
                else:
                    patch = core.pangram_preset(text) if relative.startswith('pangram_opus/') else core.opus_slop_preset()
                return self.send_json(200, {'text': text, 'patch': patch})
            if url.path == '/api/run':
                name = Path(query['file']).name
                return self.send_json(200, json.loads((RUNS / name).read_text()))
            if url.path not in routes:
                return self.send_json(404, {'error': 'Not found'})
            self.send_json(200, routes[url.path]())
        except (ValueError, KeyError, OSError) as exc:
            self.send_json(400, {'error': str(exc)})
        except RuntimeError as exc:
            self.send_json(502, {'error': str(exc)})

    def do_POST(self):
        # The API spends money and starts processes: refuse other origins.
        origin = self.headers.get('Origin')
        if origin and origin != f'http://{self.headers.get("Host")}':
            return self.send_json(403, {'error': 'Invalid origin'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= 5_000_000:
                raise ValueError('Request too large')
            data = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(data, dict):
                raise ValueError('JSON object required')
            path = urlsplit(self.path).path
            if path == '/api/config':
                vast.validate_vast(core.merged(core.DEFAULT_CONFIG, data)['vast'])
                return self.send_json(200, core.save_config(data))
            cfg = core.load_config()
            if path == '/api/reward':
                return self.send_json(200, score_request(data, cfg))
            if path == '/api/score':
                return self.send_json(200, start_job(lambda progress: score_request({**data, 'source': 'manual'}, cfg)))
            if path == '/api/dataset_run':
                return self.send_json(200, start_job(lambda progress: dataset_run(data, cfg, progress)))
            if path == '/api/generate':
                return self.send_json(200, start_job(lambda progress: generate_run(data, cfg, progress)))
            if path == '/api/train/start':
                return self.send_json(200, train_start(cfg, self.server.server_port))
            if path == '/api/vast/start':
                return self.send_json(200, vast_start(cfg))
            if path in ('/api/vast/resume', '/api/vast/stop'):
                return self.send_json(200, vast_control(path.rsplit('/', 1)[1], cfg))
            if path == '/api/vast/publish':
                return self.send_json(200, vast_publish(data, cfg))
            if path == '/api/vast/hf_resume':
                return self.send_json(200, vast_hf_resume())
            if path == '/api/vast/destroy':
                return self.send_json(200, vast_destroy(data['id']))
            if path == '/api/blind/vote':
                return self.send_json(200, blind.vote(str(data['id']), str(data['choice'])))
            if path == '/api/train/stop':
                return self.send_json(200, train_stop())
            self.send_json(404, {'error': 'Not found'})
        except (ValueError, KeyError, TypeError) as exc:
            self.send_json(400, {'error': str(exc)})
        except (core.JudgeError, RuntimeError) as exc:
            self.send_json(502, {'error': str(exc)})

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    core.load_env()
    port = int(os.environ.get('LAB_PORT', '8770'))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Antislop lab: http://127.0.0.1:{port}', flush=True)
    server.serve_forever()
