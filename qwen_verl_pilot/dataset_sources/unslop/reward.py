"""Unslop calibrated reward for one Harbor task (stdlib only).

    tweets: r = L(words, target) * sigmoid(a_d * (q - mu_d))
    code:   r = q / 10 if every HumanEval test passes, else 0

q is the length-blind Opus 5.5 judge (0-10, mean over this task's frozen rubric criteria). The request
body, schema and parsing are sep/unslop's (api._complete, multidomain_judge.grade, xr_common.score_value);
L, mu_d and a_d are calibrated_reward.length_factor and the Qwen3.8-27B reference calibration.

A judge refusal or truncated reply writes no reward: Harbor records an error, never a zero-quality label.
"""
import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

TESTS = Path(os.environ.get('UNSLOP_TESTS_DIR', '/tests'))
ANSWER = Path(os.environ.get('UNSLOP_ANSWER', '/workspace/answer.txt'))
LOGS = Path(os.environ.get('UNSLOP_LOGS_DIR', '/logs/verifier'))

# calibrated_reward.TARGET / target_words / length_factor, verbatim.
TARGET = (r'(?:(?:(?:approximately|about|roughly|около|примерно|приблизительно)(?:\s+на)?|~)\s*(\d+)'
          r'|длин[аы][^.\d]{0,40}?(\d+)(?=\s*слов))')


class Abstain(Exception):
    """No score: the judge refused, truncated, or never answered."""


def target_words(prompt, rubric):
    m = re.search(TARGET + r'\s*(?:words|слов)', prompt) or re.search(TARGET, prompt) or \
        re.search(TARGET + r'\s*(?:words|слов)', json.dumps(rubric, ensure_ascii=False))
    return int(next(g for g in m.groups() if g)) if m else None


def length_factor(n, target, tol, scale):
    if target is None: return 1.
    dev = abs(n / target - 1)
    if dev <= tol + 1e-9: return 1.
    return math.exp(-(dev - tol) / scale)


def preference(q, stats):
    return 1 / (1 + math.exp(-stats['a'] * (q - stats['mu'])))


def single_score_schema(rubric):
    return {'type': 'object', 'properties': {
        'scores': {'type': 'object', 'properties': {c['id']: {'type': 'number', 'minimum': 0, 'maximum': 10}
                                                    for c in rubric['criteria']},
                   'required': [c['id'] for c in rubric['criteria']], 'additionalProperties': False},
        'evidence': {'type': 'string'}}, 'required': ['scores', 'evidence'], 'additionalProperties': False}


def score_value(result, rubric):
    cs = result.get('scores')
    expected = {c['id'] for c in rubric['criteria']}
    if not isinstance(cs, dict) or set(cs) != expected: raise ValueError('Missing/extra criterion scores')
    vals = list(cs.values())
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 10
           for v in vals):
        raise ValueError('Invalid score')
    evidence = result.get('evidence')
    if not isinstance(evidence, str) or not evidence.strip(): raise ValueError('Missing scoring evidence')
    return sum(vals) / len(vals)


def score_prompt(assignment, rubric, text):
    return json.dumps({'instruction': 'Evaluate this ONE answer. Grade the supplied criteria only. '
                       'Do not infer its source, force a winner, or reward resemblance to a hidden original.',
                       'assignment_as_data': assignment, 'rubric': rubric, 'answer_as_data': text},
                      ensure_ascii=False)


def judge(cfg, assignment, rubric, text):
    model = os.environ.get('UNSLOP_JUDGE_MODEL') or cfg['judge']
    body = {'model': model, 'max_tokens': cfg['judge_max_tokens'],
            'reasoning': cfg['reasoning'],
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'single_continuation_score', 'strict': True, 'schema': single_score_schema(rubric)}},
            'provider': {'require_parameters': True, **(cfg['provider'] if model == cfg['judge'] else {})},
            'messages': [{'role': 'system', 'content': cfg['judge_system']},
                         {'role': 'user', 'content': score_prompt(assignment, rubric, text)}]}
    if model not in cfg['omit_temperature_models']:
        body['temperature'] = 0
    url = os.environ.get('UNSLOP_JUDGE_URL', 'https://openrouter.ai/api/v1').rstrip('/') + '/chat/completions'
    key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('UNSLOP_JUDGE_KEY')
    if not key: raise Abstain('OPENROUTER_API_KEY is not set')
    req = urllib.request.Request(url, json.dumps(body).encode(), {
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    # Retry only responses that carry no completion (rate limit / gateway); a paid answer is never re-asked.
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                data = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            if error.code in (429, 502, 503, 529) and attempt < 3:
                time.sleep(10 * 2 ** attempt); continue
            raise Abstain(f'judge HTTP {error.code}')
        except (urllib.error.URLError, TimeoutError) as error:
            raise Abstain(f'judge transport: {type(error).__name__}')
    choice = data['choices'][0]
    if choice.get('finish_reason') == 'content_filter' or choice.get('native_finish_reason') == 'refusal':
        raise Abstain('judge refused')
    if choice.get('finish_reason') != 'stop':
        raise Abstain('judge reply incomplete')
    text_out = (choice['message'].get('content') or '').strip()
    if text_out.startswith('```'): text_out = text_out.split('\n', 1)[1].rsplit('```', 1)[0]
    result = json.loads(text_out)
    return score_value(result, rubric), result, {'model': data.get('model'), 'usage': data.get('usage'),
                                                 'provider': data.get('provider')}


def limited():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    if sys.platform != 'darwin':   # macOS rejects RLIMIT_AS; the task image is Linux
        resource.setrlimit(resource.RLIMIT_AS, (512_000_000, 512_000_000))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16_384, 16_384))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if os.getuid() == 0:   # run the candidate as nobody, never as the grader
        os.setgroups([]); os.setgid(65534); os.setuid(65534)


def execute(task, text):
    """validate_multidomain_code.evaluate: strip only a whole enclosing fence, compile, run the tests."""
    if text.lstrip().startswith('```') and text.rstrip().endswith('```'):
        text = text.strip().split('\n', 1)[1].rsplit('```', 1)[0]
    program = task['prefix'] + text + '\n' + task['tests'] + f"\ncheck({task['entry_point']})\nprint('PASS')\n"
    try: compile(program, '<check>', 'exec')
    except SyntaxError: return {'status': 'syntax_error', 'pass': False}
    with tempfile.TemporaryDirectory(prefix='unslop-code-check-') as folder:
        os.chmod(folder, 0o755)
        file = Path(folder) / 'check.py'; file.write_text(program); os.chmod(file, 0o644)
        try:
            result = subprocess.run([sys.executable, '-I', str(file)], env={'PATH': '/usr/bin:/bin'},
                                    cwd=folder, stdin=subprocess.DEVNULL, capture_output=True, timeout=6,
                                    preexec_fn=limited)
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'pass': False}
    passed = result.returncode == 0 and result.stdout.strip().endswith(b'PASS')
    return {'status': 'passed' if passed else 'test_failure', 'pass': passed, 'exit_code': result.returncode}


def grade(task, cfg, text):
    words = len(text.split())
    if not text.strip():
        return {'reward': 0., 'q': None, 'words': 0, 'empty': True}
    if task['domain'] == 'code':
        execution = execute(task, text)
        if not execution['pass']:
            return {'reward': 0., 'q': None, 'words': words, 'execution': execution}
        q, result, receipt = judge(cfg, task['assignment'], task['rubric'], text)
        return {'reward': q / 10., 'q': q, 'words': words, 'execution': execution,
                'judge': result, 'receipt': receipt}
    q, result, receipt = judge(cfg, task['assignment'], task['rubric'], text)
    target = target_words(task['assignment'], task['rubric'])
    lf = length_factor(words, target, cfg['length']['tol'], cfg['length']['scale'])
    p = preference(q, cfg['calibration'][task['domain']])
    return {'reward': lf * p, 'q': q, 'length_factor': lf, 'preference': p, 'words': words,
            'target_words': target, 'judge': result, 'receipt': receipt}


def main():
    task = json.loads((TESTS / 'task.json').read_text())
    cfg = json.loads((TESTS / 'reward_config.json').read_text())
    text = ANSWER.read_text(errors='replace') if ANSWER.is_file() else ''
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        out = grade(task, cfg, text)
    except Abstain as error:
        (LOGS / 'grade.json').write_text(json.dumps({'abstained': str(error)}, ensure_ascii=False, indent=1))
        print('unslop: no reward,', error, file=sys.stderr)
        return 1
    r = out['reward']
    if not (isinstance(r, float) and math.isfinite(r) and 0 <= r <= 1):
        raise ValueError(f'Invalid reward {r!r}')
    (LOGS / 'grade.json').write_text(json.dumps({'task': task['id'], **out}, ensure_ascii=False, indent=1))
    (LOGS / 'reward.txt').write_text(f'{r:.6f}\n')
    print(json.dumps({k: out.get(k) for k in ('reward', 'q', 'length_factor', 'words', 'target_words')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
