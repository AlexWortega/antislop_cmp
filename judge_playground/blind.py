"""Blind side-by-side comparison for a person (the HAL / Prolific-style check that no judge can replace).

Two kinds of pairs, both answers to the same task, shown in random order without labels:
  human  a real human answer against a model answer (tasks_v2/contrast.jsonl, and tasks with a `reference`
         against policy answers from lab_state/rewards.jsonl)
  steps  the base model's answer to a held-out task against the last check's answer (latest run in rewards.jsonl);
         labels are step0 / stepN where N counts the checks, not train steps
Votes go to lab_state/blind.jsonl. Later these labels can train our own reward model (Overall Dialogue Impression:
a regression RM on human scores beats a zero-shot judge).
"""
import json
import random
import threading
import time
from pathlib import Path

import lab_core as core

CONTRAST = core.PROJECT / 'tasks_v2' / 'contrast.jsonl'
CHOICES = ('left', 'right', 'same', 'both_bad')
LOCK = threading.Lock()
PENDING = {}          # pair id -> labels of the two sides, kept until voted


def votes_path():
    return core.STATE / 'blind.jsonl'


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def candidates(mode):
    """All pairs of a mode as dicts: task, a, b, a_label, b_label, and judge_pick ('a' / 'b' / None) when known for free."""
    out = []
    if mode == 'human':
        rows = read_jsonl(CONTRAST)
        humans = {r['task']: r for r in rows if not r['slop']}
        for r in rows:
            if r['slop'] and r['task'] in humans:
                out.append({'task': r['task'], 'a': humans[r['task']]['text'], 'b': r['text'], 'a_label': 'human',
                            'b_label': r['source'], 'judge_pick': None})
        for rec in read_jsonl(core.STATE / 'rewards.jsonl'):
            for text, item in zip(rec.get('answers', []), rec.get('items', [])):
                if 'beat' in item and rec.get('reference'):
                    pick = 'b' if item['beat'] == 1 else 'a' if item['beat'] == 0 else None
                    out.append({'task': rec['task'], 'a': rec['reference'], 'b': text, 'a_label': 'human',
                                'b_label': f'policy-step{rec.get("step")}', 'judge_pick': pick})
    elif mode == 'steps':
        # Held-out tasks are re-run at every check (base, then every test_freq steps), so the same task has an early
        # and a late answer. Only the latest run: it starts at the last train record of step 1.
        records = read_jsonl(core.STATE / 'rewards.jsonl')
        starts = [r['ts'] for r in records if r.get('source') == 'train' and r.get('step') == 1]
        since = max(starts) - 3600 if starts else 0
        checks = {}
        for rec in records:
            if rec.get('source') == 'eval' and rec.get('ts', 0) >= since:
                checks.setdefault((rec['task'], rec.get('agent')), []).append(rec)
        for (task, _), recs in checks.items():
            recs.sort(key=lambda r: r['ts'])
            if len(recs) < 2:
                continue
            first, last = recs[0], recs[-1]
            for (t1, i1), (t2, i2) in zip(zip(first['answers'], first['items']), zip(last['answers'], last['items'])):
                if t1.strip() and t2.strip() and not i1.get('empty') and not i2.get('empty'):
                    out.append({'task': task, 'a': t1.strip(), 'b': t2.strip(), 'a_label': 'step0',
                                'b_label': f'step{len(recs) - 1}', 'judge_pick': 'b' if i2['reward'] > i1['reward'] else
                                'a' if i1['reward'] > i2['reward'] else None})
    else:
        raise ValueError('mode must be human or steps')
    return out


def next_pair(mode, seed=None):
    """A pair not voted yet, sides in random order. Labels stay on the server until the vote."""
    done = {v['id'] for v in read_jsonl(votes_path())}
    pool = [p for p in candidates(mode) if core.digest([p['a'], p['b']])[:16] not in done]
    if not pool:
        return {'done': True, 'mode': mode}
    rng = random.Random(seed)
    pair = rng.choice(pool)
    pid = core.digest([pair['a'], pair['b']])[:16]
    swap = rng.random() < 0.5
    left, right = (pair['b'], pair['a']) if swap else (pair['a'], pair['b'])
    with LOCK:
        PENDING[pid] = {'mode': mode, 'task': pair['task'], 'swap': swap, 'left_label': pair['b_label'] if swap else pair['a_label'],
                        'right_label': pair['a_label'] if swap else pair['b_label'],
                        'judge_side': None if pair['judge_pick'] is None else (
                            ('left' if pair['judge_pick'] == 'b' else 'right') if swap else ('left' if pair['judge_pick'] == 'a' else 'right'))}
    return {'id': pid, 'mode': mode, 'task': pair['task'], 'left': left, 'right': right, 'left_count': len(pool)}


def vote(pid, choice):
    if choice not in CHOICES:
        raise ValueError('choice must be one of ' + ', '.join(CHOICES))
    with LOCK:
        meta = PENDING.pop(pid, None)
    if meta is None:
        raise ValueError('Unknown or already voted pair; load the next one')
    winner = meta[choice + '_label'] if choice in ('left', 'right') else None
    record = {'ts': time.time(), 'id': pid, 'choice': choice, 'winner': winner,
              'judge_agrees': None if meta['judge_side'] is None or winner is None else meta['judge_side'] == choice,
              'later_wins': None if meta['mode'] != 'steps' or winner is None else
              winner == max(meta['left_label'], meta['right_label'], key=lambda label: int(label[4:])),
              **{k: meta[k] for k in ('mode', 'task', 'left_label', 'right_label')}}
    core.STATE.mkdir(exist_ok=True)
    with LOCK, votes_path().open('a') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return record


def summary():
    out = {}
    for v in read_jsonl(votes_path()):
        cell = out.setdefault(v['mode'], {'n': 0, 'same': 0, 'both_bad': 0, 'wins': {}, 'judge_n': 0, 'judge_agree': 0})
        cell['n'] += 1
        if v['choice'] in ('same', 'both_bad'):
            cell[v['choice']] += 1
        else:
            cell['wins'][v['winner']] = cell['wins'].get(v['winner'], 0) + 1
        if v['judge_agrees'] is not None:
            cell['judge_n'] += 1
            cell['judge_agree'] += v['judge_agrees']
    for v in read_jsonl(votes_path()):
        if v.get('later_wins') is not None:
            out[v['mode']]['later'] = out[v['mode']].get('later', 0) + v['later_wins']
    for mode, cell in out.items():
        decided = cell['n'] - cell['same'] - cell['both_bad']
        cell['decided'] = decided
        if mode == 'human':
            cell['human_win_rate'] = cell['wins'].get('human', 0) / decided if decided else None
        else:
            cell['later_win_rate'] = cell.get('later', 0) / decided if decided else None
        cell['judge_agreement'] = cell['judge_agree'] / cell['judge_n'] if cell['judge_n'] else None
    return out
