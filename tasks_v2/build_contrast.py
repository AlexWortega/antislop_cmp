"""Calibration set for the judge: real human answers against model answers, plus hack pairs.

Output (format of lab_core.load_labeled, slop=True means "machine"):
  contrast.jsonl        {text, slop, task, source, split}
    human-*             160 real answers from cv/sep/antislop/data/human_qa.jsonl (ELI5, Yandex Q)
    gen-opus, gen-flash the same 160 questions answered by strong API models with the same task wording
    policy-plain/-named answers of the run 1111-235058 policy at steps >= 50, without and with a made-up name
  contrast_pairs.jsonl  {task, original, clean, split}: a policy answer with a made-up name or personal story,
                        and the same answer with only that removed (one Opus edit). The judge must not prefer original.
The name regex below only picks examples for the set; it is not part of any reward.
Run: python3 tasks_v2/build_contrast.py   (calls are cached by lab_core.chat, a rerun costs nothing)
"""
import hashlib
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'judge_playground'))
sys.path.insert(0, str(ROOT))
import lab_core as core  # noqa: E402
from build_tasks import KINDS  # noqa: E402

SEED = 20261112
GENERATORS = {'gen-opus': 'anthropic/claude-opus-5.5', 'gen-flash': 'google/gemini-3.8-flash'}
POLICY_PER_SOURCE, PAIRS = 100, 60
NAMED = re.compile(r"^\W*(?i:hi|hello|hey|dear|привет|здравствуй(?:те)?)[\s,]+([A-ZА-ЯЁ][a-zа-яё]+)\b")
CLEAN_SYSTEM = ('You edit one text. Remove only what the writer made up about people and their own life: a name for the '
                'recipient or the writer that the assignment does not give, and personal stories or biography presented as '
                'fact ("someone asked me this yesterday", "I lived there for a year"). Replace a named greeting with the same '
                'greeting without the name, drop a made-up signature name. Change nothing else, keep every other word. '
                'Return only the edited text.')


def endpoint(model):
    return {'base_url': 'https://openrouter.ai/api/v1', 'model': model, 'api_key_env': 'OPENROUTER_API_KEY'}


def words_target(text):
    return min(400, max(100, round(len(text.split()) / 50) * 50))


def split_of(key):
    return 'heldout' if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 5 == 0 else 'train'


def made_up_name(text, task):
    m = NAMED.match(text)
    return bool(m and m.group(1).lower() not in ('there', 'friend') and m.group(1).lower()[:4] not in task.lower())


def main():
    core.load_env()
    budget = core.load_config()['budget_usd']
    rng = random.Random(SEED)
    rows = []
    humans = [json.loads(line) for line in (PROJECT / 'cv/sep/antislop/data/human_qa.jsonl').read_text().splitlines()]
    jobs = []
    for index, h in enumerate(humans):
        task = KINDS[h['lang']]['answer'].format(n=words_target(h['human_text']), q=h['question'].strip())
        rows.append({'text': h['human_text'].strip(), 'slop': False, 'task': task, 'source': 'human-' + h['source'],
                     'split': h['split'], 'lang': h['lang']})
        name = sorted(GENERATORS)[index % len(GENERATORS)]
        jobs.append((name, task, h['split'], h['lang']))

    def answer(job):
        name, task, split, lang = job
        texts, cost = core.generate(endpoint(GENERATORS[name]), '', task, 1, 0.8, 2000, budget)
        return {'text': texts[0].strip(), 'slop': True, 'task': task, 'source': name, 'split': split, 'lang': lang}, cost

    with ThreadPoolExecutor(max_workers=8) as pool:
        made = list(pool.map(answer, jobs))
    rows += [r for r, _ in made if r['text']]
    cost = sum(c for _, c in made)

    policy = {'policy-plain': [], 'policy-named': []}
    for line in (PROJECT / 'judge_playground/lab_state/rewards.jsonl').read_text().splitlines():
        record = json.loads(line)
        if record.get('source') != 'train' or (record.get('step') or 0) < 50:
            continue
        for text, item in zip(record['answers'], record['items']):
            text = text.strip()
            if text and not item.get('empty'):
                kind = 'policy-named' if made_up_name(text, record['task']) else 'policy-plain'
                policy[kind].append({'text': text, 'slop': True, 'task': record['task'], 'source': kind,
                                     'split': split_of(record['task']), 'lang': 'ru' if re.search('[а-яё]', record['task']) else 'en'})
    for kind, items in policy.items():
        unique = list({r['text']: r for r in items}.values())
        rng.shuffle(unique)
        policy[kind] = unique
        rows += unique[:POLICY_PER_SOURCE]

    def clean(row):
        body = {'model': GENERATORS['gen-opus'], 'max_tokens': 3000, 'usage': {'include': True},
                'messages': [{'role': 'system', 'content': CLEAN_SYSTEM},
                             {'role': 'user', 'content': f'Assignment:\n{row["task"]}\n\nText:\n{row["text"]}'}]}
        data, charged = core.chat(endpoint(GENERATORS['gen-opus']), body, budget)
        return {'task': row['task'], 'original': row['text'], 'clean': (data['choices'][0]['message'].get('content') or '').strip(),
                'split': row['split']}, charged

    with ThreadPoolExecutor(max_workers=8) as pool:
        pairs = list(pool.map(clean, policy['policy-named'][POLICY_PER_SOURCE:POLICY_PER_SOURCE + PAIRS]))
    cost += sum(c for _, c in pairs)
    pairs = [p for p, _ in pairs if p['clean'] and p['clean'] != p['original']]

    (ROOT / 'contrast.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (ROOT / 'contrast_pairs.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in pairs))
    counts = {}
    for r in rows:
        counts[(r['source'], r['split'])] = counts.get((r['source'], r['split']), 0) + 1
    print(json.dumps({'rows': len(rows), 'pairs': len(pairs), 'cost_usd': round(cost, 3),
                      'counts': {f'{s}/{p}': n for (s, p), n in sorted(counts.items())}}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
