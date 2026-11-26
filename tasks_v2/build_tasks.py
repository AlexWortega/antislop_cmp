"""Writing tasks for the agent harness: 500 train + 40 held-out, RU and EN, 100-400 words.

Replaces the 8 repeated tweets. Every task is unique. Output:
  rows.jsonl            one row per task with the metadata the reward needs
  harbor/<agent>/<id>/  Harbor task dirs for pi and hermes, same layout as qwen_scaffold_longrun/tasks
  regress_qa.json       factual QA tasks kept only as a regression eval
Run: python3 tasks_v2/build_tasks.py
"""
import collections
import hashlib
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
LONGRUN = PROJECT / 'qwen_scaffold_longrun'
SEED = 20261110
TRAIN_PER_LANG, HELDOUT_PER_LANG, REWRITE_TRAIN, REWRITE_HELDOUT = 250, 20, 60, 6
LENGTHS = (100, 150, 200, 300, 400)
HARNESS = '\n\nUse tools to write only the requested text to /app/answer.txt. Do not search the web. The final chat reply may briefly confirm completion.\n'

# Each kind turns a question into a different writing job, so the model cannot learn one template.
KINDS = {
    'en': {
        'answer': 'Answer the question below the way a knowledgeable person would on a forum. Write about {n} words of plain running prose: no headings, no bullet lists, no bold.\n\nQuestion: {q}',
        'post': 'Write a personal blog post of about {n} words prompted by the question below. Take a position and explain it with concrete detail. Plain prose, no headings or lists.\n\nQuestion: {q}',
        'email': 'A friend emailed you the question below. Write your reply as an email of about {n} words: greet them, answer properly, sign off. Plain text only.\n\nQuestion: {q}',
        'note': 'Write an explainer of about {n} words for a company wiki that answers the question below for a new colleague with no background. Plain paragraphs, no lists.\n\nQuestion: {q}',
    },
    'ru': {
        'answer': 'Ответь на вопрос ниже так, как ответил бы знающий человек на форуме. Примерно {n} слов связного текста: без заголовков, списков и выделения жирным.\n\nВопрос: {q}',
        'post': 'Напиши пост для личного блога примерно на {n} слов, поводом для которого стал вопрос ниже. Займи позицию и объясни её на конкретных деталях. Обычный текст, без заголовков и списков.\n\nВопрос: {q}',
        'email': 'Друг прислал тебе письмо с вопросом ниже. Напиши ответное письмо примерно на {n} слов: поздоровайся, ответь по существу, попрощайся. Только обычный текст.\n\nВопрос: {q}',
        'note': 'Напиши объяснение примерно на {n} слов для внутренней вики компании: оно должно ответить на вопрос ниже новому коллеге без подготовки. Обычные абзацы, без списков.\n\nВопрос: {q}',
    },
}
REWRITE = 'Rewrite the draft below so it reads like a person wrote it. Keep every fact and the meaning, keep it at about {n} words, and do not add new claims. Plain prose only.\n\nDraft:\n{q}'

TEST_SH = '''#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier
if [[ -s /app/answer.txt ]] && [[ $(wc -c < /app/answer.txt) -lt 20000 ]]; then
 cp /app/answer.txt /logs/verifier/answer.txt
 printf '1\\n' > /logs/verifier/reward.txt
else
 printf '0\\n' > /logs/verifier/reward.txt
fi
'''


def task_id(text):
    return 'write_' + hashlib.sha256(text.encode()).hexdigest()[:12]


def main():
    rng = random.Random(SEED)
    questions = collections.defaultdict(list)
    for line in (PROJECT / 'cv/sep/antislop/data/distill.jsonl').read_text().splitlines():
        row = json.loads(line)
        prompt = row['prompt'].strip()
        if row['source'] in ('eli5', 'yandex-q') and row['lang'] in KINDS and 20 <= len(prompt) <= 400:
            questions[row['lang']].append(prompt)
    drafts = []
    for line in (PROJECT / 'pangram_opus/round6/train.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['slop'] is True and 80 <= row['word_count'] <= 400:
            drafts.append(row['text'].strip())

    rows = []
    for lang, pool in sorted(questions.items()):
        pool = sorted(set(pool))
        rng.shuffle(pool)
        rewrites = (REWRITE_TRAIN, REWRITE_HELDOUT) if lang == 'en' else (0, 0)
        need = TRAIN_PER_LANG + HELDOUT_PER_LANG - sum(rewrites)
        if len(pool) < need:
            raise SystemExit(f'Only {len(pool)} {lang} questions, need {need}')
        kinds = sorted(KINDS[lang])
        for index, question in enumerate(pool[:need]):
            kind, words = kinds[index % len(kinds)], rng.choice(LENGTHS)
            split = 'heldout' if index < HELDOUT_PER_LANG - rewrites[1] else 'train'
            rows.append({'kind': kind, 'lang': lang, 'split': split, 'target_words': words, 'allow_markdown': False,
                         'prompt': KINDS[lang][kind].format(n=words, q=question)})
    drafts = sorted(set(drafts))
    rng.shuffle(drafts)
    if len(drafts) < REWRITE_TRAIN + REWRITE_HELDOUT:
        raise SystemExit(f'Only {len(drafts)} drafts for rewrite tasks')
    for index, draft in enumerate(drafts[:REWRITE_TRAIN + REWRITE_HELDOUT]):
        words = len(draft.split())
        rows.append({'kind': 'rewrite', 'lang': 'en', 'split': 'heldout' if index < REWRITE_HELDOUT else 'train',
                     'target_words': words, 'allow_markdown': False, 'prompt': REWRITE.format(n=words, q=draft)})
    for row in rows:
        row['task'] = task_id(row['prompt'])
    if len({r['task'] for r in rows}) != len(rows):
        raise SystemExit('Duplicate tasks')
    rows.sort(key=lambda r: (r['split'], r['task']))

    template = (LONGRUN / 'repo/examples/hello_file/task.toml').read_text()
    harbor = ROOT / 'harbor'
    if harbor.exists():
        shutil.rmtree(harbor)
    for row in rows:
        for agent in ('pi', 'hermes'):
            path = harbor / agent / row['task']
            (path / 'environment').mkdir(parents=True)
            (path / 'tests').mkdir()
            (path / 'environment/Dockerfile').write_text(f'FROM antislop-pilot-{agent}:local\nWORKDIR /app\nCMD ["bash"]\n')
            (path / 'task.toml').write_text(template.replace('antislop/hello-file', 'antislop/' + row['task'])
                                            .replace('memory_mb = 2048', 'memory_mb = 1024').replace('timeout_sec = 180.0', 'timeout_sec = 480.0'))
            (path / 'instruction.md').write_text(row['prompt'] + HARNESS)
            (path / 'tests/test.sh').write_text(TEST_SH)
    (ROOT / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

    # Factual QA stays out of training; a fixed sample checks the agent did not get worse at it.
    qa = [json.loads(line) for line in (LONGRUN / 'source_rows.jsonl').read_text().splitlines()]
    qa = sorted(r['task'] for r in qa if r['source'] == 'traffic-841')
    rng.shuffle(qa)
    (ROOT / 'regress_qa.json').write_text(json.dumps({'tasks_dir': 'qwen_scaffold_longrun/tasks', 'tasks': sorted(qa[:24])}, indent=2) + '\n')

    report = collections.Counter((r['split'], r['lang'], r['kind']) for r in rows)
    print(json.dumps({'total': len(rows), 'by_split': collections.Counter(r['split'] for r in rows),
                      'by_lang': collections.Counter(r['lang'] for r in rows),
                      'by_kind': collections.Counter(r['kind'] for r in rows),
                      'by_length': collections.Counter(r['target_words'] for r in rows if r['kind'] != 'rewrite'),
                      'heldout': {'/'.join(k[1:]): v for k, v in sorted(report.items()) if k[0] == 'heldout'}}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
