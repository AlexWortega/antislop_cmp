"""Calibration set for the task judge: texts whose "assignment done / not done" label is known by construction.

For each writing task it makes texts that do the job (two models, two styles) and texts that break it in a
known way: an answer to another task, a text that stays near the topic but never answers, the wrong language,
assistant chatter around the text, a refusal; for rewrites, dropped facts and invented claims.
The labels of LLM-made violations are only as good as the generator followed the instruction.
Output: task_calib.jsonl with task, text, ok, kind, task_kind, lang.
Run: python3 tasks_v2/build_task_calib.py   (needs OPENROUTER_API_KEY; generations are cached in the lab cache)
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'judge_playground'))
import lab_core as core  # noqa: E402

PER_KIND, REWRITES, BUDGET = 5, 12, 20.0
PLAIN = {'name': 'plain', 'base_url': 'https://openrouter.ai/api/v1', 'model': 'google/gemini-3.8-flash', 'api_key_env': 'OPENROUTER_API_KEY'}
HUMAN = {**PLAIN, 'name': 'human', 'model': 'deepseek/deepseek-v4.1-flash'}
ONLY = '\n\nOutput only the text itself.'
VARIANTS = {
    'good_human': '{p}\n\nWrite it the way a real person types in a hurry: uneven sentences, a blunt opinion, small imperfections. Still do exactly what is asked, in the language the assignment asks for.' + ONLY,
    'evasive': 'Below is a writing assignment. Write a text of the same length, genre and language that stays in the same general subject area but never answers the question and never does what was asked: drift to neighbouring matters instead. It must read as a normal fluent text and must not mention that it avoids anything.' + ONLY + '\n\nAssignment:\n{p}',
    'wrong_language': '{p}\n\nIMPORTANT: write the whole text in {other}, not in the language of the assignment.' + ONLY,
    'dropped_facts': '{p}\n\nIMPORTANT: leave out about half of the facts and details of the draft.' + ONLY,
    'added_claims': '{p}\n\nIMPORTANT: add three new specific claims (numbers, names, events) that are not in the draft, woven in naturally.' + ONLY,
}
WRAPPERS = {
    'en': [('Sure! Here is the text you asked for:\n\n', '\n\nLet me know if you would like me to adjust the tone or length.'),
           ('I have written the text and saved it to /app/answer.txt. Here it is:\n\n', '\n\nHope this helps!')],
    'ru': [('Конечно! Вот текст, который вы просили:\n\n', '\n\nЕсли нужно, могу изменить тон или длину.'),
           ('Я написал текст и сохранил его в /app/answer.txt. Вот он:\n\n', '\n\nНадеюсь, это поможет!')],
}
REFUSALS = {
    'en': ['I need a bit more context before I can write this. Who is the audience, and what tone do you want? Once you tell me, I will draft it right away.',
           'I am not able to help with this request as written. Could you rephrase it or tell me more about what you need?'],
    'ru': ['Чтобы написать этот текст, мне нужно больше контекста. Кто аудитория и какой тон нужен? Как только уточните, сразу подготовлю черновик.',
           'К сожалению, я не могу выполнить этот запрос в таком виде. Переформулируйте его, пожалуйста, или расскажите подробнее, что нужно.'],
}


def ask(endpoint, prompt):
    """One generation; a reply cut off by the token limit is retried with a larger limit and dropped if still cut."""
    for limit in (3000, 12000):
        body = {'model': endpoint['model'], 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': limit,
                'temperature': 1.0, 'usage': {'include': True}}
        data, _ = core.chat(endpoint, body, BUDGET, repeat='task-calib')
        choice = data['choices'][0]
        if choice.get('finish_reason') == 'stop':
            return (choice['message'].get('content') or '').strip()
    return ''


def main():
    core.load_env()
    rows = [json.loads(line) for line in (ROOT / 'rows.jsonl').read_text().splitlines()]
    rows.sort(key=lambda r: (r['split'] != 'heldout', r['task']))
    tasks, taken = [], {}
    for row in rows:
        key = (row['lang'], row['kind'])
        limit = REWRITES if row['kind'] == 'rewrite' else PER_KIND
        if taken.get(key, 0) < limit:
            taken[key] = taken.get(key, 0) + 1
            tasks.append(row)

    jobs = []   # (task index, label kind, endpoint, prompt)
    for index, row in enumerate(tasks):
        jobs.append((index, 'good_plain', PLAIN, row['prompt'] + ONLY))
        if row['kind'] == 'rewrite':
            kinds = ('good_human', 'dropped_facts', 'added_claims')
        else:
            kinds = ('good_human', 'evasive', 'wrong_language')
        for kind in kinds:
            prompt = VARIANTS[kind].replace('{p}', row['prompt']).replace('{other}', 'Russian' if row['lang'] == 'en' else 'English')
            jobs.append((index, kind, HUMAN if kind == 'good_human' else PLAIN, prompt))
    with ThreadPoolExecutor(max_workers=8) as pool:
        texts = list(pool.map(lambda job: ask(job[2], job[3]), jobs))

    out, plain = [], {}

    def add(row, kind, text):
        if text.strip():
            out.append({'task': row['prompt'], 'text': text.strip(), 'ok': kind.startswith('good'), 'kind': kind,
                        'task_kind': row['kind'], 'lang': row['lang']})

    for (index, kind, _, _), text in zip(jobs, texts):
        add(tasks[index], kind, text)
        if kind == 'good_plain':
            plain[index] = text
    for index, row in enumerate(tasks):
        if row['kind'] == 'rewrite' or not plain.get(index):
            continue
        # A good answer to the next task of the same language and genre is fluent, on-genre and off-topic here.
        peers = [i for i, other in enumerate(tasks) if (other['lang'], other['kind']) == (row['lang'], row['kind']) and plain.get(i)]
        other = peers[(peers.index(index) + 1) % len(peers)]
        if other != index:
            add(row, 'off_topic', plain[other])
        before, after = WRAPPERS[row['lang']][index % 2]
        add(row, 'chatter', before + plain[index] + after)
        if index % 4 == 0:
            add(row, 'refusal', REFUSALS[row['lang']][index // 4 % 2])
    (ROOT / 'task_calib.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    counts = {}
    for r in out:
        counts[r['kind']] = counts.get(r['kind'], 0) + 1
    print(json.dumps({'tasks': len(tasks), 'texts': len(out), 'by_kind': counts, 'spent_usd': round(core.ledger()['spent_usd'], 3)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
