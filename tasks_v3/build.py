"""Task set v3: real requests people sent to an LLM (WildChat-1M), plus emails and posts with a persona.
Optionally (--with-references) ELI5 questions with a real human answer as reference for the pairwise judge.

Stages (run in order; the paid ones go through lab_core.chat, so they are cached and counted in the judge ledger):
  python3 tasks_v3/build.py sources    free: download and filter human answers written before ChatGPT
  python3 tasks_v3/build.py screen     paid (~$1, Gemini Flash): drop references that are junk, links, copy-paste
  python3 tasks_v3/build.py personas   paid (~$1, Opus): 120 short personas for emails and posts
  python3 tasks_v3/build.py build      free: rows.jsonl + harbor/<agent>/<task>/ in the tasks_v2 layout

Sources (pre-ChatGPT, so the reference is human by construction):
  EN  ELI5-Category (r/explainlikeimfive 2017-2021): top answer with score >= 10
  RU  Yandex Q (its5Q/yandex-q full dump, CC0): answers created before 2022-11 with positiveVotes >= 5, net votes > 0,
      not banned / deleted / hidden / adult
Both: 60-400 words, no URLs, question not in cv/sep/antislop/data/human_qa.jsonl or tasks_v2/contrast.jsonl.
The prompt gives an upper bound ("not longer than N words", N = 1.4 x the reference) instead of "about N words",
so the model is not paid to pad; the reward's length target is the reference's own length.
"""
import collections
import gzip
import hashlib
import io
import json
import random
import re
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'judge_playground'))
sys.path.insert(0, str(PROJECT / 'tasks_v2'))
import lab_core as core  # noqa: E402
from build_tasks import HARNESS, LONGRUN, TEST_SH, task_id  # noqa: E402

SEED = 20261113
SOURCES = ROOT / 'sources'
ELI5_URL = 'https://jingshensn2.github.io/eli5c/datasets/eli5-category-train.json.gz'
YANDEX_URL = 'https://huggingface.co/datasets/its5Q/yandex-q/resolve/main/full.jsonl.gz'
PER_LANG_TRAIN, PER_LANG_HELDOUT = 200, 20
CANDIDATES_PER_LANG = 400          # screened down to PER_LANG_TRAIN + PER_LANG_HELDOUT
SCREEN_MODEL, PERSONA_MODEL = 'google/gemini-3.8-flash', 'anthropic/claude-opus-5.5'
URL = re.compile(r'https?://|www\.|_URL_\d*_', re.I)

PROMPTS = {
    'en': 'Answer the question below the way a knowledgeable person would on a forum. Plain running prose, no headings, no bullet lists, no bold. Not longer than {cap} words.\n\nQuestion: {q}',
    'ru': 'Ответь на вопрос ниже так, как ответил бы знающий человек на форуме. Обычный связный текст, без заголовков, списков и выделения жирным. Не длиннее {cap} слов.\n\nВопрос: {q}',
}
PERSONA_BLOCK = {
    'en': 'Who you are (use these details if they fit; do not invent other facts about yourself or the other person):\n{p}\n\n',
    'ru': 'Кто ты (используй эти детали, если к месту; других фактов о себе и собеседнике не придумывай):\n{p}\n\n',
}
PERSONA_PROMPT = """Write {n} short personas for a writing exercise, in {lang}. Each persona says who is writing, to whom (for emails: the friend's first name and how they know each other), the writer's situation right now and manner of writing (for example: typing on a phone on the train, tired after a night shift, likes dry jokes, blunt, chatty), plus one or two concrete true details about the writer that may come up (job, city, a hobby). 2-3 sentences each, plain, no clichés, varied ages, jobs, places and moods; nobody is a writer, coach or marketer.
Return JSON only: {{"personas": ["...", "..."]}}"""
SCREEN_PROMPT = """You check whether a text is a usable example of a real person's forum answer. Reply JSON only: {"ok": true|false, "why": "..."}.
ok = false if it is: not an answer to the question, mostly a link or a reference to elsewhere, copy-paste from an encyclopedia or a news article, an advertisement, abusive, or so garbled that it cannot be read. Informal, blunt, partial, opinionated or slightly wrong answers are ok."""


def words(text):
    return len(text.split())


def clean(text):
    return re.sub(r'[ \t]+', ' ', text.replace('\r', '')).strip()


def excluded_questions():
    out = set()
    for line in (PROJECT / 'cv/sep/antislop/data/human_qa.jsonl').read_text().splitlines():
        out.add(json.loads(line)['question'].strip().lower())
    return out


def eli5_candidates(skip):
    raw = urllib.request.urlopen(ELI5_URL, timeout=120).read()
    rows = json.load(io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(raw)), encoding='utf-8'))
    out = []
    for row in rows:
        question = clean(row['title'])
        answers = sorted(zip(row['answers']['text'], row['answers']['score']), key=lambda a: -a[1])
        if not answers or question.lower() in skip:
            continue
        text, score = answers[0]
        text = clean(text.replace(' .', '.').replace(' ,', ',').replace(" n't", "n't").replace(" 's", "'s"))
        if score >= 10 and 60 <= words(text) <= 400 and not URL.search(text):
            out.append({'lang': 'en', 'source': 'eli5', 'id': row['q_id'], 'question': question, 'reference': text, 'votes': score})
    return out


def yandex_candidates(skip):
    out = []
    with urllib.request.urlopen(YANDEX_URL, timeout=600) as response, gzip.GzipFile(fileobj=response) as handle:
        for line in io.TextIOWrapper(handle, encoding='utf-8'):
            row = json.loads(line)
            question = clean((row.get('title') or '') + ('\n' + row['plainText'] if row.get('plainText') else ''))
            if row.get('adultContent') or not row.get('answers') or row.get('title', '').strip().lower() in skip:
                continue
            best = None
            for a in row['answers']:
                text = clean(a.get('plainText') or '')
                if (a.get('banned') or a.get('deleted') or a.get('hidden') or a.get('adultContent')
                        or (a.get('created') or '9') >= '2022-11' or URL.search(text) or not 60 <= words(text) <= 400):
                    continue
                net = (a.get('positiveVotes') or 0) - (a.get('negativeVotes') or 0)
                if (a.get('positiveVotes') or 0) >= 5 and net > 0 and (best is None or net > best[0]):
                    best = (net, text)
            if best and 10 <= len(question) <= 600:
                out.append({'lang': 'ru', 'source': 'yandex-q', 'id': row['id'], 'question': question, 'reference': best[1], 'votes': best[0]})
    return out


def sources():
    SOURCES.mkdir(parents=True, exist_ok=True)
    skip = excluded_questions()
    rng = random.Random(SEED)
    for name, fn in (('eli5', eli5_candidates), ('yandex', yandex_candidates)):
        rows = fn(skip)
        rows.sort(key=lambda r: r['id'])
        rng.shuffle(rows)
        (SOURCES / f'{name}_candidates.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows[:CANDIDATES_PER_LANG]))
        print(name, 'passed filters:', len(rows), 'kept', min(len(rows), CANDIDATES_PER_LANG), flush=True)


def candidates():
    return [json.loads(l) for name in ('eli5', 'yandex') for l in (SOURCES / f'{name}_candidates.jsonl').read_text().splitlines()]


def endpoint(model):
    return {'base_url': 'https://openrouter.ai/api/v1', 'model': model, 'api_key_env': 'OPENROUTER_API_KEY'}


def ask_json(model, system, user, budget, max_tokens=800):
    body = {'model': model, 'max_tokens': max_tokens, 'usage': {'include': True},
            'messages': ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]}
    data, cost = core.chat(endpoint(model), body, budget)
    content = data['choices'][0]['message'].get('content') or ''
    return json.loads(content[content.find('{'):content.rfind('}') + 1]), cost


def screen():
    core.load_env()
    budget = core.load_config()['budget_usd']
    rows = candidates()

    def one(row):
        try:
            verdict, cost = ask_json(SCREEN_MODEL, SCREEN_PROMPT, f'Question: {row["question"]}\n\nAnswer:\n{row["reference"]}', budget)
            return {'id': row['id'], 'ok': bool(verdict.get('ok')), 'why': str(verdict.get('why', ''))}, cost
        except (ValueError, KeyError) as exc:
            return {'id': row['id'], 'ok': False, 'why': 'unparsed: ' + str(exc)}, 0.0
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, rows))
    (SOURCES / 'screen.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r, _ in results))
    print('screened', len(results), 'ok', sum(r['ok'] for r, _ in results), f'cost ${sum(c for _, c in results):.2f}')


WILD_SCREEN_PROMPT = """You sort real requests that people sent to a chat assistant. Decide if the request is a good exercise for a model that must answer in natural, human-sounding prose. Reply JSON only:
{"ok": true|false, "kind": "question"|"advice"|"explain"|"email"|"post"|"story"|"other", "words": <int>, "markdown_ok": true|false, "why": "..."}
ok = false for: code or anything technical that needs code, math or chemistry problems, test or homework items (fill the blank, multiple choice), requests that need live web data or files, prompts for image generators, translation, requests that only make sense mid-conversation, sexual or hateful content, and requests for very long texts (over ~700 words).
ok = true for questions, explanations, advice, and writing requests (emails, posts, letters, short stories, speeches), even when they are casual, vague or oddly phrased.
words: how long a good answer by a knowledgeable person would be (40 to 600).
markdown_ok: true only if a person would naturally use a list or headings for this request (a plan, a recipe, a step-by-step guide)."""
WILD_PER_LANG = 900
# Flash marks lists as natural for ~70% of requests; prose is the point here, so a list is allowed only when the
# request itself asks for a plan, list, steps or recipe.
ASKS_FOR_LIST = re.compile(r'\b(plan|list|steps?|step-by-step|recipe|schedule|checklist|outline|table)\b|план|спис|шаг|рецепт|расписан|чек-лист|таблиц|перечисл', re.I)
WILD_HELDOUT = 30          # per language
WITH_REFERENCES = '--with-references' in sys.argv   # ELI5 tasks with a human reference answer (beat the human)


def screen_wild():
    core.load_env()
    budget = core.load_config()['budget_usd']
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in (SOURCES / 'wildchat_candidates.jsonl').read_text().split('\n') if l.strip()]
    sample = []
    for lang in ('ru', 'en'):
        pool = [r for r in rows if r['lang'] == lang]
        rng.shuffle(pool)
        sample += pool[:WILD_PER_LANG]

    def one(row):
        try:
            verdict, cost = ask_json(SCREEN_MODEL, WILD_SCREEN_PROMPT, row['prompt'], budget)
            words_ = int(verdict.get('words') or 0)
            return {**row, 'ok': bool(verdict.get('ok')) and 40 <= words_ <= 600, 'kind': str(verdict.get('kind', 'other')),
                    'words': words_, 'markdown_ok': bool(verdict.get('markdown_ok')), 'why': str(verdict.get('why', ''))}, cost
        except (ValueError, KeyError, TypeError) as exc:
            return {**row, 'ok': False, 'why': 'unparsed: ' + str(exc)}, 0.0
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, sample))
    (SOURCES / 'wildchat_screened.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r, _ in results))
    by = collections.Counter((r['lang'], r.get('kind'), r['ok']) for r, _ in results)
    print('cost $%.2f' % sum(c for _, c in results))
    for k, n in sorted(by.items(), key=str):
        print(k, n)


def personas():
    core.load_env()
    budget = core.load_config()['budget_usd']
    out, cost = [], 0.0
    for lang, name in (('en', 'English'), ('ru', 'Russian')):
        for k in range(3):
            data, charged = ask_json(PERSONA_MODEL, '', PERSONA_PROMPT.format(n=20, lang=name) + f'\n(batch {k + 1})', budget, 6000)
            cost += charged
            out += [{'lang': lang, 'persona': clean(p)} for p in data['personas'] if isinstance(p, str) and p.strip()]
    (ROOT / 'personas.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in out))
    print('personas', len(out), f'cost ${cost:.2f}')


def build():
    rng = random.Random(SEED)
    ok = None
    if (SOURCES / 'screen.jsonl').exists():
        ok = {r['id'] for r in map(json.loads, (SOURCES / 'screen.jsonl').read_text().splitlines()) if r['ok']}
    rows = []
    wild = [json.loads(l) for l in (SOURCES / 'wildchat_screened.jsonl').read_text().split('\n') if l.strip()]
    for lang in ('en', 'ru'):
        pool = sorted((r for r in wild if r['lang'] == lang and r['ok']), key=lambda r: r['id'])
        rng.shuffle(pool)
        for index, req in enumerate(pool):
            rows.append({'kind': 'wild_' + req['kind'], 'lang': lang, 'split': 'heldout' if index < WILD_HELDOUT else 'train',
                         'target_words': req['words'], 'length_tolerance': 0.5,
                         'allow_markdown': req['markdown_ok'] and bool(ASKS_FOR_LIST.search(req['prompt'])),
                         'prompt': req['prompt'], 'source': 'wildchat:' + req['id']})
    if WITH_REFERENCES:
        for lang in ('en', 'ru'):
            pool = [r for r in candidates() if r['lang'] == lang and (ok is None or r['id'] in ok)]
            for index, ref in enumerate(pool[:PER_LANG_TRAIN + PER_LANG_HELDOUT]):
                ref_words = words(ref['reference'])
                cap = max(80, round(ref_words * 1.4 / 10) * 10)
                rows.append({'kind': 'ref_answer', 'lang': lang, 'split': 'heldout' if index < PER_LANG_HELDOUT else 'train',
                             'target_words': ref_words, 'allow_markdown': False, 'prompt': PROMPTS[lang].format(cap=cap, q=ref['question']),
                             'reference': ref['reference'], 'reference_source': f'{ref["source"]}:{ref["id"]}', 'reference_votes': ref['votes']})
    people = {}
    if (ROOT / 'personas.jsonl').exists():
        for p in map(json.loads, (ROOT / 'personas.jsonl').read_text().splitlines()):
            people.setdefault(p['lang'], []).append(p['persona'])
    for old in map(json.loads, (PROJECT / 'tasks_v2/rows.jsonl').read_text().splitlines()):
        if old['kind'] not in ('email', 'post'):
            continue
        row = {k: v for k, v in old.items() if k != 'task'}
        if people.get(old['lang']):
            row['persona'] = rng.choice(people[old['lang']])
            row['prompt'] = PERSONA_BLOCK[old['lang']].format(p=row['persona']) + old['prompt']
        rows.append(row)
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
    counts = {}
    for r in rows:
        counts[f'{r["kind"]}/{r["lang"]}/{r["split"]}'] = counts.get(f'{r["kind"]}/{r["lang"]}/{r["split"]}', 0) + 1
    print(json.dumps({'rows': len(rows), 'screened': ok is not None, 'personas': sum(map(len, people.values())), 'counts': counts}, indent=1))


if __name__ == '__main__':
    {'sources': sources, 'screen': screen, 'screen_wild': screen_wild, 'personas': personas, 'build': build}[sys.argv[1]]()
