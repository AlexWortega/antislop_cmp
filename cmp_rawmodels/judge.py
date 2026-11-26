"""Blind SBS for live Rawmodels Antislop against two previously frozen writing arms."""
import argparse
import hashlib
import html
import json
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'cmp_hemmingway/out'
CODE = ROOT.parent / 'cmp_hemmingway/source/sep/unslop'
sys.path.insert(0, str(CODE))
from api import API, CompletionRefused
from compare_sbs import boot, recover_fenced_receipt, sbs_system, validate
from xr_common import write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    rows = [json.loads(s) for s in (SOURCE / 'sbs/examples.jsonl').read_text().splitlines()]
    raw = json.loads((ROOT / 'default_generations.json').read_text())
    assert raw['source_input_sha256'] == sha(SOURCE / 'sbs/examples.jsonl')
    models = {'rawmodels': {g['id']: g for g in raw['generations']}}
    for opponent in ('rft1', 'hemmingway'):
        record = json.loads((SOURCE / (opponent + '_generations.json')).read_text())
        assert record['input_sha256'] == sha(SOURCE / 'examples.jsonl')
        assert record['enable_thinking'] is False
        models[opponent] = {g['id']: g for g in record['generations']}
    assert len(rows) == 24 and all(r['domain'] != 'code' for r in rows)
    assert all(models[a][r['id']]['finish_reason'] == 'stop' for a in models for r in rows)
    cfg = json.loads((SOURCE / 'config.json').read_text())
    system = sbs_system(json.loads((SOURCE / 'best_candidate.json').read_text())['judge_system'])
    protocol = {'format': 'Writing-only blind joint SBS A/B/tie, both orders',
        'source_commit': 'da1013e94dfeb83f755e225471f6f49385507c8b',
        'judge': cfg['judge'], 'judge_system': system,
        'judge_candidate_sha256': sha(SOURCE / 'best_candidate.json'),
        'tasks_sha256': sha(SOURCE / 'sbs/examples.jsonl'), 'n_tasks': 24,
        'opponents': ['rft1', 'hemmingway'], 'n_judgments': 96,
        'rawmodels_generation': raw['generation'], 'rawmodels_checkpoint_revision': None,
        'rawmodels_checkpoint_note': raw['checkpoint_note'],
        'opponent_manifest': json.loads((SOURCE / 'model_manifest.json').read_text()),
        'generation_sha256': {'rawmodels': sha(ROOT / 'default_generations.json'),
            **{a: sha(SOURCE/(a+'_generations.json')) for a in ('rft1','hemmingway')}},
        'limitation': 'Live API with default thinking and 8192 token cap versus local FP8 models without thinking and 2048 cap; checkpoint and deployment internals of live API are not exposed. Same submitted temperature/top_p/top_k/seed. This compares service configurations, not isolated weight quality.'}
    target = ROOT / 'sbs_protocol.json'
    if target.exists():
        assert json.loads(target.read_text()) == protocol
    else:
        write_json(target, protocol)
    return rows, models, cfg, system


def judge():
    rows, models, cfg, system = inputs()
    keyfile = Path('/Users/alexwortega/Desktop/side_projects/rawmodels/humanizer/.local/secrets.json')
    os.environ['OPENROUTER_API_KEY'] = json.loads(keyfile.read_text())['OPENROUTER_API_KEY']
    api = API(SOURCE / 'api', cfg)
    jobs = [(row, opponent, order) for row in rows for opponent in ('rft1','hemmingway') for order in (0,1)]
    random.Random(271828).shuffle(jobs)
    def one(job):
        row, opponent, order = job
        dest = ROOT / 'verdicts' / opponent / (row['id'] + '-' + str(order) + '.json')
        if dest.exists():
            previous = json.loads(dest.read_text())
            if previous['status'] == 'judged': validate(previous['result'])
            return
        arms = ('rawmodels', opponent) if order == 0 else (opponent, 'rawmodels')
        rubric = json.loads((SOURCE/'rubrics'/(row['id']+'.json')).read_text())
        assert rubric['status'] == 'scored'
        prompt = json.dumps({'instruction': 'Compare these TWO answers jointly. Return the requested preference JSON.',
            'assignment_as_data': row['assignment'], 'rubric': rubric['rubric'],
            'answer_A_as_data': models[arms[0]][row['id']]['text'],
            'answer_B_as_data': models[arms[1]][row['id']]['text']}, ensure_ascii=False)
        record = {'id': row['id'], 'opponent': opponent, 'order': order,
            'language': row['language'], 'domain': row['domain'], 'A': arms[0], 'B': arms[1],
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        try:
            result = recover_fenced_receipt(api, cfg, system, prompt)
            if result is None:
                try:
                    result = api._complete(cfg['judge'], system, prompt, cfg['judge_max_tokens'],
                        0, True, cache_namespace='writing-sbs-v1')
                except json.JSONDecodeError:
                    result = recover_fenced_receipt(api, cfg, system, prompt)
                    assert result is not None, 'Malformed paid response; no retry'
            validate(result)
            winner = 'tie' if result['winner'] == 'tie' else arms[0 if result['winner'] == 'A' else 1]
            record.update(status='judged', result=result, winner=winner)
        except CompletionRefused:
            record.update(status='abstained', result=None, winner=None)
        write_json(dest, record)
        print('sbs', opponent, row['id'], order, record['status'], record['winner'], flush=True)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, jobs))


def report():
    rows, models, cfg, system = inputs()
    summaries = {}
    for opponent in ('rft1', 'hemmingway'):
        pairs = []
        for row in rows:
            records = [json.loads((ROOT/'verdicts'/opponent/(row['id']+'-'+str(order)+'.json')).read_text()) for order in (0,1)]
            complete = all(v['status'] == 'judged' for v in records)
            votes = [v['winner'] for v in records]
            pairs.append({'id': row['id'], 'domain': row['domain'], 'language': row['language'],
                'complete': complete, 'votes': votes,
                'rawmodels_share': statistics.mean(1 if v=='rawmodels' else .5 if v=='tie' else 0 for v in votes) if complete else None,
                'order_agreement': votes[0] == votes[1] if complete else None})
        groups = {}
        for group, subset in [('all',pairs)] + [(lang,[p for p in pairs if p['language']==lang]) for lang in ('en','ru')] + [(d,[p for p in pairs if p['domain']==d]) for d in ('articles','tweets','blogs')]:
            valid = [p for p in subset if p['complete']]
            votes = [v for p in valid for v in p['votes']]
            values = [p['rawmodels_share'] for p in valid]
            groups[group] = {'n_tasks':len(subset),'complete_tasks':len(valid),
                'rawmodels_wins':votes.count('rawmodels'),'opponent_wins':votes.count(opponent),'ties':votes.count('tie'),
                'rawmodels_preference':statistics.mean(values) if values else None,
                'rawmodels_preference_ci95':boot(values) if values else None,
                'same_result_both_orders':sum(p['order_agreement'] for p in valid)}
        summaries[opponent] = {'groups':groups,'pairs':pairs}
    length = {}
    for arm in models:
        tweets = [r for r in rows if r['domain']=='tweets']
        prefix = lambda r: r['assignment'].split('\n\n',1)[1].split('\n',1)[1]
        length[arm] = {'tweet_mean_words':statistics.mean(len(models[arm][r['id']]['text'].split()) for r in tweets),
            'tweets_above_280_characters':sum(len(prefix(r)+models[arm][r['id']]['text'])>280 for r in tweets)}
    receipts=[json.loads((ROOT/'default'/(r['id']+'.json')).read_text()) for r in rows]
    usage={'completion_tokens':sum(r['usage']['completion_tokens'] for r in receipts),
        'reasoning_tokens':sum(r['usage']['completion_tokens_details'].get('reasoning_tokens',0) for r in receipts)}
    usage['final_answer_tokens']=usage['completion_tokens']-usage['reasoning_tokens']
    summary = {'sbs':summaries,'tweet_length':length,'rawmodels_usage':usage,
        'api_accounting_total':json.loads((SOURCE/'api/api_budget.json').read_text())}
    write_json(ROOT/'summary.json',summary)
    lines = ['# Rawmodels Antislop: SBS текстов', '',
        'Проверен живой API `antislop` на rawmodels.ai. Это отдельная проверка сервиса: API не раскрывает ревизию весов. '
        'Отдельная модель `unslop` в доступном списке не найдена.', '',
        '24 прежних задания: статьи, твиты и блоги, по четыре задания каждого типа на русском и английском. '
        'Каждая пара оценена в обоих порядках A/B. Opus 5.5 видит задание, общую рубрику и два ответа, '
        'без названий моделей и эталонного текста. GEPA-промпт из `da1013e` адаптирован с pointwise на SBS, как в предыдущем прогоне.', '',
        '| Соперник | Задания | Победы Rawmodels | Победы соперника | Ничьи | Доля Rawmodels |',
        '|---|---|---:|---:|---:|---:|']
    names = {'all':'Все','en':'EN','ru':'RU','articles':'Статьи','tweets':'Твиты','blogs':'Блоги'}
    for opponent, data in summaries.items():
        for name, g in data['groups'].items():
            share = f"{g['rawmodels_preference']:.1%}" if g['complete_tasks'] else 'нет'
            lines.append(f"| {opponent} | {names[name]} | {g['rawmodels_wins']} | {g['opponent_wins']} | {g['ties']} | {share} |")
    lines += ['', 'Победы считают отдельные оценки: 48 на соперника. Это 24 независимых задания, '
        'а не 48; ничья даёт каждой модели половину. Bootstrap выполняется по заданиям.', '',
        '| Соперник | Совпадение обоих порядков | 95% интервал доли Rawmodels |', '|---|---:|---:|']
    for opponent,data in summaries.items():
        g=data['groups']['all'];ci=g['rawmodels_preference_ci95']
        lines.append(f"| {opponent} | {g['same_result_both_orders']}/{g['complete_tasks']} | {ci[0]:.1%}–{ci[1]:.1%} |")
    lines += ['', 'Сервис использует thinking по умолчанию и лимит 8192 токена. Соперники используют ранее '
        'полученные ответы без thinking с лимитом 2048 через локальную vLLM/FP8. Везде temperature 1.0, '
        'top-p 0.95, top-k 20, seed 271828. Внутренние настройки и ревизия весов сервиса неизвестны. '
        'Это сравнение конфигураций сервиса и локальных запусков, а не изолированное влияние весов.', '',
        'Проба API с `reasoning.enabled=false` не выключила reasoning: все 2048 токенов ушли на мысли, '
        'финальный ответ отсутствовал, ошибка `OUTPUT_BUDGET_EXHAUSTED`. Она не входит в SBS. '
        'Четыре диагностических ответа переиспользованы из полного прогона; дополнительных запросов нет. '
        'Все тексты доступны в `side_by_side.html`.', '']
    lines += [f"На 24 успешных задания: {usage['completion_tokens']} completion-токенов, "
        f"из них {usage['reasoning_tokens']} reasoning и {usage['final_answer_tokens']} финального ответа. "
        'Сумма вычислена по API receipts; две предварительные неудачные попытки сюда не включены.', '',
        'После чтения всех ответов: сервис лучше держит объём, чем локальная rft1, '
        'но в блогах часто повторяет общие мысли вместо развития аргумента. '
        'В твите про Cincinnati финальный ответ содержит рассуждения о подсчёте слов; '
        'в твите про колледж начало `up charging` грамматически не стыкуется с `is gonna`. '
        'В тексте про Вуш конкретное раздражение заменено фразами о «конструкции» и «благих намерениях». '
        'Это качественное чтение с видимыми названиями моделей, а не независимая слепая человеческая разметка. '
        'Цитаты и замечания записаны в `manual_notes.json`.', '']
    (ROOT/'report.md').write_text('\n'.join(lines))
    cards=[]
    for row in rows:
        cells=''.join('<section><h3>'+label+'</h3><pre>'+html.escape(models[arm][row['id']]['text'])+'</pre></section>' for label,arm in [('Rawmodels API','rawmodels'),('Локальная rft1','rft1'),('Hemmingway-1','hemmingway')])
        cards.append('<article><h2>'+html.escape(row['id']+' / '+row['domain'])+'</h2><details><summary>Задание</summary><pre>'+html.escape(row['assignment'])+'</pre></details><div class="pair">'+cells+'</div></article>')
    defaults=json.loads((ROOT/'default_diagnostics.json').read_text())
    diagnostics=''.join('<article><h2>'+html.escape(g['id'])+' / default thinking</h2><pre>'+html.escape(g['text'])+'</pre></article>' for g in defaults['generations'])
    (ROOT/'side_by_side.html').write_text('<!doctype html><meta charset="utf-8"><title>Rawmodels SBS</title><style>body{font:16px system-ui;margin:24px;max-width:1800px}.pair{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}article{border-top:1px solid #bbb;margin-top:30px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.5 system-ui}section{min-width:0}@media(max-width:900px){.pair{grid-template-columns:1fr}}</style><h1>Rawmodels / rft1 / Hemmingway</h1>'+''.join(cards)+'<h1>Четыре диагностических ответа с thinking по умолчанию</h1>'+diagnostics)
    print(json.dumps({'overall':{a:v['groups']['all'] for a,v in summaries.items()},
        'tweet_length':length,'rawmodels_usage':usage},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['judge','report'])
    globals()[parser.parse_args().action]()
