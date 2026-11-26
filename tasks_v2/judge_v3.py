"""Judge v3: a pleasantness rubric derived from real human answers (Meta RL-XAR recipe), and the gates it must pass.

  python3 tasks_v2/judge_v3.py derive   # 3 candidate rubrics from human-vs-model train pairs, keep the best by train AUROC
  python3 tasks_v2/judge_v3.py check [repeats] [pairs]   # v2 vs v3 on held-out contrast.jsonl and on contrast_pairs.jsonl
  python3 tasks_v2/judge_v3.py apply    # write the v3 judge, calibration and formula into judge_config.json
Every judge call goes through lab_core (cache, budget ledger), so reruns are free.
"""
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'judge_playground'))
import lab_core as core  # noqa: E402

SEED = 20261112
RUBRIC = ROOT / 'rubric_v3.txt'
REPORT = ROOT / 'judge_v3_report.json'
WRITER = 'anthropic/claude-opus-5.5'
# Picked on held-out contrast.jsonl + 30 hack pairs (cached scores): human 0.7 / slop 0.3 inside, invented/6.
# AUROC human>model 0.859; hack pairs: clean wins 21, tie 6, original 3. "pleasant" alone gave AUROC 0.551, so it stays a note.
FORMULA = '(0.49*human/10 + 0.21*(1 - slop/10) + 0.3*length) * min(1, task/7) * max(0, 1 - invented/6) - penalty'

# Always in the rubric, whatever the derivation finds. Sources:
#   HAL (2026) human-likeness items 1, 6, 10, 13, 14 (its items about typos, lowercase and emoji are left out on purpose);
#   "Learning to Simulate Human Dialogue" (2026): judge-only RL drifts into padding, affirmations, repeating the question;
#   AttuneBench (2026): tone matched to the other person, emotional fit;
#   QA-CompletenessRelevance (2026) and Meta RL-XAR: completeness pulls toward the assistant voice, humans leave things out.
# v3 (the run-1111 calibration) had only the first four rules (FIXED_V3, exactly as applied); v4 adds the rest.
FIXED_V3 = """Fixed rules (always apply):
- Invented people and events are defects, never "personality": a name for the recipient or the writer that the assignment does not give, a personal story or biography presented as fact (where the writer lives, "someone asked me this yesterday", "I tried this last summer"). A person who does not know a detail says so or leaves it out.
- Padding is a defect: restating the question, affirming the asker ("great question", "you're right to wonder"), summaries of what was just said, offers of more help.
- Length is not quality and is checked separately; do not reward a longer text for covering more.
- Typos, lowercase, and slang are not evidence of quality by themselves."""
FIXED = """Fixed rules (always apply):
- Invented people and events are defects, never "personality": a name for the recipient or the writer that the assignment does not give, a personal story or biography presented as fact (where the writer lives, "someone asked me this yesterday", "I tried this last summer"). A person who does not know a detail says so or leaves it out. Details that the assignment itself gives about the writer (a persona) are not invented.
- Padding is a defect: restating the question, affirming the asker ("great question", "you're right to wonder"), summaries of what was just said, offers of more help.
- Length is not quality and is checked separately; do not reward a longer text for covering more.
- Typos, lowercase, and slang are not evidence of quality by themselves.
- A real person keeps it brief and does not over-explain; they answer the main thing and leave the rest out. Completeness is not a merit.
- A real person builds on what the other person actually said and stays on their topic, instead of drifting into a general lecture.
- When they give a reason, it is a plain one (cost, habit, what worked for them if the assignment allows it), not a list of considerations.
- In emails and replies the tone fits the other person and the moment: no assistant-style care, no unasked reassurance or life advice."""

DERIVE_PROMPT = """Below are pairs of answers to the same question. In each pair, HUMAN is a real person's answer from a Q&A site, MODEL is a language model's answer to the same question with the same instructions.

Write a grading rubric that a judge will use on a single text, with no pair to compare, to score how pleasant and trustworthy it is to read, the way readers actually prefer the human answers. Rules for the rubric:
- At most 15 numbered criteria, each one sentence, each about genuine quality for the reader (how the answer gets to the point, what it chooses to say and to leave out, voice, specificity that is real rather than invented, how it treats the reader), not superficial tells such as dashes or a specific word.
- Do not reward typos, lowercase, or slang as such. Do not reward length or completeness.
- Criteria must work for Russian and English, and for forum answers, emails to a friend, blog posts and wiki notes.
- End with a short scale: what a 2, a 5, an 8 and a 10 look like.
Return only the rubric text.

{pairs}"""

USER_TEMPLATE = """Task the text was written for:
<task>{task}</task>

Text to judge:
<text>{text}</text>

Do this:
1. List every AI-writing tell from the rulebook that appears in the text, as {"pattern": "<rulebook pattern name>", "quote": "<short exact quote>"}.
2. List in "invented_items" every made-up person, name, personal event or biography of the writer, or specific claim that is neither in the task nor common knowledge, as short quotes.
3. Scores:
   - "pleasant": 1-10 by the pleasantness rubric: would a reader enjoy this and trust the writer.
   - "human": 1-10, how much it reads like a specific real person wrote it. Invented personal details do not make it more human.
   - "slop": 0-10, how much AI slop by the rulebook (0 = none, 10 = saturated).
   - "invented": 0-10, how much of the text rests on made-up people, events or facts (0 = none, 3 = one small made-up detail such as a name in the greeting, 6 = a made-up story, 10 = mostly invented).

Answer with JSON only:
{"tells": [...], "invented_items": [...], "pleasant": int, "human": int, "slop": int, "invented": int}"""

FIELDS = [{'name': 'pleasant', 'min': 1, 'max': 10, 'description': 'pleasant and trustworthy to read, by the rubric'},
          {'name': 'human', 'min': 1, 'max': 10, 'description': 'reads like a specific real person wrote it'},
          {'name': 'slop', 'min': 0, 'max': 10, 'description': 'AI slop by the rulebook'},
          {'name': 'invented', 'min': 0, 'max': 10, 'description': 'made-up people, events or facts'},
          {'name': 'tells', 'min': 0, 'max': 500, 'description': 'number of items the judge listed'},
          {'name': 'invented_items', 'min': 0, 'max': 500, 'description': 'number of items the judge listed'}]


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines()]


def base_system(cfg):
    """The v2 system prompt (antislop rulebook) without any earlier v3 rubric appended."""
    return cfg['judge']['system'].split('\n\n<pleasantness_rubric>')[0]


def v3_cfg(cfg, rubric, fixed=None):
    fixed = FIXED if fixed is None else fixed
    judge = {**cfg['judge'], 'system': base_system(cfg) + f'\n\n<pleasantness_rubric>\n{rubric.strip()}\n\n{fixed}\n</pleasantness_rubric>',
             'user_template': USER_TEMPLATE, 'fields': FIELDS, 'input': 'template', 'mode': 'single'}
    return {**cfg, 'judge': judge, 'calibration': {'ai_when': 'human < 7', 'score_field': 'human', 'higher_is_ai': False}}


def v2_cfg(cfg):
    return {**cfg, 'calibration': {'ai_when': 'human < 8', 'score_field': 'human', 'higher_is_ai': False}}


def labeled(items):
    return [{'text': r['text'], 'slop': r['slop'], 'task': r['task'], 'source': r['source'],
             'words': core.word_count(r['text']), 'task_set': False} for r in items]


def score_texts(cfg, items):
    """Judge scores for [(task, text)], in order."""
    def one(item):
        return core.judge_call(cfg, item[0], [item[1]])[0][0]
    with ThreadPoolExecutor(max_workers=int(cfg['judge'].get('concurrency', 4))) as pool:
        return list(pool.map(one, items))


def derive():
    core.load_env()
    cfg = core.load_config()
    rng = random.Random(SEED)
    data = rows('contrast.jsonl')
    train = [r for r in data if r['split'] == 'train']
    humans = {r['task']: r for r in train if not r['slop']}
    pairs = [(humans[r['task']], r) for r in train if r['source'].startswith('gen-') and r['task'] in humans]
    machine = [r for r in train if r['slop']]
    score_set = rng.sample([r for r in train if not r['slop']], 30) + rng.sample(machine, 30)
    endpoint = {'base_url': 'https://openrouter.ai/api/v1', 'model': WRITER, 'api_key_env': 'OPENROUTER_API_KEY'}
    results = []
    for k in range(3):
        sample = rng.sample(pairs, 20)
        text = '\n\n'.join(f'### Pair {i + 1}\nQuestion task: {h["task"]}\n\nHUMAN:\n{h["text"]}\n\nMODEL:\n{m["text"]}'
                           for i, (h, m) in enumerate(sample))
        body = {'model': WRITER, 'max_tokens': 4000, 'usage': {'include': True},
                'messages': [{'role': 'user', 'content': DERIVE_PROMPT.format(pairs=text)}]}
        reply, _ = core.chat(endpoint, body, cfg['budget_usd'], repeat=f'rubric-{k}')
        rubric = reply['choices'][0]['message']['content'].strip()
        metrics = core.evaluate_labeled(v3_cfg(cfg, rubric), labeled(score_set))
        results.append({'k': k, 'auroc': metrics['auroc'], 'cost_usd': metrics['cost_usd'], 'rubric': rubric,
                        'by_source': metrics['by_source']})
        print(f'candidate {k}: train AUROC {metrics["auroc"]:.3f}, judge ${metrics["cost_usd"]:.2f}', flush=True)
    best = max(results, key=lambda r: r['auroc'])
    RUBRIC.write_text(best['rubric'] + '\n')
    REPORT.write_text(json.dumps({'derive': results, 'best': best['k']}, ensure_ascii=False, indent=1))
    print('kept candidate', best['k'], '->', RUBRIC)


REFINE_PROMPT = """You wrote the grading rubric below. A judge applied it to single texts, and it must rank real human answers above model answers to the same questions. It does not yet: AUROC {auroc:.3f} (0.5 = chance, 1 = every human above every model). Mean scores by source: {means}.

Below are the cases it got most wrong: human answers it scored low and model answers it scored high, with the judge's reasons. Revise the rubric so the judge rewards what genuinely makes the human answers better for a reader, and stops rewarding what makes the model answers merely look polished or complete. Same rules as before: at most 15 one-sentence criteria about genuine quality, no reward for typos, lowercase, slang, length or completeness, works for RU and EN and for answers, emails, posts and wiki notes, ends with a short 2/5/8/10 scale. If a human answer is genuinely poor (off-topic, lazy, wrong), do not bend the rubric to reward it. Return only the rubric text.

<rubric>
{rubric}
</rubric>

{cases}"""


def refine(rounds=3):
    core.load_env()
    cfg = core.load_config()
    rng = random.Random(SEED + 1)
    train = [r for r in rows('contrast.jsonl') if r['split'] == 'train']
    score_set = labeled(rng.sample([r for r in train if not r['slop']], 30) + rng.sample([r for r in train if r['slop']], 30))
    endpoint = {'base_url': 'https://openrouter.ai/api/v1', 'model': WRITER, 'api_key_env': 'OPENROUTER_API_KEY'}
    report = json.loads(REPORT.read_text())
    rubric, history = RUBRIC.read_text(), report.get('refine', [])
    best = None
    for k in range(rounds + 1):
        judge_cfg = v3_cfg(cfg, rubric)
        scores = score_texts(judge_cfg, [(r['task'], r['text']) for r in score_set])
        human = [s['pleasant'] for r, s in zip(score_set, scores) if not r['slop']]
        model = [s['pleasant'] for r, s in zip(score_set, scores) if r['slop']]
        auroc = core.auroc(human, model)   # P(human scored above model)
        means = {}
        for r, s in zip(score_set, scores):
            means.setdefault(r['source'], []).append(s['pleasant'])
        means = {src: round(statistics.mean(v), 2) for src, v in means.items()}
        history.append({'round': k, 'auroc': auroc, 'means': means, 'rubric': rubric})
        print(f'round {k}: train AUROC {auroc:.3f} {means}', flush=True)
        if best is None or auroc > best['auroc']:
            best = history[-1]
        if k == rounds:
            break
        wrong = sorted(zip(score_set, scores), key=lambda x: x[1]['pleasant'] if not x[0]['slop'] else -x[1]['pleasant'])
        cases = '\n\n'.join(f'### {"HUMAN (scored too low)" if not r["slop"] else "MODEL (scored too high)"}: pleasant {s["pleasant"]:.0f}\n'
                            f'Task: {r["task"]}\nText:\n{r["text"]}\nJudge notes: {s["reason"][:600]}'
                            for r, s in wrong[:5] + [w for w in wrong if w[0]['slop']][:5])
        body = {'model': WRITER, 'max_tokens': 4000, 'usage': {'include': True},
                'messages': [{'role': 'user', 'content': REFINE_PROMPT.format(auroc=auroc, means=means, rubric=rubric, cases=cases)}]}
        reply, _ = core.chat(endpoint, body, cfg['budget_usd'], repeat=f'refine-{len(history)}')
        rubric = reply['choices'][0]['message']['content'].strip()
    RUBRIC.write_text(best['rubric'] + '\n')
    report['refine'] = history
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f'kept round {best["round"]} (AUROC {best["auroc"]:.3f}) -> {RUBRIC}')


def check():
    core.load_env()
    cfg = core.load_config()
    held = labeled([r for r in rows('contrast.jsonl') if r['split'] == 'heldout'])
    pairs = rows('contrast_pairs.jsonl')[:int(sys.argv[3]) if len(sys.argv) > 3 else None]
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    configs = {'v2': (v2_cfg(cfg), 'human'), 'v3': (v3_cfg(cfg, RUBRIC.read_text()), 'pleasant')}
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    for name, (judge_cfg, field) in configs.items():
        metrics = core.evaluate_labeled(judge_cfg, held, repeats=repeats)
        originals = score_texts(judge_cfg, [(p['task'], p['original']) for p in pairs])
        cleans = score_texts(judge_cfg, [(p['task'], p['clean']) for p in pairs])
        wins = sum(c[field] > o[field] for o, c in zip(originals, cleans))
        ties = sum(c[field] == o[field] for o, c in zip(originals, cleans))
        out = {'auroc': metrics['auroc'], 'mean_abs_diff_repeat': metrics['mean_abs_diff'], 'flip_rate': metrics['flip_rate'],
               'n': metrics['n'], 'cost_usd': metrics['cost_usd'],
               'mean_by_source': {k: round(v['mean_score'], 2) for k, v in metrics['by_source'].items()},
               'pairs': {'n': len(pairs), 'clean_wins': wins, 'ties': ties, 'original_wins': len(pairs) - wins - ties,
                         'mean_original': statistics.mean(o[field] for o in originals),
                         'mean_clean': statistics.mean(c[field] for c in cleans)}}
        if name == 'v3':
            out['pairs']['median_invented_original'] = statistics.median(o['invented'] for o in originals)
            out['pairs']['median_invented_clean'] = statistics.median(c['invented'] for c in cleans)
        report[name] = out
        print(name, json.dumps(out, ensure_ascii=False), flush=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1))


def core_score(s):
    """The judge part of FORMULA (length and the task gate do not depend on the rules): what training optimises."""
    return (0.7 * s['human'] / 10 + 0.3 * (1 - s['slop'] / 10)) * max(0, 1 - s['invented'] / 6)


def check_v4():
    """Fixed rules v3 vs v4 with the same rubric, on what the reward uses: human, the formula's judge part, hack pairs."""
    core.load_env()
    cfg = core.load_config()
    held = [r for r in rows('contrast.jsonl') if r['split'] == 'heldout']
    pairs = rows('contrast_pairs.jsonl')[:30]
    rubric = RUBRIC.read_text()
    report = json.loads(REPORT.read_text())
    for name, fixed in (('fixed_v3', FIXED_V3), ('fixed_v4', FIXED)):
        judge_cfg = v3_cfg(cfg, rubric, fixed)
        scores = score_texts(judge_cfg, [(r['task'], r['text']) for r in held])
        originals = score_texts(judge_cfg, [(p['task'], p['original']) for p in pairs])
        cleans = score_texts(judge_cfg, [(p['task'], p['clean']) for p in pairs])
        out = {'n': len(held), 'pairs_n': len(pairs)}
        for field, fn in (('human', lambda s: s['human']), ('formula', core_score)):
            hum = [fn(s) for r, s in zip(held, scores) if not r['slop']]
            mod = [fn(s) for r, s in zip(held, scores) if r['slop']]
            wins = sum(fn(c) > fn(o) for o, c in zip(originals, cleans))
            ties = sum(fn(c) == fn(o) for o, c in zip(originals, cleans))
            out[field] = {'auroc': core.auroc(hum, mod), 'clean_wins': wins, 'ties': ties,
                          'original_wins': len(pairs) - wins - ties}
        by = {}
        for r, s in zip(held, scores):
            by.setdefault(r['source'], []).append(s)
        out['by_source'] = {src: {k: round(statistics.mean(x[k] for x in v), 2) for k in ('human', 'slop', 'invented')}
                            for src, v in by.items()}
        report[name] = out
        print(name, json.dumps(out, ensure_ascii=False), flush=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1))


def apply():
    cfg = core.load_config()
    new = v3_cfg(cfg, RUBRIC.read_text())
    new['reward'] = {**cfg['reward'], 'formula': FORMULA}
    print(core.save_config(new)['reward']['formula'])


if __name__ == '__main__':
    {'derive': derive, 'refine': refine, 'check': check, 'check_v4': check_v4, 'apply': apply}[sys.argv[1]]()
