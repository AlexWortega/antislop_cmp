"""Gates for the pairwise "beat the human" judge (lab_core.ref_compare) before it goes into training.

  python3 tasks_v2/check_ref.py            # Opus (config ref_judge) + Sonnet agreement; writes tasks_v2/ref_report.json
Sets, all held-out or never trained on:
  model_vs_human  contrast.jsonl held-out: an Opus or Flash answer against the real human answer to the same question
  hack            contrast_pairs.jsonl: a policy answer with a made-up name/story against the same answer cleaned
                  (the clean one is passed as the "reference"; the judge should prefer it)
Each pair is judged in both orders, which measures position bias and gives an order-free verdict.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'judge_playground'))
import lab_core as core  # noqa: E402

SONNET = 'anthropic/claude-sonnet-5.5'


def rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines()]


def both_orders(cfg, items):
    """items: [(task, candidate, reference)] -> [(beat when candidate first, beat when candidate second)]"""
    def one(item):
        task, text, ref = item
        first = core.ref_compare(cfg, task, text, ref, candidate_first=True)[0]['beat']
        second = core.ref_compare(cfg, task, text, ref, candidate_first=False)[0]['beat']
        return first, second
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, items))


def stats(results):
    n = len(results)
    first = sum(a for a, _ in results) / n
    second = sum(b for _, b in results) / n
    consistent = sum(a == b for a, b in results) / n
    return {'n': n, 'candidate_win_rate': (first + second) / 2, 'win_rate_when_first': first, 'win_rate_when_second': second,
            'position_bias_pp': round(100 * (first - second), 1), 'same_verdict_both_orders': consistent}


def main():
    core.load_env()
    cfg = core.load_config()
    data = rows('contrast.jsonl')
    humans = {r['task']: r['text'] for r in data if not r['slop'] and r['split'] == 'heldout'}
    model_vs_human = [(r['task'], r['text'], humans[r['task']]) for r in data
                      if r['slop'] and r['split'] == 'heldout' and r['source'].startswith('gen-') and r['task'] in humans]
    hack = [(p['task'], p['original'], p['clean']) for p in rows('contrast_pairs.jsonl')[:30]]
    before = core.ledger()['spent_usd']
    report = {}
    opus_mvh, opus_hack = both_orders(cfg, model_vs_human), both_orders(cfg, hack)
    report['opus'] = {'model_vs_human': stats(opus_mvh), 'hack_original_vs_clean': stats(opus_hack)}
    sonnet_cfg = {**cfg, 'ref_judge': {**cfg['ref_judge'], 'model': SONNET}}
    sonnet_mvh, sonnet_hack = both_orders(sonnet_cfg, model_vs_human), both_orders(sonnet_cfg, hack)
    report['sonnet'] = {'model_vs_human': stats(sonnet_mvh), 'hack_original_vs_clean': stats(sonnet_hack)}
    pairs = list(zip(opus_mvh + opus_hack, sonnet_mvh + sonnet_hack))
    report['sonnet_agrees_with_opus'] = sum((a1 + a2) == (b1 + b2) for (a1, a2), (b1, b2) in pairs) / len(pairs)
    report['cost_usd'] = round(core.ledger()['spent_usd'] - before, 3)
    gates = report['opus']
    report['gates'] = {
        'position_bias_le_10pp': abs(gates['model_vs_human']['position_bias_pp']) <= 10 and abs(gates['hack_original_vs_clean']['position_bias_pp']) <= 10,
        'made_up_loses_to_clean': gates['hack_original_vs_clean']['candidate_win_rate'] < 0.5,
        'sonnet_ok_for_cheap_judge': report['sonnet_agrees_with_opus'] >= 0.85,
    }
    (ROOT / 'ref_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
