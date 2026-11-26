"""Shinka evaluator: run antislop with the candidate SYSTEM on the fixed train tasks."""
import argparse
import json
from pathlib import Path
from common import load_rows, read_candidate, score_set

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--program_path', required=True)
    p.add_argument('--results_dir', required=True)
    a = p.parse_args()
    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        summary, results = score_set(read_candidate(a.program_path), load_rows('train.jsonl'))
        failed = sorted([r for r in results if not r['passed']], key=lambda r: -(r['ai_score'] or 100))
        def brief(r):
            reason = ('empty output: reasoning used the whole 16384-token budget' if not r['text'] else
                      'off-task' if r['task'] < 6 else
                      f"length {r['words']} words vs target {r['target']}" if not r['length_ok'] else
                      f"flagged as AI (score {r['ai_score']})")
            return {'kind': r['kind'], 'reason': reason, 'text_start': r['text'][:500]}
        passed = [{'kind': r['kind'], 'ai_score': r['ai_score'], 'text_start': r['text'][:400]}
                  for r in results if r['passed']][:3]
        import hashlib, random
        anchors = load_rows('anchors.jsonl')
        rng = random.Random(hashlib.sha256(Path(a.program_path).read_bytes()).hexdigest())
        style = [{'source': x['origin'], 'text': x['text'][:700]} for x in rng.sample(anchors, 4)]
        feedback = {'summary': summary, 'failures': [brief(r) for r in failed[:10]], 'passing_examples': passed,
                    'target_style_examples': {'note': 'Machine-written texts that the REAL Pangram detector labeled '
                                              'Human Written. Infer their general traits (structure, rhythm, word '
                                              'choice, specificity); do not copy their content.', 'examples': style}}
        result = {'combined_score': summary['fitness'], 'public': summary,
                  'text_feedback': json.dumps(feedback, ensure_ascii=False)}
        (out / 'results_detail.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))
        correct = {'correct': True}
    except Exception as e:
        result = {'combined_score': 0, 'text_feedback': type(e).__name__ + ': ' + str(e)}
        correct = {'correct': False, 'error': result['text_feedback']}
    (out / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False))
    (out / 'correct.json').write_text(json.dumps(correct))
