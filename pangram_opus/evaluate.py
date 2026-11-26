import argparse
import json
from pathlib import Path
from core import ROOT, read_prompt, score

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--program_path', required=True)
    p.add_argument('--results_dir', required=True)
    a = p.parse_args()
    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        rows = [json.loads(line) for line in (ROOT/'data/train.jsonl').read_text().splitlines()]
        metrics, predictions = score(read_prompt(a.program_path), rows)
        errors = [{'text': r['text'], 'expected_slop': r['slop'], 'predicted_slop': pred}
                  for r, pred in zip(rows, predictions) if r['slop'] != pred][:24]
        anchors = []
        for label in [False, True]:
            anchors.extend([{'text': r['text'], 'expected_slop': label} for r, pred in zip(rows, predictions)
                            if r['slop'] == pred == label][:2])
        result = {'combined_score': metrics['balanced_accuracy'], 'public': metrics,
                  'text_feedback': json.dumps({'training_errors': errors, 'correct_training_examples_to_preserve': anchors}, ensure_ascii=False)}
        correct = {'correct': True}
    except Exception as e:
        result = {'combined_score': 0, 'text_feedback': type(e).__name__ + ': ' + str(e)}
        correct = {'correct': False, 'error': result['text_feedback']}
    (out/'metrics.json').write_text(json.dumps(result, ensure_ascii=False))
    (out/'correct.json').write_text(json.dumps(correct))
