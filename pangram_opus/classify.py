"""Classify English text with the frozen prompt. Requires OPENROUTER_API_KEY."""
import argparse
import json
import sys
from pathlib import Path
from core import ROOT, predict, PredictionUnavailable

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('file', nargs='?', help='UTF-8 text file; omit to read stdin')
    p.add_argument('--prompt', default=str(ROOT/'best_prompt.txt'))
    p.add_argument('--mode', choices=['minimal','default','medium','high'], help='Override the calibrated runtime')
    a = p.parse_args()
    prompt_path = Path(a.prompt)
    if not prompt_path.exists():
        raise SystemExit('No calibrated prompt exists yet')
    text = Path(a.file).read_text() if a.file else sys.stdin.read()
    if not text.strip():
        raise SystemExit('Empty input')
    config_path=ROOT/'inference_config.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else {'mode':'minimal'}
    mode=a.mode or config['mode']
    kind=config.get('kind','binary')
    try:
        if kind=='score':
            if mode!='default':p.error('The calibrated score classifier requires --mode default')
            threshold=config.get('threshold')
            if type(threshold) is not int or not 0<=threshold<=101:raise ValueError('Invalid calibrated threshold')
            from round5.score_runtime import score
            label=score(prompt_path.read_text(),text)>=threshold
        elif kind!='binary':
            raise ValueError('Unknown inference kind')
        elif mode=='minimal':
            label = predict(prompt_path.read_text(), text)
        else:
            from round3.runtime import predict_mode
            label = predict_mode(prompt_path.read_text(), text, mode)
    except PredictionUnavailable as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({'slop': label}))
