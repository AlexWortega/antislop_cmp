"""Blind SBS with the user's frozen GEPA judge and shared spend ledger."""
import argparse
import hashlib
import itertools
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'cmp_hemmingway/out'
sys.path.insert(0,str(ROOT.parent/'cmp_hemmingway/source/sep/unslop'))
from api import API
from compare_sbs import sbs_system,validate,recover_fenced_receipt
from serving import SYSTEM

RUBRIC={'criteria':[
    {'id':'fidelity','description':'Preserve all substantive source facts, names, quantities, dates, negations, uncertainty, exclusions and instructions. No invented details or strengthening claims. Changing word forms or writing a number in words is valid if the fact is retained.'},
    {'id':'prose','description':'Natural, direct prose in the original source language, appropriate voice and useful politeness. Remove corporate filler and inflated generic claims without stripping useful substance or becoming poetic.'},
    {'id':'format','description':'Only the finished edited text; no preface, analysis, thinking or word-count commentary. Maintain useful paragraph/markdown structure. Complete all sentences. No unnecessary expansion or loss of substantive content.'}
]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',required=True)
    p.add_argument('--arms',default='q4-native,q4-balanced,q4-careful')
    a=p.parse_args(); arms=a.arms.split(',')
    rows=[r for r in json.loads((ROOT/'calibration/cases.json').read_text()) if r['split']==a.split]
    cfg=json.loads((SOURCE/'config.json').read_text())
    system=sbs_system(json.loads((SOURCE/'best_candidate.json').read_text())['judge_system'])
    os.environ['OPENROUTER_API_KEY']=json.loads(Path('/Users/alexwortega/Desktop/side_projects/rawmodels/humanizer/.local/secrets.json').read_text())['OPENROUTER_API_KEY']
    api=API(SOURCE/'api',cfg)
    jobs=[(r,x,y,k) for r in rows for x,y in itertools.combinations(arms,2) for k in (0,1)]
    random.Random(271828).shuffle(jobs)
    def one(job):
        r,x,y,k=job; pair=(x,y) if not k else (y,x)
        dest=ROOT/'calibration/verdicts'/a.split/f'{r["id"]}-{x}-vs-{y}-{k}.json'
        if dest.exists(): validate(json.loads(dest.read_text())['result']);return
        texts=[json.loads((ROOT/'calibration/generations'/f'{arm}-{r["id"]}.json').read_text()) for arm in pair]
        prompt=json.dumps({'instruction':'Compare these TWO rewrites jointly. Return the requested preference JSON.',
            'assignment_as_data':SYSTEM+'\n\nDRAFT:\n'+r['draft'],'rubric':RUBRIC,
            'answer_A_as_data':texts[0]['text'],'answer_B_as_data':texts[1]['text']},ensure_ascii=False)
        result=recover_fenced_receipt(api,cfg,system,prompt)
        if result is None:
            try: result=api._complete(cfg['judge'],system,prompt,cfg['judge_max_tokens'],0,True,cache_namespace='writing-sbs-v1')
            except json.JSONDecodeError:
                # No rebilling of a paid malformed response.
                result=recover_fenced_receipt(api,cfg,system,prompt)
                if result is None: raise
        validate(result)
        winner='tie' if result['winner']=='tie' else pair[0 if result['winner']=='A' else 1]
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(dict(id=r['id'],A=pair[0],B=pair[1],winner=winner,result=result,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()),ensure_ascii=False,indent=2)+'\n')
        print(r['id'],x,y,k,winner,flush=True)
    with ThreadPoolExecutor(4) as pool: list(pool.map(one,jobs))
    counts={arm:dict(wins=0,losses=0,ties=0) for arm in arms}
    for f in (ROOT/'calibration/verdicts'/a.split).glob('*.json'):
        d=json.loads(f.read_text())
        if d['A'] not in arms or d['B'] not in arms: continue
        for arm in (d['A'],d['B']):
            counts[arm]['ties' if d['winner']=='tie' else 'wins' if d['winner']==arm else 'losses']+=1
    target=ROOT/'calibration'/f'{a.split}-summary.json'
    target.write_text(json.dumps(counts,indent=2)+'\n');print(counts)

if __name__=='__main__':main()
