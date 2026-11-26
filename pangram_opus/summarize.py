"""Combine frozen-prompt test results without changing the selected prompt."""
import json
import math
from pathlib import Path
from core import ROOT, digest, read_prompt

def wilson(correct,n):
    z=1.959963984540054;p=correct/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [center-delta,center+delta]

def main():
    holdout=json.loads((ROOT/'holdout_results.json').read_text())
    stress=json.loads((ROOT/'robustness/report.json').read_text())
    repeat=json.loads((ROOT/'robustness/repeatability.json').read_text())
    expected=digest(read_prompt(ROOT/'best_candidate.py'))
    assert holdout['optimized']['prompt_hash']==stress['prompt_hash']==expected
    report={'prompt_hash':expected,'holdout':{},'stress':stress,'repeatability':repeat}
    for name,r in holdout.items():
        correct=sum(p['teacher_slop']==p['opus_slop'] for p in r['predictions'])
        report['holdout'][name]={k:v for k,v in r.items() if k!='predictions'}
        report['holdout'][name].update(correct=correct,agreement_wilson95=wilson(correct,r['n']))
    base={p['hash']:p for p in holdout['baseline']['predictions']}
    better=worse=0
    for p in holdout['optimized']['predictions']:
        b=base[p['hash']];bc=b['teacher_slop']==b['opus_slop'];pc=p['teacher_slop']==p['opus_slop']
        better+=not bc and pc;worse+=bc and not pc
    discordant=better+worse
    pvalue=min(1,2*sum(math.comb(discordant,k) for k in range(min(better,worse)+1))/2**discordant) if discordant else 1
    report['holdout']['paired_comparison']={'improved':better,'regressed':worse,'exact_mcnemar_p':pvalue}
    overall=stress['overall'];n=overall['n'];positive=overall['tp']+overall['fn']
    report['stress']['always_slop_baseline']={'agreement':positive/n,'balanced_accuracy':0.5,'n':n}
    rows=[json.loads(l) for l in (ROOT/'robustness/predictions.jsonl').read_text().splitlines()]
    fresh=[r for r in rows if r['origin']=='fresh'];legacy=[r for r in rows if r['origin']!='fresh']
    report['stress']['fresh_vs_legacy']={name:{'n':len(group),'correct':sum(r['agree'] for r in group),'teacher_positive':sum(r['slop'] for r in group)} for name,group in [('fresh',fresh),('legacy',legacy)]}
    state=json.loads((ROOT/'cache/budget.json').read_text())
    teachers=[json.loads(p.read_text()) for p in (ROOT/'cache').glob('pangram4_*.json')]
    units=sum(math.ceil(sum(w['word_count'] for w in r['windows'])/100) for r in teachers)
    report['costs']={'openrouter_approx_reported_usd':round(state['spent']['openrouter']/1.1,4),
                     'pangram_estimate_usd':round(units*.05,2),'pangram_completed_requests':len(teachers),
                     'pangram_price_source':'https://www.pangram.com/pricing',
                     'conservative_budget_totals':{k:round(state['spent'][k]+sum(x['amount'] for x in state['pending'].values() if x['provider']==k),4) for k in state['limits']}}
    report['limitations']=[
        'Only 31 main holdout texts; confidence intervals are wide. Training score is not deployed holdout performance.',
        'Fresh model/scaffold cells contain four shared topics each. They are correlated and all 96 fresh texts received positive Pangram labels.',
        'Only six Pangram-negative stress texts: report both class recalls and compare with the always-slop baseline.',
        'Legacy generator revisions are not fully verifiable. Complete short outputs were retained; truncated/empty outputs were retried.',
        'Prompt selected using training batches; final measurements use single-text calls with strict JSON schema after a format-only client fix.',
        'Repeatability sample deliberately includes hard cases; it is not an unbiased population estimate.',
        'AI-assisted is mapped to slop. This reproduces a Pangram decision rule, not a universal measure of writing quality.']
    (ROOT/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['holdout','costs']},indent=2))
    print('Stress:',stress['overall'])
    print('Repeatability:',repeat['stable_texts'],'/',repeat['n_texts'])

if __name__=='__main__':main()
