"""Summarize this frozen pilot, keeping infrastructure errors in denominators."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / 'scaffold_training/private'

def main():
    runs = {}
    for phase in ('baseline', 'after'):
        for agent, tag in (('pi', ''), ('hermes', '-r2')):
            name = f'{phase}-{agent}{tag}'
            metric_path = PRIVATE / f'scored-{name}/metrics.jsonl'
            summary_path = PRIVATE / f'dataset-{name}/summary.json'
            if not metric_path.exists():
                continue
            rows = [json.loads(x) for x in metric_path.read_text().splitlines()]
            valid = [x for x in rows if x['valid']]
            runs[name] = {
                'collection': json.loads(summary_path.read_text()),
                'scored': len(valid),
                'mean_reward_valid': sum(x['score'] for x in valid)/len(valid) if valid else None,
                'mean_reward_all_attempts': sum(x['score'] for x in valid)/len(rows),
                'anti_slop_rate_valid': sum(x['anti_slop'] for x in valid)/len(valid) if valid else None,
                'rows': rows,
            }
    update = PRIVATE/'rl-adapter/update.json'
    comparison = {}
    for agent, tag in (('pi',''), ('hermes','-r2')):
        before, after = runs.get(f'baseline-{agent}{tag}'), runs.get(f'after-{agent}{tag}')
        if before and after:
            b, a = before['mean_reward_all_attempts'], after['mean_reward_all_attempts']
            comparison[agent] = {'before':b, 'after':a, 'delta':a-b}
    budget = json.loads((ROOT/'api_budget.json').read_text())
    rental = json.loads((ROOT/'instance.json').read_text())
    gpu_quote = (rental.get('finished',time.time())-rental['created'])*rental['hourly_quote']/3600
    pending = sum(budget['pending'].values())
    costs = {'gpu_quote_estimate':gpu_quote, 'api_with_10pct_margin':budget['spent'],
             'unsettled_api_reserve':pending, 'storage_network_reserve':2.0,
             'conservative_total_estimate':gpu_quote+budget['spent']+pending+2.0,
             'user_limit':20.0, 'final_invoice_verified':False}
    result = {'runs': runs, 'comparison': comparison, 'cost_estimate':costs,
              'conclusion': 'One verified optimizer step and working adapter in both scaffolds; robust quality improvement is not established.',
              'effective_training_sources': ['unslop-harbor English tweets'],
              'traffic_training_exclusion': 'Three zero-variance groups and one group with an unavailable timeout reward.',
              'trainer': 'scaffold_training/grpo_update.py at 959e356 plus recorded local patch',
              'update': json.loads(update.read_text()) if update.exists() else None,
              'budget': budget,
              'rental': rental,
              'limitations': ['Four train tasks and two dev tasks; one sample per scaffold/dev task.',
                              'One optimizer step cannot establish robust improvement.',
                              'Traffic uses supplied frozen evidence; no live search evaluation.',
                              'Unslop subset covers English tweet continuations, not code.',
                              'Final trainer is repository GRPO implementation, not verl.',
                              'Evaluation is sampled; before/after differences include sampling noise.',
                              'Pi baseline used 16K serving context; corrected Hermes and after runs used 64K. Validation sequences were below 4K.',
                              'After evaluation ran both scaffolds concurrently; all four attempts completed within the unchanged timeout.',
                              'English-only small pilot does not establish resistance to new generator models or unseen scaffolds.']}
    (ROOT/'scaffold_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k:{j:v for j,v in r.items() if j!='rows'} for k,r in runs.items()},indent=2))

if __name__ == '__main__':
    main()
