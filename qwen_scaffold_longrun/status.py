"""Read-only local status, safe to run without any API credentials."""
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def read(name,default=None):
    path=ROOT/name
    return json.loads(path.read_text()) if path.exists() else default

state=read('progress.json',{})
rental=read('instance.json',{})
rate=read('instance_status.json',{}).get('dph_total')
budget=read('opus_budget.json',{})
out={'progress':state,'rental_status':rental.get('status'),
     'gpu_quoted_cost_estimate':(rental.get('finished',time.time())-rental['created'])/3600*rate if rate else None,
     'opus_spent_with_margin':budget.get('spent_with_margin'),
     'opus_pending_reserve':sum(v for k,v in budget.get('pending',{}).items() if k not in budget.get('policy_holds',[])),
     'opus_effective_user_limit':budget.get('effective_user_limit',budget.get('limit')),
     'account_credit_last_check':read('credit_status.json'),
     'latest_validation':None}
out['prior_gpu_reserved_usd']=read('restart_accounting.json',{}).get('prior_gpu_reserved_usd',0.)
if out['gpu_quoted_cost_estimate'] is not None:
    out['gpu_total_conservative_estimate']=out['gpu_quoted_cost_estimate']+out['prior_gpu_reserved_usd']
evaluations=sorted((ROOT/'evaluations').glob('*/evaluation.json')) if (ROOT/'evaluations').exists() else []
if evaluations:out['latest_validation']=json.loads(evaluations[-1].read_text())
phase=state.get('phase','')
active=(ROOT/'evaluations'/f"eval{state.get('evaluation_step',0):04d}") if phase.startswith('validation') else ROOT/'steps'/f"step{state.get('active_step',0):04d}"
rows=[]
for path in (active/'attempts').glob('*/outcomes.jsonl'):
    try:rows.extend(json.loads(x) for x in path.read_text().splitlines())
    except json.JSONDecodeError:pass
out['active_collection']={'finished_attempts':len(rows),'valid_verifiers':sum(r['grade_outcome'].get('valid',False) for r in rows),
                          'file_successes':sum(r['grade_outcome'].get('reward')==1 for r in rows)}
print(json.dumps(out,indent=2))
