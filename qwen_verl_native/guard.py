"""Independent run watchdog: records every step, alerts on anomalies, stops the run on hard faults."""
import json,math,os,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'qwen_scaffold_longrun'
LR=float(os.environ.get('ANTISLOP_LR','5e-7'));PRIOR_GPU=float(os.environ.get('PRIOR_GPU_USD',42.02));GPU_CAP=150.;STALL_ALERT=2700;STALL_STOP=5400;VAL_STOP=.55;REWARD_STOP=.35
ANSI=re.compile(r'\x1b\[[0-9;]*m');PAIR=re.compile(r'([\w\-/@.]+):(?:np\.float64\()?(-?[\d.]+(?:e[-+]?\d+)?|nan|inf|-inf)')
def parse(lines):
 """Newest values per step across restarts, from verl's console logger."""
 steps={}
 for line in lines:
  line=ANSI.sub('',line);match=re.search(r'step:(\d+) - ',line)
  if match:steps.setdefault(int(match[1]),{}).update({k:float(v) for k,v in PAIR.findall(line[match.end():])})
 return steps
def rules(steps):
 """Return (stops, alerts) for the parsed metrics."""
 stops=[];alerts=[];train=[(s,m) for s,m in sorted(steps.items()) if 'actor/lr' in m]
 for s,m in train[-1:]:
  if not math.isclose(m['actor/lr'],LR,rel_tol=1e-6):stops.append(f"step {s}: actor/lr={m['actor/lr']} instead of {LR}")
  norm=m.get('actor/grad_norm')
  if norm is not None and (not math.isfinite(norm) or norm<=0):stops.append(f'step {s}: grad_norm={norm}')
  if m.get('actor/entropy',0)>1.5:alerts.append(f"step {s}: entropy={m['actor/entropy']:.3f}")
  if m.get('actor/kl_loss',0)>.02:alerts.append(f"step {s}: kl_loss={m['actor/kl_loss']:.4f}")
 rewards=[m['critic/rewards/mean'] for _,m in train if 'critic/rewards/mean' in m][-20:]
 if len(rewards)==20 and sum(rewards)/20<REWARD_STOP:stops.append(f'mean train reward over 20 steps={sum(rewards)/20:.3f}')
 vals=[(s,m['val-core/harbor/reward/mean@4']) for s,m in sorted(steps.items()) if 'val-core/harbor/reward/mean@4' in m]
 if vals and vals[-1][1]<VAL_STOP:stops.append(f'step {vals[-1][0]}: validation={vals[-1][1]:.3f}')
 return stops,alerts
def trial_health():
 """Timeout and retry rates from the newest local trials."""
 files=sorted((ROOT/'rollouts').glob('*/*/turn_cap_classification.json'),key=lambda p:p.stat().st_mtime)
 outcomes=sorted((ROOT/'rollouts').glob('*/*/outcomes.jsonl'),key=lambda p:p.stat().st_mtime)[-40:]
 timeouts=sum('AgentTimeoutError' in p.read_text() for p in outcomes)
 return {'recent_trials':len(outcomes),'recent_timeouts':timeouts,'startup_retries':sum(1 for _ in (ROOT/'rollouts_startup_failed').glob('*/*')),'rescued_total':len(files)}
def event(data):
 with (ROOT/'guard_events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**data})+'\n')
 print(json.dumps(data),flush=True)
def main():
 sys.path.insert(0,str(OLD));import run_loop as t
 t.ROOT=ROOT;known=-1;last_step=time.time();sent=set();failures=0;halted=False
 def halt(reason):
  nonlocal halted
  if halted:return
  halted=True;event({'STOP':reason});(ROOT/'GUARD_STOP').touch();(ROOT/'REQUEST_STOP').touch()
  try:t.remote('/workspace/longrun/vendor/verl/.venv/bin/ray stop --force',timeout=120)
  except Exception as exc:event({'ALERT':'ray stop failed: '+str(exc)})
 while True:
  state=json.loads((ROOT/'instance.json').read_text()) if (ROOT/'instance.json').exists() else {}
  if state.get('status') in ('destroyed','absent'):event({'rental_gone':state.get('cleanup_reason')});break
  phase=json.loads((ROOT/'progress.json').read_text()).get('phase') if (ROOT/'progress.json').exists() else 'starting'
  try:
   if phase in ('native_verl_training','native_verl_resume'):
    text=t.remote("cd /workspace/native/output; cat continuation-attempt*.log continuation.log 2>/dev/null | grep -aE 'step:[0-9]+ - ' || true",timeout=120);failures=0;sent.discard('ssh')
    steps=parse(text.splitlines());newest=max(steps,default=-1)
    if newest!=known and newest>=0:
     m=steps[newest];known=newest;last_step=time.time()
     event({'step':newest,**{k:round(v,6) for k,v in m.items() if k in ('actor/lr','actor/grad_norm','actor/entropy','actor/kl_loss','actor/pg_loss','critic/rewards/mean','rollout_corr/kl','val-core/harbor/reward/mean@4')},**trial_health()})
    stops,alerts=rules(steps)
    for reason in stops:halt(reason)
    idle=time.time()-last_step
    if idle>STALL_STOP:halt(f'no new step for {idle/60:.0f} min')
    elif idle>STALL_ALERT:alerts.append(f'no new step for {idle/60:.0f} min at step {known}')
    health=trial_health()
    if health['recent_trials']>=20 and health['recent_timeouts']>=health['recent_trials']/10:alerts.append(f"{health['recent_timeouts']}/{health['recent_trials']} recent trials timed out")
    for reason in alerts:
     if reason not in sent:sent.add(reason);event({'ALERT':reason})
   else:last_step=time.time()
   if state.get('created'):
    spent=PRIOR_GPU+(time.time()-state['created'])/3600*state['hourly_bound']
    ledger=json.loads((OLD/'opus_budget.json').read_text());opus=ledger['limit']-ledger['spent_with_margin']-sum(ledger['pending'].values())
    report={'phase':phase,'step':known,'gpu_spent_bound':round(spent,2),'opus_remaining':round(opus,2),'checked_at':time.time()}
    (ROOT/'guard_status.json').write_text(json.dumps(report,indent=2))
    if spent>GPU_CAP-5 and 'gpu' not in sent:sent.add('gpu');event({'ALERT':f'GPU spend bound ${spent:.2f}'})
    if opus<5 and 'opus' not in sent:sent.add('opus');event({'ALERT':f'Opus remaining ${opus:.2f}'})
   for name in ('failure.json','backup_error.json','launch_failure.json'):
    if (ROOT/name).exists() and name not in sent:sent.add(name);event({'ALERT':name+': '+(ROOT/name).read_text()[:400]})
  except Exception as exc:
   failures+=1
   if failures>=3 and 'ssh' not in sent:sent.add('ssh');event({'ALERT':f'Guard check failing: {type(exc).__name__}: {exc}'[:400]})
  time.sleep(60)
if __name__=='__main__':main()
