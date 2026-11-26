"""Read-only run observer with durable alerts; budget enforcement is separate."""
import json,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(name,default):
 try:return json.loads((ROOT/name).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def emit(event):
 event['time']=time.time()
 with (ROOT/'monitor_events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
 print(json.dumps(event),flush=True)

previous=None
while True:
 state=read('progress.json',{});budget=read('opus_budget.json',{});history=read('history.json',[])
 alerts=[]
 if state.get('status')=='failed':alerts.append(state.get('error','Run failed'))
 if state and time.time()-state.get('updated_at',time.time())>5400:alerts.append('Phase unchanged for more than 90 minutes')
 for item in history:
  u=item['update']
  if u.get('status')!='updated' or not u.get('adapter_probe_changed'):alerts.append('Invalid completed update '+str(item['step']))
 if state.get('completed_steps',0)!=len(history):alerts.append('Step counter/history mismatch (may be mid-write)')
 key=(state.get('status'),state.get('phase'),state.get('active_step'),state.get('completed_steps'),tuple(alerts))
 recent=[x for x in history if x['update'].get('rank')==32][-5:]
 report={'pid':os.getpid(),'checked_at':time.time(),'progress':state,'alerts':alerts,
         'recent_mean_step_minutes':sum(x['seconds'] for x in recent)/60/len(recent) if recent else None,
         'opus_spent_with_margin':budget.get('spent_with_margin'),
         'latest_validation':None}
 evaluations=sorted((ROOT/'evaluations').glob('*/evaluation.json'))
 if evaluations:report['latest_validation']=json.loads(evaluations[-1].read_text())
 temp=ROOT/'monitor_status.json.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(ROOT/'monitor_status.json')
 if key!=previous:emit({'progress':state,'alerts':alerts});previous=key
 if state.get('phase')=='finished':break
 time.sleep(30)
