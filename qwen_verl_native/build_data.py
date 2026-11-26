import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'qwen_scaffold_longrun'
plan=json.loads((OLD/'step_plan.json').read_text());protocol=json.loads((OLD/'protocol.json').read_text())
rows=[]
for step in plan:
 if step['step']<=10:continue
 for g in step['groups']:
  rows.append({'data_source':'harbor','prompt':[{'role':'user','content':'Run the frozen Harbor task '+g['task']}],'agent_name':'harbor_native','extra_info':{'task':g['task'],'agent':g['agent'],'split':'train','policy':'verl-native','group':f"train-step{step['step']:04d}-{g['agent']}-{g['task']}"},'reward_model':{'style':'rule','ground_truth':''}})
val=[]
for task in protocol['tasks']:
 if task['split']!='dev':continue
 for agent in ('pi','hermes'):
  val.append({'data_source':'harbor','prompt':[{'role':'user','content':'Run frozen dev task '+task['task']}],'agent_name':'harbor_native','extra_info':{'task':task['task'],'agent':agent,'split':'dev','policy':'verl-native','group':f"dev-{agent}-{task['task']}"},'reward_model':{'style':'rule','ground_truth':''}})
pd.DataFrame(rows).to_parquet(ROOT/'train.parquet',index=False);pd.DataFrame(val).to_parquet(ROOT/'val.parquet',index=False)
print({'train_prompts':len(rows),'dev_prompts':len(val)})
