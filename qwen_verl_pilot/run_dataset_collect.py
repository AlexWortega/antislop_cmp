import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

root=Path(__file__).resolve().parent
repo=root/'scaffold_training'
p=argparse.ArgumentParser();p.add_argument('--agent',choices=['pi','hermes'],required=True);p.add_argument('--phase',choices=['before_train','after'],default='before_train');p.add_argument('--tag',default='');p.add_argument('--model',default='antislop');p.add_argument('--policy-id',default='antislop-395c2c-base');a=p.parse_args()
env=os.environ.copy();env.update(ANTISLOP_SERVE_KEY=Path('/tmp/antislop-pilot-serve.key').read_text().strip(),DOCKER_HOST='unix:///Users/alexwortega/.colima/antislop-pilot/docker.sock',DOCKER_CONFIG='/tmp/antislop-docker-config')
image='antislop-pilot-'+a.agent+':local'
image_id=subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],env=env,text=True).strip()
task_root=repo/'private'/('dataset_tasks_'+a.agent)
shutil.copytree(repo/'private/dataset_tasks',task_root,dirs_exist_ok=True)
for dockerfile in task_root.glob('*/environment/Dockerfile'):
    dockerfile.write_text(f'FROM {image}\nWORKDIR /app\nCMD ["bash"]\n')
jobs=[('baseline','dev',1),('train','train',2)] if a.phase=='before_train' else [('after','dev',1)]
for label,split,samples in jobs:
    deadline=json.loads((root/'instance.json').read_text())['deadline']
    if time.time()>deadline-600:raise SystemExit('Stopping collection with ten minutes left for artifact retrieval')
    out=repo/'private'/('dataset-'+label+'-'+a.agent+a.tag);out.mkdir(parents=True,exist_ok=True)
    (out/'runtime.json').write_text(json.dumps({'image':image,'image_id':image_id,'policy_id':a.policy_id,'source_commit':'959e3566727d2fbb6cbfccc47691c15dcdfe5a2c','preinstalled':True,'hermes_toolsets':'terminal' if a.agent=='hermes' else None},indent=2))
    cmd=[str(repo/'.venv/bin/python'),'scaffold_training/hermes_collect.py','--agent',a.agent,
         '--preinstalled','--terminal-only','--agent-version','1.4.0' if a.agent=='pi' else 'v0.22.5',
         '--protocol',str(root/'dataset_protocol.json'),'--task-root',str(task_root),'--split',split,
         '--api-base','http://host.docker.internal:18972/v1','--policy-id',a.policy_id,'--model',a.model,
         '--samples',str(samples),'--max-turns','4','--timeout','180','--setup-timeout','300','--out',str(out)]
    with (out/'collector.log').open('a') as log:
        subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(label,a.agent,json.loads((out/'summary.json').read_text()),flush=True)
