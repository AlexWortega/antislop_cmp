"""Concurrent real Harbor trials; keep invalid attempts, replace without relabeling."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def collect(run_dir, groups, policy_id, model, concurrency=12):
    run_dir.mkdir(parents=True,exist_ok=True)
    plan_path = run_dir/'collection_plan.json'
    plan = {'groups':groups,'policy_id':policy_id,'model':model,'concurrency':concurrency}
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Refusing to mix collection policies or plans')
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    def trial(group_index, attempt):
        if (ROOT/'STOP').exists() or (ROOT/'REQUEST_STOP').exists():
            raise RuntimeError('Budget stop requested before trial')
        g = groups[group_index]
        out = run_dir/'attempts'/f'{run_dir.name}-g{group_index}-a{attempt:03d}'
        out.mkdir(parents=True,exist_ok=True)
        cmd = [sys.executable,str(ROOT/'repo/scaffold_training/hermes_collect.py'),
               '--agent',g['agent'],'--agent-version','1.4.0' if g['agent']=='pi' else 'v0.22.5',
               '--preinstalled','--terminal-only','--protocol',str(ROOT/'protocol.json'),
               '--task-root',str(ROOT/'tasks'/g['agent']),'--tasks',g['task'],'--split',g.get('split','train'),
               '--api-base','http://host.docker.internal:18974/v1','--policy-id',policy_id,'--model',model,
               '--samples','1','--max-turns','4','--timeout','480','--setup-timeout','300','--out',str(out)]
        with (out/'collector.log').open('a') as log:
            allowed={'PATH','HOME','LANG','LC_ALL','TMPDIR','DOCKER_HOST','DOCKER_CONFIG','ANTISLOP_SERVE_KEY','SSL_CERT_FILE'}
            env={k:v for k,v in os.environ.items() if k in allowed}
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1000)
        rows = [json.loads(x) for x in (out/'outcomes.jsonl').read_text().splitlines()]
        assert len(rows)==1
        row = rows[0]
        row['repeat'] = attempt
        row['group_index'] = group_index
        return row
    accepted = [[] for _ in groups]
    attempts = [0 for _ in groups]
    all_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        while any(len(rows)<g['samples'] for rows,g in zip(accepted,groups)):
            jobs = {}
            for gi,g in enumerate(groups):
                missing = g['samples']-len(accepted[gi])
                if attempts[gi]+missing > g['samples']+16:
                    raise RuntimeError(f'Group {gi} exceeded 16 replacement attempts; inspect infrastructure')
                for _ in range(missing):
                    attempt = attempts[gi]; attempts[gi]+=1
                    jobs[pool.submit(trial,gi,attempt)] = gi
            for future in concurrent.futures.as_completed(jobs):
                gi = jobs[future]; row = future.result(); all_rows.append(row)
                if row['grade_outcome'].get('valid'):
                    accepted[gi].append(row)
                with (run_dir/'attempt_outcomes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps({'group':gi,'accepted':len(accepted[gi]),'required':groups[gi]['samples'],'trial':row['trial_id'],'grade':row['grade_outcome']}),flush=True)
    accepted_rows = [r for rows in accepted for r in sorted(rows,key=lambda x:x['repeat'])]
    assert len(accepted_rows)==sum(g['samples'] for g in groups)
    (run_dir/'outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in accepted_rows))
    (run_dir/'attempt_outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
    summary = {'attempts':len(all_rows),'accepted':len(accepted_rows),'invalid_attempts':sum(not r['grade_outcome']['valid'] for r in all_rows),
               'file_successes':sum(r['grade_outcome'].get('reward')==1 for r in accepted_rows),'policy_id':policy_id}
    (run_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--policy-id',required=True);ap.add_argument('--model',required=True);ap.add_argument('--concurrency',type=int,default=12);a=ap.parse_args()
    print(json.dumps(collect(a.out,json.loads(a.plan.read_text()),a.policy_id,a.model,a.concurrency),indent=2))
