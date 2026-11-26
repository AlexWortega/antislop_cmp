"""Bounded on-policy Harbor -> Opus -> GRPO -> checkpoint -> fresh rollout loop."""
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
import requests
from collect_batch import collect
from score_groups import score, JudgeBatchError

ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/longrun'
PYTHON=sys.executable
STATE={'status':'starting','completed_steps':0,'phase':'starting'}
STOP_EVENT=threading.Event()
SSH_KEY=os.environ.get('ANTISLOP_SSH_KEY','/Users/alexwortega/.ssh/qwen_verl_pilot_20261101')

def save(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');os.replace(temp,path)

def progress(**updates):
    STATE.update(updates,updated_at=time.time());save(ROOT/'progress.json',STATE)
    print(json.dumps(STATE),flush=True)

def heartbeat():
    while not STOP_EVENT.is_set():
        save(ROOT/'heartbeat.json',{'time':time.time(),'pid':os.getpid(),'phase':STATE['phase']})
        STOP_EVENT.wait(20)

def ssh_args():
    state=json.loads((ROOT/'instance_status.json').read_text())
    return ['ssh','-i',SSH_KEY,'-o','StrictHostKeyChecking=accept-new','-o','ServerAliveInterval=20','-o','ConnectTimeout=20','-p',str(state['ssh_port']),'root@'+state['ssh_host']]

def remote(command,timeout=120):
    result=subprocess.run(ssh_args()+[command],capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError(f'Remote command failed ({result.returncode}): '+result.stderr[-1000:])
    return result.stdout

def rsync(remote_path,local_path,upload=False,exclude=()):
    state=json.loads((ROOT/'instance_status.json').read_text())
    shell=shlex.join(ssh_args()[:-1])
    host='root@'+state['ssh_host']
    remote_spec=host+':'+remote_path
    command=['rsync','-az','--partial','-e',shell]
    for pattern in exclude:command+=['--exclude',pattern]
    command += [str(local_path),remote_spec] if upload else [remote_spec,str(local_path)]
    subprocess.run(command,check=True,timeout=1800)

def budget_remaining():
    budget=json.loads((ROOT/'opus_budget.json').read_text())
    return budget['limit']-budget['spent_with_margin']-sum(budget['pending'].values())

def available_credit():
    path=ROOT/'credit_status.json'
    cached=json.loads(path.read_text()) if path.exists() else None
    if cached and time.time()-cached['checked_at']<180:return cached['remaining']
    try:
        response=requests.get('https://openrouter.ai/api/v1/credits',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']},timeout=30)
        response.raise_for_status();data=response.json()['data']
        remaining=float(data['total_credits'])-float(data['total_usage'])
        save(path,{'remaining':remaining,'checked_at':time.time(),'ledger_remaining':budget_remaining()})
        return remaining
    except requests.RequestException:
        if cached:return max(0.,cached['remaining']-(cached['ledger_remaining']-budget_remaining()))
        raise

def should_stop(expected_seconds=1800,expected_api=2.):
    rental=json.loads((ROOT/'instance.json').read_text())
    return ((ROOT/'REQUEST_STOP').exists() or (ROOT/'STOP').exists() or
            rental['status'] not in ('created','running') or
            time.time()+expected_seconds+1800>=rental['deadline'] or
            budget_remaining()<expected_api+5. or available_credit()<expected_api+5.)

def stop_server():
    command="python3 - <<'PY'\nimport os,signal\nfrom pathlib import Path\np=Path('/workspace/longrun/server.pid')\nif p.exists():\n try:os.kill(int(p.read_text()),signal.SIGTERM)\n except ProcessLookupError:pass\n p.unlink()\nPY"
    remote(command)
    for _ in range(30):
        used=remote('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits').strip()
        if int(used)<1024:return
        time.sleep(2)
    raise RuntimeError('Inference did not release GPU memory')

def start_server(policy,adapter):
    log=REMOTE+'/output/serve-'+policy+'.log'
    command='nohup bash '+REMOTE+'/serve_gpu.sh '+shlex.quote(policy)+' '+shlex.quote(adapter)+' > '+shlex.quote(log)+' 2>&1 < /dev/null &'
    remote(command)
    headers={'Authorization':'Bearer '+os.environ['ANTISLOP_SERVE_KEY']}
    for _ in range(150):
        try:
            result=requests.get('http://127.0.0.1:18974/v1/models',headers=headers,timeout=3)
            if result.ok:
                models={x['id'] for x in result.json()['data']}
                if ('antislop-scaffold' if adapter else 'antislop') in models:return
        except requests.RequestException:pass
        time.sleep(4)
    raise RuntimeError('vLLM startup failed; inspect '+log)

def prepare_and_score(run_dir):
    audit=ROOT/'audit';audit.mkdir(exist_ok=True)
    if not (run_dir/'prepared/manifest.json').exists():
        rsync(REMOTE+'/audit/',str(audit)+'/')
    prepared=run_dir/'prepared'
    if not (prepared/'manifest.json').exists():
        subprocess.run([PYTHON,str(ROOT/'repo/scaffold_training/prepare.py'),'--outcomes',str(run_dir/'outcomes.jsonl'),
                        '--audit-dir',str(audit),'--protocol',str(ROOT/'protocol.json'),'--out',str(prepared)],check=True)
    manifest=json.loads((prepared/'manifest.json').read_text())
    if manifest['rl_tokenized_trajectories']!=manifest['attempts']:
        raise RuntimeError('Exact sampled log-probabilities are missing; no update allowed')
    scored=run_dir/'scored'
    if not (scored/'summary.json').exists():score(prepared/'groups.jsonl',run_dir,scored)
    return json.loads((scored/'summary.json').read_text())

def checkpoint(step):
    name=f'step{step:04d}';path=ROOT/'checkpoints'/name;path.mkdir(parents=True,exist_ok=True)
    rsync(REMOTE+'/checkpoints/'+name+'/',str(path)+'/')
    expected={}
    for line in remote('cd '+REMOTE+'/checkpoints/'+name+' && sha256sum adapter_model.safetensors optimizer.pt').splitlines():
        digest,filename=line.split();expected[filename]=digest
    for filename,digest in expected.items():
        h=hashlib.sha256()
        with (path/filename).open('rb') as f:
            for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
        if h.hexdigest()!=digest:raise RuntimeError('Checkpoint transfer hash mismatch')
    save(path/'transfer_verification.json',expected)
    # Preserve the server config before making the downloaded copy portable.
    config=json.loads((path/'adapter_config.json').read_text());save(path/'adapter_config.server.json',config)
    config.update(base_model_name_or_path='rawmodels/Qwen3.8-27B-antislop',revision='395c2c057ac4d7d6fa333dd9774a652039ebddb6')
    save(path/'adapter_config.json',config)
    subprocess.run(['chown','-R','alexwortega:staff',str(path)],check=True)
    progress(last_backed_up_step=step)

def evaluate(step,policy,adapter):
    phase=ROOT/'evaluations'/f'eval{step:04d}'
    tasks=[r for r in json.loads((ROOT/'protocol.json').read_text())['tasks'] if r['split']=='dev']
    groups=[{'agent':agent,'task':r['task'],'samples':4,'split':'dev'} for agent in ('pi','hermes') for r in tasks]
    progress(phase='validation_collection',evaluation_step=step)
    if not (phase/'outcomes.jsonl').exists():
        start_server(policy,adapter)
        try:collect(phase,groups,policy,'antislop-scaffold' if adapter else 'antislop',concurrency=12)
        finally:stop_server()
    progress(phase='validation_scoring',evaluation_step=step)
    try:summary=prepare_and_score(phase)
    except JudgeBatchError as exc:
        save(phase/'skipped.json',{'reason':str(exc),'step':step,'time':time.time()})
        print('VALIDATION_SKIPPED '+str(exc),flush=True)
        return None
    save(phase/'evaluation.json',{'step':step,**summary})
    print('VALIDATION '+json.dumps({'step':step,**summary}),flush=True)
    return summary

def cleanup(tunnel):
    progress(phase='cleanup')
    try:
        if STATE['completed_steps'] and STATE.get('last_backed_up_step')!=STATE['latest_checkpoint_step']:
            checkpoint(STATE['latest_checkpoint_step'])
        (ROOT/'gpu_output').mkdir(exist_ok=True)
        rsync(REMOTE+'/output/',str(ROOT/'gpu_output')+'/')
    finally:
        (ROOT/'STOP').write_text('Orchestrator finished; destroy owned rental.\n')
        if tunnel is not None:
            tunnel.terminate()
        subprocess.run(['sudo','-u','alexwortega','/opt/homebrew/bin/colima','stop','antislop-pilot'],timeout=120)
        subprocess.run(['sudo','-u','alexwortega','/opt/homebrew/bin/docker','context','use','default'],timeout=30)
        Path('/tmp/antislop-longrun-serve.key').unlink(missing_ok=True)

def main():
    if (ROOT/'progress.json').exists() and not (ROOT/'resume.json').exists():raise SystemExit('Existing loop state: inspect before resuming')
    resume=json.loads((ROOT/'resume.json').read_text()) if (ROOT/'resume.json').exists() else None
    if resume:STATE.update(json.loads((ROOT/'progress.json').read_text()))
    config=json.loads((ROOT/'run.request.json').read_text())
    assert config['lora']['rank']==32 and config['lora']['alpha']==512
    assert config['budget_usd']=={'gpu':150,'opus':120}
    os.environ['ANTISLOP_SERVE_KEY']=Path('/tmp/antislop-longrun-serve.key').read_text().strip()
    os.environ['DOCKER_HOST']='unix:///Users/alexwortega/.colima/antislop-pilot/docker.sock'
    os.environ['DOCKER_CONFIG']='/tmp/antislop-longrun-docker-config'
    os.environ['HF_HOME']=str(ROOT/'hf_cache')
    heartbeat_thread=threading.Thread(target=heartbeat,daemon=True);heartbeat_thread.start()
    tunnel=None;failure=None
    try:
        if remote('test -f '+REMOTE+'/BOOTSTRAP_DONE && echo ready').strip()!='ready':raise RuntimeError('Bootstrap not ready')
        tunnel=subprocess.Popen(ssh_args()[:-1]+['-N','-L','0.0.0.0:18974:127.0.0.1:8000',ssh_args()[-1]],stdout=(ROOT/'tunnel.log').open('w'),stderr=subprocess.STDOUT)
        time.sleep(2)
        if tunnel.poll() is not None:raise RuntimeError('SSH tunnel exited')
        progress(status='running',phase='baseline')
        baseline=evaluate(0,'longrun-base','')
        if baseline is not None and baseline['file_successes']<baseline['attempts']*0.8:raise RuntimeError('Baseline completion below 80%; inspect before training')
        plan=json.loads((ROOT/'step_plan.json').read_text());history=json.loads((ROOT/'history.json').read_text()) if resume else [];adapter=resume['adapter'] if resume else '';policy=resume['policy'] if resume else 'longrun-base';consecutive_skips=0
        for item in plan:
            step=item['step'];name=f'step{step:04d}'
            if resume and step<=resume['last_plan_step']:continue
            recent=history[-5:]
            expected_seconds=max(1800,sum(x['seconds'] for x in recent)/len(recent)*1.3) if recent else 3600
            expected_api=max(1.,sum(x['api_spent'] for x in recent)/len(recent)*1.3) if recent else 3.
            if should_stop(expected_seconds,expected_api):
                progress(status='budget_stop',phase='budget_stop',stop_before_step=step);break
            started=time.time();api_before=budget_remaining();run_dir=ROOT/'steps'/name
            progress(phase='rollout',active_step=step)
            start_server(policy,adapter)
            try:collection=collect(run_dir,item['groups'],policy,'antislop-scaffold' if adapter else 'antislop',concurrency=12)
            finally:stop_server()
            progress(phase='scoring',active_step=step)
            try:summary=prepare_and_score(run_dir)
            except JudgeBatchError as exc:
                consecutive_skips+=1
                save(run_dir/'skipped.json',{'reason':str(exc),'time':time.time(),'policy':policy,'optimizer_updated':False})
                progress(phase='batch_skipped',skipped_batches=STATE.get('skipped_batches',0)+1,last_skip=str(exc))
                if consecutive_skips>=3:raise RuntimeError('Three consecutive invalid judge batches; stop to protect budget')
                continue
            scored_groups=[json.loads(line) for line in (run_dir/'scored/groups.jsonl').read_text().splitlines()]
            if all(len({t['outcome']['reward'] for t in g['trajectories']})<=1 for g in scored_groups):
                save(run_dir/'skipped.json',{'reason':'all_groups_zero_reward_variance','optimizer_updated':False})
                progress(phase='batch_skipped',skipped_batches=STATE.get('skipped_batches',0)+1)
                consecutive_skips+=1
                if consecutive_skips>=3:raise RuntimeError('Three consecutive unusable batches')
                continue
            consecutive_skips=0
            remote('mkdir -p '+REMOTE+'/steps/'+name+'/scored')
            rsync(REMOTE+'/steps/'+name+'/scored/groups.jsonl',run_dir/'scored/groups.jsonl',upload=True)
            progress(phase='optimizer_step',active_step=step)
            lr=2e-7*min(1.,(len(history)+1)/5.)
            command='bash '+REMOTE+'/update_gpu.sh '+shlex.join([name,policy,adapter,str(lr)])+' > '+REMOTE+'/output/train-'+name+'.log 2>&1'
            remote(command,timeout=7200)
            report=json.loads(remote('cat '+REMOTE+'/checkpoints/'+name+'/update.json'))
            assert report['status']=='updated' and report['optimizer_steps']==1 and report['adapter_probe_changed']
            save(run_dir/'update.json',report)
            adapter=REMOTE+'/checkpoints/'+name;policy='longrun-'+name
            progress(completed_steps=len(history)+1,latest_checkpoint_step=step,phase='step_complete',latest_checkpoint=adapter)
            entry={'step':step,'seconds':time.time()-started,'api_spent':api_before-budget_remaining(),'collection':collection,'metrics':summary,'update':report}
            history.append(entry);save(ROOT/'history.json',history)
            if len(history)==1 or step%10==0:checkpoint(step)
            if step%10==0 and not should_stop(1200,3.):evaluate(step,policy,adapter)
            # Retain every tenth full checkpoint and the latest two; old adapter
            # snapshots remain, while large intermediate optimizer states expire.
            if step>2 and (step-2)%10 and step-2!=1:
                remote('rm -f '+REMOTE+f'/checkpoints/step{step-2:04d}/optimizer.pt')
        else:progress(status='completed',phase='training_complete')
        if STATE['completed_steps']%10 and not should_stop(1200,3.):evaluate(STATE['latest_checkpoint_step'],policy,adapter)
    except BaseException as exc:
        failure=exc;progress(status='budget_stop' if (ROOT/'REQUEST_STOP').exists() else 'failed',phase='error',error=type(exc).__name__+': '+str(exc))
    finally:
        try:cleanup(tunnel)
        except BaseException as exc:
            (ROOT/'STOP').touch()
            progress(cleanup_error=type(exc).__name__+': '+str(exc))
        STOP_EVENT.set();heartbeat_thread.join(timeout=5)
        progress(phase='finished',finished_at=time.time())
        subprocess.run(['chown','-R','alexwortega:staff',str(ROOT)],check=False)
    if failure is not None:raise failure

if __name__=='__main__':main()
