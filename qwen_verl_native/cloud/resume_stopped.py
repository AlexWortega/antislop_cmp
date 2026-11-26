"""Restart a run whose two rentals Vast stopped (for example when the account ran out of credit).

Disks survive a stop, but no process does: this starts both rentals, brings back the GPU's self-destruct
guard, its second sshd and reverse tunnel, then the agent host's guard, services and runner, which
resumes verl from the latest checkpoint on the GPU disk. Run it from the laptop: it needs the Vast key.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cloud_launch as c  # noqa: E402

R = c.REMOTE_ROOT + '/qwen_verl_native'
PY = c.REMOTE_ROOT + '/qwen_scaffold_longrun/.venv/bin/python'


def main():
    state = json.loads(c.STATE.read_text())
    vast = c.Vast(c.load_secrets()['VAST_API_KEY'])
    for ident in (state['host_id'], state['gpu_id']):
        print('start', ident, vast.call('PUT', f'/instances/{ident}/', json={'state': 'running'}), flush=True)
    hh, hp = c.wait_ssh(vast, state['host_id'], 'agent host')
    # The GPU dials this endpoint with its own key, which Vast's ssh proxy rejects: use the direct port.
    direct = [e for e in c.endpoints(vast.instance(state['host_id'])) if not e[0].endswith('vast.ai')]
    if direct and c.ssh(*direct[0], 'echo ready', timeout=30).strip().endswith('ready'):
        hh, hp = direct[0]
    gh, gp = c.wait_ssh(vast, state['gpu_id'], 'GPU')
    print('both reachable', (hh, hp), (gh, gp), flush=True)

    # GPU: self-destruct first, then the sshd the agent host uses and the tunnel that exposes it.
    # Each step is skipped when already running, so the script can be rerun after a partial resume.
    if not c.ssh(gh, gp, 'pgrep -f "[g]pu_guard.py" || true').strip():
        c.ssh(gh, gp, 'cd /workspace/native && rm -f STOP agent_heartbeat && (setsid nohup python3 -u gpu_guard.py >> output/gpu_guard.out 2>&1 < /dev/null &)')
        time.sleep(6)
        if '"started"' not in c.ssh(gh, gp, 'tail -1 /workspace/native/output/gpu_guard.jsonl'):
            raise RuntimeError('gpu_guard did not restart')
    print('gpu guard alive', flush=True)
    # A restarted VM gets a fresh authorized_keys from Vast, which drops the GPU's tunnel key.
    c.ssh(hh, hp, 'K=$(cat); grep -qF "$K" /root/.ssh/authorized_keys 2>/dev/null || echo "$K" >> /root/.ssh/authorized_keys',
          stdin=c.ssh(gh, gp, 'cat /root/.ssh/to_agents.pub').strip() + '\n')
    if not c.ssh(gh, gp, 'pgrep -f "[s]shd -f agent_sshd_config" || true').strip():
        c.ssh(gh, gp, 'cd /workspace/native && mkdir -p /run/sshd && $(command -v sshd || echo /usr/sbin/sshd) -f agent_sshd_config', timeout=60)
    if not c.ssh(gh, gp, 'pgrep -f "[r]everse_tunnel.sh" || true').strip():
        c.ssh(gh, gp, f'cd /workspace/native && (setsid nohup bash reverse_tunnel.sh {hh} {hp} >> output/reverse_tunnel.log 2>&1 < /dev/null &)', timeout=60)
    for _ in range(12):
        time.sleep(5)
        reach = c.ssh(hh, hp, 'ssh -i /root/.ssh/antislop_gpu -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p 2222 root@127.0.0.1 echo host-to-gpu-ok 2>&1 | tail -1', timeout=60)
        if 'host-to-gpu-ok' in reach:
            break
    else:
        raise RuntimeError('reverse tunnel did not come back: ' + reach[-200:])
    print('host to gpu: ok', flush=True)

    # Agent host: a stale STOP would end the run at once; the runner resumes from the latest checkpoint.
    script = f'''cd {R}; rm -f STOP REQUEST_STOP CIRCUIT_OPEN; mv -f failure.json failure-before-resume.json 2>/dev/null
set -a; . ./.secrets.env; . ./cloud/run.env; set +a
(setsid nohup python3 -u cloud/agent_guard.py >> cloud/agent_guard.out 2>&1 < /dev/null &)
(setsid nohup {PY} -u coordinator.py >> coordinator.log 2>&1 < /dev/null &)
sleep 3
(setsid nohup {PY} -u guard.py >> guard.log 2>&1 < /dev/null &)
(setsid nohup {PY} -u monitor.py >> monitor.log 2>&1 < /dev/null &)
(setsid nohup sh -c "{PY} -u {R}/rejoin_runner.py; touch {R}/STOP" >> native_runner.log 2>&1 < /dev/null &)
'''
    c.ssh(hh, hp, 'cat > /root/resume_services.sh', stdin=script)
    c.ssh(hh, hp, 'bash /root/resume_services.sh', timeout=60)
    for _ in range(30):
        time.sleep(10)
        if c.ssh(gh, gp, 'test -f /workspace/native/agent_heartbeat && echo yes || echo no').strip() == 'yes':
            break
    else:
        raise RuntimeError('agent host never sent a heartbeat after resume')
    state.update(gpu_ssh=[gh, gp], host_ssh=[hh, hp], resumed_at=time.time())
    c.STATE.write_text(json.dumps(state, indent=2))
    print(json.dumps({'resumed': True, 'watch': f'ssh -i {c.SSH_KEY} -p {hp} root@{hh} /root/status.sh'}), flush=True)


if __name__ == '__main__':
    main()
