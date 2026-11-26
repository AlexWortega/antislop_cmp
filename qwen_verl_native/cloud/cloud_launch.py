"""Rent a GPU and an agent host on Vast, hand the run to the agent host, then let the laptop go.

The laptop is needed only until both guards are confirmed alive. Before that point any failure
destroys both rentals. After it, the GPU destroys itself (gpu_guard) and the agent host stops
itself (agent_guard); the full Vast key never leaves the laptop.

Usage: cloud_launch.py --gpu-hours 2 --steps 3 [--dry-run]
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent
LONGRUN = NATIVE.parent / 'qwen_scaffold_longrun'
VERL = NATIVE.parent / 'qwen_verl_pilot' / 'vendor' / 'verl'
STATE = HERE / 'launch_state.json'
SSH_KEY = '/Users/alexwortega/.ssh/qwen_verl_pilot_20261101'
BASE = 'https://console.vast.ai/api/v0'
REMOTE_ROOT = '/root/antislop_cmp'
# Run state from earlier runs must not travel to the new agent host.
NATIVE_EXCLUDE = ('runs', 'checkpoints', 'milestone_backups', 'rollouts', 'rollouts_startup_failed', 'output', 'audit',
                  '__pycache__', '.git', '*.log', '*.jsonl', 'STOP', 'GUARD_STOP', 'REQUEST_STOP', 'instance.json',
                  'instance_status.json', 'progress.json', 'heartbeat.json', 'failure.json', 'launch_failure.json',
                  'backup_error.json', 'guard_status.json', 'monitor_status.json', 'offers.json',
                  'rental_cleanup_verification.json', '.secrets.env', 'cloud/launch_state.json', 'cloud/SETUP_DONE',
                  'native_step_verification.json')
LONGRUN_EXCLUDE = ('.venv', 'checkpoints', 'steps', 'gpu_output', 'evaluations', 'restarts', 'audit', '__pycache__',
                   '*.log', 'STOP', 'instance.json', 'instance_status.json', 'heartbeat.json', 'progress.json')


def load_secrets():
    env = {}
    for line in (NATIVE / '.secrets.env').read_text().splitlines():
        key, _, value = line.partition('=')
        if key.strip():
            env[key.strip()] = value.strip()
    return env


class Vast:
    def __init__(self, key):
        self.head = {'Authorization': 'Bearer ' + key}

    def call(self, method, path, retries=40, **kw):
        # The laptop's network drops at times; keep trying for about ten minutes before giving up.
        for attempt in range(retries):
            try:
                r = requests.request(method, BASE + path, headers=self.head, timeout=60, **kw)
            except requests.RequestException as exc:
                print(f'network retry {attempt + 1}: {type(exc).__name__}', flush=True)
                time.sleep(15)
                continue
            if r.status_code >= 500 or r.status_code == 429:
                time.sleep(15)
                continue
            if not r.ok:
                raise RuntimeError(f'Vast HTTP {r.status_code} for {method} {path}: {r.text[:300]}')
            return r.json()
        raise RuntimeError(f'Vast unreachable for {method} {path}')

    def offers(self, query):
        body = {'verified': {'eq': True}, 'rentable': {'eq': True}, 'type': 'on-demand',
                'order': [['dph_total', 'asc']], 'limit': 300, **query}
        return self.call('POST', '/bundles/', json=body)['offers']

    def instance(self, ident):
        return next((x for x in self.call('GET', '/instances/')['instances'] if x['id'] == ident), None)

    def destroy(self, ident):
        return self.call('DELETE', f'/instances/{ident}/')


def pick_gpu(vast):
    offers = [x for x in vast.offers({'num_gpus': {'eq': 1}, 'gpu_ram': {'gte': 140000}})
              if x['gpu_name'] in ('H200', 'H200 NVL') and x.get('reliability2', 0) >= .97 and x['dph_total'] <= 4.5
              and x.get('inet_down', 0) >= 500 and x.get('disk_space', 0) >= 250 and x.get('cuda_max_good', 0) >= 13
              and x.get('cpu_ram', 0) >= 150000]
    if not offers:
        raise SystemExit('No H200 offer meets the filters')
    return min(offers, key=lambda x: x['dph_total'])


def pick_hosts(vast, exclude=()):
    offers = [x for x in vast.offers({'vms_enabled': {'eq': True}})
              if x.get('vms_enabled') and x.get('cpu_cores_effective', 0) >= 16 and x.get('cpu_ram', 0) >= 128000
              and x.get('reliability2', 0) >= .97 and x['dph_total'] <= .6 and x.get('inet_up', 0) >= 300
              and x.get('inet_down', 0) >= 300 and x.get('disk_space', 0) >= 150 and x.get('machine_id') not in exclude]
    if not offers:
        raise SystemExit('No VM-capable agent host meets the filters')
    return sorted(offers, key=lambda x: x['dph_total'])


def bound(offer, disk):
    """Hourly price with storage, plus 10% margin, as vast_controller computed it."""
    return max(offer['dph_total'], offer['dph_base'] + offer['storage_cost'] * disk / 720) * 1.1


def ssh(host, port, command, timeout=120, stdin=None):
    args = ['ssh', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=20',
            '-o', 'ServerAliveInterval=20', '-p', str(port), 'root@' + host, command]
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, input=stdin)
    if r.returncode:
        raise RuntimeError(f'ssh {host}:{port} failed ({r.returncode}): {r.stderr[-400:]}')
    return r.stdout


def rsync(host, port, source, dest, exclude=()):
    shell = shlex.join(['ssh', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=accept-new', '-p', str(port)])
    cmd = ['rsync', '-az', '--partial', '-e', shell]
    for pattern in exclude:
        cmd += ['--exclude', pattern]
    subprocess.run(cmd + [str(source), f'root@{host}:{dest}'], check=True, timeout=3600)


def endpoints(info):
    # Vast's ssh proxy does not always serve VMs; the direct port mapping is the fallback.
    found = [(info['ssh_host'], info['ssh_port'])] if info.get('ssh_host') else []
    direct = (info.get('ports') or {}).get('22/tcp') or []
    if info.get('public_ipaddr') and direct:
        found.append((info['public_ipaddr'].strip(), int(direct[0]['HostPort'])))
    return found


def wait_ssh(vast, ident, label, minutes=40):
    for _ in range(minutes * 6):
        info = vast.instance(ident)
        if info and info.get('actual_status') == 'running':
            for host, port in endpoints(info):
                try:
                    if ssh(host, port, 'echo ready', timeout=30).strip().endswith('ready'):
                        return host, port
                except Exception:
                    pass
        time.sleep(10)
    raise RuntimeError(f'{label} never became reachable over ssh')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu-hours', type=float, required=True)
    p.add_argument('--host-extra-hours', type=float, default=2.0, help='agent host lifetime past the GPU deadline')
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--lr', default='4e-6')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    secrets = load_secrets()
    vast = Vast(secrets['VAST_API_KEY'])
    bad_hosts = set(json.loads((HERE / 'bad_hosts.json').read_text())) if (HERE / 'bad_hosts.json').exists() else set()
    gpu_offer, host_offers = pick_gpu(vast), pick_hosts(vast, bad_hosts)
    host_offer = host_offers[0]
    gpu_disk, host_disk = 250, 150
    plan = {'gpu': {k: gpu_offer.get(k) for k in ('id', 'gpu_name', 'dph_total', 'reliability2', 'geolocation')},
            'host': {k: host_offer.get(k) for k in ('id', 'gpu_name', 'dph_total', 'cpu_cores_effective', 'cpu_ram', 'geolocation')},
            'gpu_bound_usd': round(bound(gpu_offer, gpu_disk) * a.gpu_hours, 2),
            'host_bound_usd': round(bound(host_offer, host_disk) * (a.gpu_hours + a.host_extra_hours), 2)}
    print(json.dumps(plan, indent=2), flush=True)
    if a.dry_run:
        return
    if STATE.exists() and json.loads(STATE.read_text()).get('active'):
        raise SystemExit('A launch is already recorded in cloud/launch_state.json: inspect it first')
    state = {'active': True, 'plan': plan}
    STATE.write_text(json.dumps(state, indent=2))
    created = time.time()
    handed_off = False
    try:
        for candidate in host_offers[:3]:
            host = vast.call('PUT', f"/asks/{candidate['id']}/", retries=1, json={
                'image': 'docker.io/vastai/kvm:ubuntu_terminal', 'disk': host_disk, 'runtype': 'ssh', 'target_state': 'running',
                'cancel_unavail': True, 'label': f'antislop-agents-{int(created)}'})
            state['host_id'] = host['new_contract']
            STATE.write_text(json.dumps(state, indent=2))
            try:
                hh, hp = wait_ssh(vast, state['host_id'], 'agent host', minutes=15)
                host_offer = candidate
                break
            except RuntimeError:
                print('agent host never booted, trying the next offer', flush=True)
                bad_hosts.add(candidate.get('machine_id'))
                (HERE / 'bad_hosts.json').write_text(json.dumps(sorted(x for x in bad_hosts if x)))
                vast.call('DELETE', f"/instances/{state['host_id']}/", retries=200)
                state['host_id'] = None
        else:
            raise RuntimeError('No agent host booted')
        print('agent host reachable', flush=True)
        # Checked before the GPU is rented: a host that cannot stop itself is not worth starting.
        probe = ssh(hh, hp, 'docker info --format "{{.ServerVersion}}" 2>&1 | head -1; nproc; free -g | sed -n 2p')
        has_key = ssh(hh, hp, '(grep -q "^CONTAINER_API_KEY=." /etc/environment || [ -n "${CONTAINER_API_KEY:-}" ]) && echo yes || echo no').strip()
        state['host_probe'] = probe + 'instance_key=' + has_key
        STATE.write_text(json.dumps(state, indent=2))
        print('agent host probe:', state['host_probe'].replace('\n', ' | '), flush=True)
        if has_key != 'yes':
            raise RuntimeError('Agent host has no CONTAINER_API_KEY, so it could not stop itself')
        created = time.time()
        # Offers move while the agent host boots; take the cheapest one available now.
        gpu_offer = pick_gpu(vast)
        print('gpu offer', gpu_offer['id'], gpu_offer['gpu_name'], round(gpu_offer['dph_total'], 2), flush=True)
        gpu = vast.call('PUT', f"/asks/{gpu_offer['id']}/", retries=1, json={
            'image': 'nvidia/cuda:13.0.2-devel-ubuntu24.04', 'disk': gpu_disk, 'runtype': 'ssh', 'target_state': 'running',
            'cancel_unavail': True, 'label': f'antislop-gpu-{int(created)}', 'onstart': 'mkdir -p /workspace/native/output'})
        state['gpu_id'] = gpu['new_contract']
        STATE.write_text(json.dumps(state, indent=2))
        gpu_deadline = created + a.gpu_hours * 3600
        host_deadline = gpu_deadline + a.host_extra_hours * 3600

        gh, gp = wait_ssh(vast, state['gpu_id'], 'GPU')
        print('gpu reachable', flush=True)
        # Self-destruct first: from here on the GPU can stop its own billing.
        ssh(gh, gp, 'mkdir -p /workspace/native/output && (command -v python3 || (apt-get update -qq && apt-get install -y -qq python3)) >/dev/null 2>&1', timeout=600)
        ssh(gh, gp, 'cat > /workspace/native/guard_config.json', stdin=json.dumps({'deadline': gpu_deadline}))
        rsync(gh, gp, HERE / 'gpu_guard.py', '/workspace/native/gpu_guard.py')
        ssh(gh, gp, 'nohup python3 -u /workspace/native/gpu_guard.py > /workspace/native/output/gpu_guard.out 2>&1 < /dev/null &')
        time.sleep(8)
        guard_log = ssh(gh, gp, 'cat /workspace/native/output/gpu_guard.jsonl 2>/dev/null; pgrep -fc gpu_guard.py || true')
        if '"started"' not in guard_log:
            raise RuntimeError('gpu_guard did not start: ' + guard_log[-300:])
        print('gpu guard alive', flush=True)
        ssh(hh, hp, f'mkdir -p {REMOTE_ROOT}/qwen_verl_native {REMOTE_ROOT}/qwen_scaffold_longrun {REMOTE_ROOT}/qwen_verl_pilot/vendor/verl /root/.ssh'
                    ' && ([ -f /root/.ssh/antislop_gpu ] || ssh-keygen -q -t ed25519 -N "" -f /root/.ssh/antislop_gpu)')
        public = ssh(hh, hp, 'cat /root/.ssh/antislop_gpu.pub').strip()
        ssh(gh, gp, 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys', stdin=public + '\n')
        rsync(hh, hp, str(NATIVE) + '/', f'{REMOTE_ROOT}/qwen_verl_native/', exclude=NATIVE_EXCLUDE)
        rsync(hh, hp, str(LONGRUN) + '/', f'{REMOTE_ROOT}/qwen_scaffold_longrun/', exclude=LONGRUN_EXCLUDE)
        rsync(hh, hp, str(VERL) + '/', f'{REMOTE_ROOT}/qwen_verl_pilot/vendor/verl/', exclude=('.git', '.venv', '__pycache__'))
        # score_groups imports the unslop reward from the pilot's dataset sources.
        rsync(hh, hp, str(NATIVE.parent / 'qwen_verl_pilot' / 'dataset_sources'), f'{REMOTE_ROOT}/qwen_verl_pilot/')
        files = {
            '.secrets.env': f"OPENROUTER_API_KEY={secrets['OPENROUTER_API_KEY']}\n",
            'instance.json': json.dumps({'status': 'created', 'id': state['gpu_id'], 'created': created,
                                         'hourly_bound': bound(gpu_offer, gpu_disk), 'deadline': gpu_deadline, 'gpu_budget': 150.}),
            'instance_status.json': json.dumps({'id': state['gpu_id'], 'ssh_host': gh, 'ssh_port': gp}),
            'cloud/vm.json': json.dumps({'id': state['host_id'], 'deadline': host_deadline, 'keep_after_run': 3600}),
            'cloud/run.env': '\n'.join([
                'ANTISLOP_SSH_KEY=/root/.ssh/antislop_gpu', 'ANTISLOP_COORDINATOR_PORT=18976',
                'ANTISLOP_DOCKER_HOST=unix:///var/run/docker.sock',
                f'ANTISLOP_EXTRA_COMPOSE={REMOTE_ROOT}/qwen_verl_native/cloud/host_gateway.compose.yaml',
                'ANTISLOP_GROUP_SIZE=8', 'ANTISLOP_TRIAL_SLOTS=32', 'ANTISLOP_STOP_BUFFER=900', f'ANTISLOP_LR={a.lr}', 'ANTISLOP_SKIP_PROOF=1',
                'ANTISLOP_SKIP_RESUME_VAL=1', f'ANTISLOP_TOTAL_STEPS={a.steps}', 'PRIOR_GPU_USD=0']) + '\n'}
        # Vast's ssh proxy accepts only account keys, so the GPU dials out to the agent host instead.
        ssh(gh, gp, 'mkdir -p ~/.ssh && ([ -f ~/.ssh/to_agents ] || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/to_agents) && (command -v ssh || (apt-get install -y -qq openssh-client)) >/dev/null', timeout=300)
        ssh(hh, hp, 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys', stdin=ssh(gh, gp, 'cat ~/.ssh/to_agents.pub').strip() + '\n')
        rsync(gh, gp, HERE / 'reverse_tunnel.sh', '/workspace/native/reverse_tunnel.sh')
        rsync(gh, gp, HERE / 'agent_sshd_config', '/workspace/native/agent_sshd_config')
        ssh(gh, gp, 'cd /workspace/native && cat > agent_authorized_keys && chmod 600 agent_authorized_keys'
                    ' && ([ -f agent_sshd_host_key ] || ssh-keygen -q -t ed25519 -N "" -f agent_sshd_host_key)'
                    ' && mkdir -p /run/sshd && $(command -v sshd || echo /usr/sbin/sshd) -f agent_sshd_config && sleep 1 && cat agent_sshd.pid',
            stdin=public + '\n', timeout=60)
        ssh(gh, gp, f'setsid nohup bash /workspace/native/reverse_tunnel.sh {hh} {hp} > /workspace/native/output/reverse_tunnel.log 2>&1 < /dev/null &', timeout=60)
        for _ in range(12):
            time.sleep(5)
            reach = ssh(hh, hp, 'ssh -i /root/.ssh/antislop_gpu -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p 2222 root@127.0.0.1 echo host-to-gpu-ok 2>&1 | tail -2', timeout=60)
            if 'host-to-gpu-ok' in reach:
                break
        else:
            raise RuntimeError('Agent host cannot reach the GPU through the reverse tunnel: ' + reach[-300:]
                               + ' | tunnel log: ' + ssh(gh, gp, 'tail -5 /workspace/native/output/reverse_tunnel.log')[-300:])
        print('host to gpu: ok through the reverse tunnel', flush=True)
        files['instance_status.json'] = json.dumps({'id': state['gpu_id'], 'ssh_host': '127.0.0.1', 'ssh_port': 2222})
        for name, text in files.items():
            ssh(hh, hp, f'umask 077; cat > {REMOTE_ROOT}/qwen_verl_native/{name}', stdin=text)
        # Only the detached script goes to the background; a backgrounded `cd && ...` list keeps ssh's stdout open.
        ssh(hh, hp, f'cd {REMOTE_ROOT}/qwen_verl_native; setsid nohup bash cloud/vm_start.sh > cloud/vm_start.out 2>&1 < /dev/null &', timeout=60)
        # Hand-off is done once the agent host has reached the GPU on its own key.
        for _ in range(30):
            time.sleep(10)
            beat = ssh(gh, gp, 'test -f /workspace/native/agent_heartbeat && echo yes || echo no').strip()
            if beat == 'yes':
                break
        else:
            raise RuntimeError('Agent host never sent a heartbeat to the GPU')
        handed_off = True
        state.update(gpu_ssh=[gh, gp], host_ssh=[hh, hp], gpu_deadline=gpu_deadline, host_deadline=host_deadline, handed_off=True)
        STATE.write_text(json.dumps(state, indent=2))
        print(json.dumps({'handed_off': True, 'gpu_deadline': time.strftime('%d %H:%M', time.localtime(gpu_deadline)),
                          'host_deadline': time.strftime('%d %H:%M', time.localtime(host_deadline)),
                          'watch': f'ssh -i {SSH_KEY} -p {hp} root@{hh}'}, indent=2), flush=True)
    finally:
        if not handed_off and state.get('host_id') and 'hh' in locals():
            # Keep the evidence before the rentals go away.
            try:
                diag = ssh(hh, hp, f'cd {REMOTE_ROOT}/qwen_verl_native 2>/dev/null && for f in cloud/vm_start.out cloud/agent_guard.out cloud/agent_guard.jsonl cloud/setup.log launch_failure.json; do echo "== $f"; tail -20 $f 2>/dev/null; done', timeout=60)
                (HERE / 'launch_diagnostics.txt').write_text(diag)
                print(diag[-3000:], flush=True)
            except Exception as exc:
                print('diagnostics unavailable:', exc, flush=True)
        if not handed_off:
            for key in ('gpu_id', 'host_id'):
                if state.get(key):
                    try:
                        print('cleanup', key, vast.call('DELETE', f'/instances/{state[key]}/', retries=200), flush=True)
                    except Exception as exc:
                        print('CLEANUP FAILED', key, exc, flush=True)
            state['active'] = False
            STATE.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
