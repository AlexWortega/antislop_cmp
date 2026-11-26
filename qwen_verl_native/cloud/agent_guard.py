"""Runs on the agent host: heartbeats the GPU, ends the run before its deadline, then stops this host.

Teardown order keeps the checkpoint: native_runner copies the final checkpoint here and touches STOP;
only then is the GPU told to destroy itself. This host is stopped, not destroyed, so its disk
(checkpoints, rollouts, logs) survives until someone collects it.
"""
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPU = ROOT / 'instance.json'
VM = ROOT / 'cloud' / 'vm.json'
STATE = ROOT / 'cloud' / 'agent_guard_state.json'
LOG = ROOT / 'cloud' / 'agent_guard.jsonl'
GONE_AFTER = 600


def decide(now, gpu, vm, state, run_finished):
    """Return the actions due now. Pure, so the teardown order can be tested."""
    actions = []
    gpu_alive = gpu.get('status') == 'created' and not state.get('gpu_gone_at')
    if gpu_alive:
        actions.append('heartbeat')
        if now > gpu['deadline'] - 1800:
            actions.append('request_stop')
        if (run_finished or now > gpu['deadline']) and not state.get('stop_sent_at'):
            actions.append('stop_gpu')
        sent = state.get('stop_sent_at')
        if sent and now - max(sent, state.get('last_ssh_ok', 0)) > GONE_AFTER:
            actions.append('mark_gpu_gone')
        elif now > gpu['deadline'] + GONE_AFTER:
            # The GPU's own guard destroys it at the deadline.
            actions.append('mark_gpu_gone')
    gone = state.get('gpu_gone_at')
    if (gone and now > gone + vm.get('keep_after_run', 3600)) or now > vm['deadline']:
        actions.append('stop_self')
    return actions


def credentials():
    env = dict(os.environ)
    for path in ('/etc/environment',):
        try:
            for line in Path(path).read_text().splitlines():
                key, _, value = line.partition('=')
                env.setdefault(key.strip(), value.strip().strip('"'))
        except OSError:
            pass
    return env.get('CONTAINER_API_KEY')


def stop_self(ident, key):
    request = urllib.request.Request(f'https://console.vast.ai/api/v0/instances/{ident}/', method='PUT',
                                     data=json.dumps({'state': 'stopped'}).encode(),
                                     headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.loads(response.read() or b'{}')


def remote(command, timeout=60):
    # Standard library only: this guard starts before the Python environment is installed.
    ssh = json.loads((ROOT / 'instance_status.json').read_text())
    key = os.environ.get('ANTISLOP_SSH_KEY', '/root/.ssh/antislop_gpu')
    result = subprocess.run(['ssh', '-i', key, '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=20',
                             '-p', str(ssh['ssh_port']), 'root@' + ssh['ssh_host'], command],
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'ssh {result.returncode}: ' + result.stderr[-300:])
    return result.stdout


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def log(**event):
    with LOG.open('a') as f:
        f.write(json.dumps({'time': time.time(), **event}) + '\n')


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    log(event='started')
    while True:
        gpu = json.loads(GPU.read_text())
        vm = json.loads(VM.read_text())
        now = time.time()
        for action in decide(now, gpu, vm, state, (ROOT / 'STOP').exists()):
            try:
                if action == 'heartbeat':
                    remote('touch /workspace/native/agent_heartbeat', timeout=60)
                    state['last_ssh_ok'] = now
                elif action == 'request_stop' and not (ROOT / 'REQUEST_STOP').exists():
                    (ROOT / 'REQUEST_STOP').touch()
                    log(event='request_stop')
                elif action == 'stop_gpu':
                    remote('touch /workspace/native/STOP', timeout=60)
                    state['stop_sent_at'] = now
                    log(event='gpu_stop_sent')
                elif action == 'mark_gpu_gone':
                    state['gpu_gone_at'] = now
                    gpu['status'] = 'destroyed'
                    gpu['cleanup_reason'] = 'gpu_guard'
                    save(GPU, gpu)
                    log(event='gpu_gone')
                elif action == 'stop_self':
                    key = credentials()
                    if not key:
                        log(event='no_instance_key_cannot_stop_self')
                        time.sleep(600)
                        continue
                    log(event='stopping_self', response=stop_self(vm['id'], key))
                    return
            except Exception as exc:
                log(event='action_failed', action=action, error=type(exc).__name__ + ': ' + str(exc)[:300])
        save(STATE, state)
        time.sleep(60)


if __name__ == '__main__':
    main()
