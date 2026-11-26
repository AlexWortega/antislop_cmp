"""Runs on the GPU rental and destroys it, so billing stops even when every other machine is gone.

The rental is destroyed when the agent host asks (STOP, after it has copied the checkpoint),
when the hard deadline passes, or when the agent host stops sending heartbeats.
It uses only the per-instance key Vast injects, which can manage this one instance and nothing else.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

NATIVE = Path(os.environ.get('ANTISLOP_NATIVE_ROOT', '/workspace/native'))
CONFIG = NATIVE / 'guard_config.json'
HEARTBEAT = NATIVE / 'agent_heartbeat'
STOP = NATIVE / 'STOP'
LOG = NATIVE / 'output' / 'gpu_guard.jsonl'


def decide(now, config, started, heartbeat_mtime, stop_requested):
    """Return the reason to destroy the rental, or None to keep it."""
    if stop_requested:
        return 'agent_host_requested_stop'
    if now >= config['deadline']:
        return 'deadline'
    limit = config.get('heartbeat_timeout', 1200)
    if heartbeat_mtime is None:
        # The agent host starts heartbeats while it bootstraps this machine.
        if now - started > config.get('first_heartbeat_grace', 3600):
            return 'agent_host_never_connected'
    elif now - heartbeat_mtime > limit:
        return 'agent_host_heartbeat_stale'
    return None


def instance_credentials():
    """Vast puts the instance key in the container environment, which ssh sessions may not inherit."""
    env = dict(os.environ)
    try:
        for item in Path('/proc/1/environ').read_bytes().split(b'\0'):
            key, _, value = item.decode(errors='replace').partition('=')
            env.setdefault(key, value)
    except OSError:
        pass
    return env.get('CONTAINER_ID'), env.get('CONTAINER_API_KEY')


def destroy(ident, key):
    request = urllib.request.Request(f'https://console.vast.ai/api/v0/instances/{ident}/', method='DELETE',
                                     headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.loads(response.read() or b'{}')


def log(**event):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        f.write(json.dumps({'time': time.time(), **event}) + '\n')


def main():
    config = json.loads(CONFIG.read_text())
    ident, key = instance_credentials()
    if not ident or not key:
        log(event='no_instance_key')
        raise SystemExit('CONTAINER_ID or CONTAINER_API_KEY missing: cannot self-destruct')
    started = time.time()
    log(event='started', deadline=config['deadline'], instance=ident)
    reason = None
    while reason is None:
        mtime = HEARTBEAT.stat().st_mtime if HEARTBEAT.exists() else None
        reason = decide(time.time(), config, started, mtime, STOP.exists())
        if reason is None:
            time.sleep(30)
    log(event='destroying', reason=reason)
    while True:
        try:
            log(event='destroy_response', response=destroy(ident, key))
            return
        except Exception as exc:
            log(event='destroy_retry', error=type(exc).__name__ + ': ' + str(exc)[:200])
            time.sleep(15)


if __name__ == '__main__':
    main()
