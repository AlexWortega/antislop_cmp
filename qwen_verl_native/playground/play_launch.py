"""Rent one H200, serve the base model and the step-50 LoRA with vLLM, and self-destruct after --hours.

Run from the laptop. vLLM listens on the pod's localhost only; reach it with the ssh tunnel this prints.
"""
import argparse
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'cloud'))
import cloud_launch as c  # noqa: E402

STATE = HERE / 'play_state.json'
REPO = 'AlexWortega/qwen-antislop-grpo-step50'
ADAPTER = HERE.parent / 'cloud/run_full50/adapter_step50'
IMAGE = 'nvidia/cuda:13.0.2-devel-ubuntu24.04'  # Vast's ssh works on this image; the vLLM image rejected our key


def signed_url():
    import requests
    from huggingface_hub import get_token
    r = requests.head(f'https://huggingface.co/{REPO}/resolve/main/adapter_model.safetensors',
                      headers={'Authorization': 'Bearer ' + get_token()}, allow_redirects=False, timeout=60)
    if r.status_code != 302 or 'Signature' not in r.headers.get('Location', ''):
        raise SystemExit(f'HF did not return a signed link: {r.status_code}')
    return r.headers['Location']


def pick(vast):
    offers = [x for x in vast.offers({'num_gpus': {'eq': 1}, 'gpu_ram': {'gte': 140000}})
              if x['gpu_name'] in ('H200', 'H200 NVL') and x.get('reliability2', 0) >= .97 and x['dph_total'] <= 5
              and x.get('inet_down', 0) >= 500 and x.get('disk_space', 0) >= 150 and x.get('cuda_max_good', 0) >= 12.9]
    if not offers:
        raise SystemExit('No H200 offer meets the filters')
    return sorted(offers, key=lambda x: x['dph_total'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hours', type=float, default=4)
    a = p.parse_args()
    if STATE.exists() and json.loads(STATE.read_text()).get('active'):
        raise SystemExit(f'{STATE} says a playground is already active')
    vast = c.Vast(c.load_secrets()['VAST_API_KEY'])
    key = secrets.token_urlsafe(24)
    bad = HERE / 'bad_machines.json'
    tried = set(json.loads(bad.read_text())) if bad.exists() else set()
    host = None
    for _ in range(4):  # some hosts never accept our key; move on to the next offer
        ident = None
        for offer in [o for o in pick(vast) if o['machine_id'] not in tried][:5]:
            try:
                ident = vast.call('PUT', f"/asks/{offer['id']}/", retries=1, json={
                    'image': IMAGE, 'disk': 150, 'runtype': 'ssh', 'target_state': 'running', 'cancel_unavail': True,
                    'label': f'antislop-play-{int(time.time())}', 'onstart': 'mkdir -p /workspace'})['new_contract']
                tried.add(offer['machine_id'])
                print('rented', ident, offer['gpu_name'], round(offer['dph_total'], 2), '$/h', offer.get('geolocation'), flush=True)
                break
            except RuntimeError as exc:
                print('offer', offer['id'], 'failed:', str(exc)[:150], flush=True)
        if ident is None:
            raise SystemExit('Could not rent any offer')
        deadline = time.time() + a.hours * 3600
        state = {'active': True, 'id': ident, 'deadline': deadline, 'serve_key': key}
        STATE.write_text(json.dumps(state, indent=2))
        STATE.chmod(0o600)
        try:
            host, port = c.wait_ssh(vast, ident, 'playground GPU', minutes=8)
            c.ssh(host, port, 'which rsync curl >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync python3 curl ca-certificates) >/dev/null 2>&1', timeout=600)
            break
        except Exception as exc:
            bad.write_text(json.dumps(sorted(tried)))
            print('pod unusable, destroying', ident, vast.destroy(ident), str(exc)[:120], flush=True)
            state['active'] = False
            STATE.write_text(json.dumps(state, indent=2))
            host = None
    if host is None:
        raise SystemExit('no rented pod became reachable')
    state['ssh'] = [host, port]
    STATE.write_text(json.dumps(state, indent=2))
    # Self-destruct first, so nothing after this can leave the pod billing.
    c.ssh(host, port, 'mkdir -p /workspace/adapter_step50 /workspace/base')
    c.rsync(host, port, HERE / 'pod_guard.py', '/workspace/pod_guard.py')
    c.rsync(host, port, HERE / 'serve.sh', '/workspace/serve.sh')
    c.ssh(host, port, f'cd /workspace && echo {deadline} > deadline && (setsid nohup python3 -u pod_guard.py {deadline} > guard.log 2>&1 < /dev/null &)')
    time.sleep(5)
    print('guard:', c.ssh(host, port, 'cat /workspace/guard_started; tail -2 /workspace/guard.log').strip(), flush=True)
    c.ssh(host, port, 'cat > /workspace/serve.key && chmod 600 /workspace/serve.key', stdin=key)
    # The pod gets only a short-lived signed CDN link to the one file; no HF token leaves the laptop.
    c.rsync(host, port, ADAPTER / 'adapter_config.json', '/workspace/adapter_step50/adapter_config.json')
    c.ssh(host, port, 'cat > /workspace/.adapter_url && cd /workspace && (setsid nohup sh -c "curl -sSfL --retry 8 --retry-all-errors -o adapter_step50/adapter_model.safetensors \\"\\$(cat .adapter_url)\\" && rm -f .adapter_url && touch adapter_step50/FETCHED || echo ADAPTER_FAILED >> setup.log" > adapter.log 2>&1 < /dev/null &)',
          stdin=signed_url())
    c.rsync(host, port, HERE / 'pod_setup.sh', '/workspace/pod_setup.sh')
    c.ssh(host, port, 'cd /workspace && (setsid nohup bash pod_setup.sh > setup.log 2>&1 < /dev/null &) ; (setsid nohup bash serve.sh > serve.out 2>&1 < /dev/null &)')
    print('pod installs vLLM, then pulls the base model and the adapter from HF', flush=True)
    local = subprocess.run(['shasum', '-a', '256', str(ADAPTER / 'adapter_model.safetensors')], capture_output=True, text=True).stdout.split()[0]
    for _ in range(240):
        done = c.ssh(host, port, 'ls /workspace/base/READY /workspace/adapter_step50/FETCHED 2>/dev/null | wc -l; tail -n 3 /workspace/setup.log | LC_ALL=C tr -cd "\\11\\12\\40-\\176"').split('\n', 1)
        if done[0].strip() == '2':
            break
        if 'SETUP_FAILED' in done[-1] or 'ADAPTER_FAILED' in done[-1]:
            raise SystemExit('pod setup failed: ' + done[-1])
        time.sleep(15)
    else:
        raise SystemExit('pod setup took over an hour; see /workspace/setup.log')
    remote = c.ssh(host, port, 'sha256sum /workspace/adapter_step50/adapter_model.safetensors', timeout=600).split()[0]
    if local != remote:
        raise SystemExit('adapter hash mismatch after download')
    c.ssh(host, port, 'touch /workspace/adapter_step50/READY')
    print('adapter verified; waiting for vLLM', flush=True)
    for _ in range(180):
        out = c.ssh(host, port, f'curl -s -H "Authorization: Bearer {key}" 127.0.0.1:8000/v1/models || true')
        if '"step50"' in out:
            break
        time.sleep(10)
    else:
        raise SystemExit('vLLM never came up; see /workspace/vllm.log on the pod')
    print(json.dumps({'ready': True, 'tunnel': f'ssh -N -L 8000:127.0.0.1:8000 -i {c.SSH_KEY} -p {port} root@{host}'}), flush=True)


if __name__ == '__main__':
    main()
