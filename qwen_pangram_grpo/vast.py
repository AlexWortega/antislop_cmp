"""Minimal Vast API helper for the Pangram-emulator GRPO run. Only touches instances labelled LABEL."""
import json, os, shlex, sys, urllib.request, urllib.error
from pathlib import Path
BASE = 'https://console.vast.ai/api/v0'
LABEL = 'antislop-pgemu-grpo'

def key():
    for line in (Path(__file__).resolve().parent.parent / 'qwen_verl_native/.secrets.env').read_text().splitlines():
        line = line.strip().removeprefix('export ')
        if line.startswith('VAST_API_KEY='):
            return shlex.split(line.split('=', 1)[1])[0]
    raise RuntimeError('no VAST_API_KEY')

def api(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + key(), 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'Vast HTTP {e.code}: ' + e.read().decode(errors='replace').replace(key(), '<k>')[:400]) from None

def offers(gpu, n, disk=400):
    q = {'gpu_name': {'eq': gpu}, 'num_gpus': {'eq': n}, 'rentable': {'eq': True}, 'verified': {'eq': True},
         'disk_space': {'gte': disk}, 'inet_down': {'gte': 500}, 'reliability2': {'gte': 0.97},
         'cuda_max_good': {'gte': 12.8}, 'order': [['dph_total', 'asc']], 'type': 'on-demand', 'limit': 8}
    return api('POST', '/bundles/', q).get('offers', [])

def mine():
    return [x for x in api('GET', '/instances/')['instances'] if x.get('label') == LABEL]

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'offers':
        for gpu in sys.argv[2:] or ['H200', 'RTX PRO 6000 WS', 'RTX PRO 6000 S', 'H100 SXM']:
            for x in offers(gpu, 2):
                print(f"{x['id']:>9} {gpu:16} ${x['dph_total']:.2f}/h disk {x['disk_space']:.0f}G ram {x['cpu_ram']/1000:.0f}G "
                      f"down {x['inet_down']:.0f} rel {x['reliability2']:.3f} {x.get('geolocation','')} cuda {x.get('cuda_max_good')}")
    elif cmd == 'all':
        for x in api('GET', '/instances/')['instances']:
            print(x['id'], x.get('label'), x.get('gpu_name'), x.get('num_gpus'), x.get('actual_status'), f"${x.get('dph_total',0):.2f}/h")
    elif cmd == 'mine':
        for x in mine():
            print(json.dumps({k: x.get(k) for k in ('id', 'actual_status', 'ssh_host', 'ssh_port', 'public_ipaddr', 'dph_total', 'gpu_name', 'num_gpus', 'status_msg')}))

def destroy(iid):
    x = [i for i in mine() if i['id'] == iid]
    if x:
        print(api('DELETE', f'/instances/{iid}/'))
    else:
        print('already gone')

if __name__ == '__main__' and sys.argv[1] == 'destroy':
    destroy(int(sys.argv[2]))
