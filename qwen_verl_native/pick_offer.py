"""Refresh offers.json from Vast and print the cheapest reliable offer the controller accepts."""
import json,os,sys
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent;DISK=200
q={'num_gpus':{'eq':1},'gpu_ram':{'gte':140000},'cuda_max_good':{'gte':13},'rentable':{'eq':True},'rented':{'eq':False},'disk_space':{'gte':DISK},'reliability2':{'gte':.98},'type':'on-demand','limit':200}
r=requests.get('https://console.vast.ai/api/v0/bundles/',params={'q':json.dumps(q)},headers={'Authorization':'Bearer '+os.environ['VAST_API_KEY']},timeout=40);r.raise_for_status()
offers=[x for x in r.json()['offers'] if x['num_gpus']==1 and x['gpu_ram']>=140000 and x['cuda_max_good']>=13]
def hourly(x):return max(x['dph_total'],x['dph_base']+x['storage_cost']*DISK/720)*1.1
offers=sorted((x for x in offers if hourly(x)<4.5 and x.get('inet_up',0)>=500 and x.get('inet_down',0)>=500),key=hourly)
if not offers:sys.exit('No acceptable Vast offer under the hourly cap')
(ROOT/'offers.json').write_text(json.dumps({'offers':offers},indent=2))
best=offers[0];print(json.dumps({k:best.get(k) for k in ('id','gpu_name','gpu_ram','dph_total','reliability2','geolocation')}|{'hourly_bound':hourly(best)}),file=sys.stderr);print(best['id'])
