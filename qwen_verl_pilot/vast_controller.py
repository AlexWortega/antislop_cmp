"""Create one bounded pilot rental and destroy it on STOP or deadline."""
import json,os,time,sys
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
BASE='https://console.vast.ai/api/v0'
HEAD={'Authorization':'Bearer '+os.environ['VAST_API_KEY']}
def request(method,path,**kwargs):
 r=requests.request(method,BASE+path,headers=HEAD,timeout=40,**kwargs)
 if not r.ok:raise RuntimeError(f'Vast {method} {path}: HTTP {r.status_code}: '+r.text[:250])
 return r.json()
if __name__=='__main__':
 assert not (ROOT/'instance.json').exists(),'Existing rental record; inspect instead of creating another'
 key=Path('/Users/alexwortega/.ssh/qwen_verl_pilot_20261101.pub').read_text().strip()
 k=request('POST','/ssh',json={'ssh_key':key});(ROOT/'ssh_registration.json').write_text(json.dumps(k,indent=2))
 offer=json.loads((ROOT/'offers_large.json').read_text())['offers'][0]
 assert offer['num_gpus']==1 and offer['gpu_ram']>=140000 and offer['dph_total']<=4.1
 assert offer['dph_total']*3+4+2<20
 body={'image':'nvidia/cuda:13.0.2-devel-ubuntu24.04','disk':180,'runtype':'ssh','target_state':'running','cancel_unavail':True,'label':'qwen-verl-pilot-budget20','onstart':'mkdir -p /workspace/pilot /workspace/pilot/output'}
 t=time.time();r=request('PUT',f"/asks/{offer['id']}/",json=body);assert r.get('success') and r.get('new_contract')
 ident=r['new_contract'];state={'id':ident,'created':t,'deadline':t+10800,'offer':offer['id'],'hourly_quote':offer['dph_total'],'max_hours':3,'image':body['image'],'status':'created'}
 (ROOT/'instance.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
 # Cleanup is independent of the training/SSH process and persists across assistant turns.
 while time.time()<state['deadline'] and not (ROOT/'STOP').exists():
  try:
   j=request('GET','/instances/');items=j.get('instances',[]);own=next((x for x in items if x['id']==ident),None)
   if own is None:state['status']='absent';break
   keep={key:own.get(key) for key in ['id','actual_status','status_msg','ssh_host','ssh_port','public_ipaddr','ports','dph_total','start_date','disk_space']}
   (ROOT/'instance_status.json').write_text(json.dumps(keep,indent=2))
  except Exception as e:print(type(e).__name__+': '+str(e),flush=True)
  time.sleep(30)
 if state['status']!='absent':
  while True:
   try:
    result=request('DELETE',f'/instances/{ident}/');state['destroy_response']=result;state['status']='destroyed';break
   except Exception as e:print('Cleanup retry: '+str(e),flush=True);time.sleep(15)
 state['finished']=time.time();(ROOT/'instance.json').write_text(json.dumps(state,indent=2));print('Rental cleanup',state['status'],flush=True)
