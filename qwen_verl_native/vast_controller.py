"""One rental, separate $150 cap, independent deadline and failure cleanup."""
import argparse
import json
import os
import time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
BASE='https://console.vast.ai/api/v0'
HEAD={'Authorization':'Bearer '+os.environ['VAST_API_KEY']}

def request(method,path,**kwargs):
    r=requests.request(method,BASE+path,headers=HEAD,timeout=40,**kwargs)
    if not r.ok:raise RuntimeError(f'Vast HTTP {r.status_code} for {method} {path}')
    return r.json()

def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');os.replace(temp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--offer',type=int);p.add_argument('--adopt-deadline',type=float);a=p.parse_args()
    if a.adopt_deadline:
        # Take over a live rental from a stopped controller and move its deadline, within the total cap.
        state_path=ROOT/'instance.json';state=json.loads(state_path.read_text());ident=state['id'];created=state['created']
        accounting=json.loads((ROOT/'restart_accounting.json').read_text())
        bound=accounting['prior_gpu_reserved_usd']+(a.adopt_deadline-created)/3600*state['hourly_bound']
        if state['status']!='created' or a.adopt_deadline<=state['deadline'] or bound>accounting['total_gpu_cap_usd']:raise SystemExit(f'Refusing to adopt: bound ${bound:.2f}')
        state['deadline']=a.adopt_deadline;save(state_path,state);print(json.dumps(state),flush=True)
    else:
        state_path=ROOT/'instance.json'
        if state_path.exists():raise SystemExit('Existing rental state: inspect, never create a duplicate')
        offer=next(x for x in json.loads((ROOT/'offers.json').read_text())['offers'] if x['id']==a.offer)
        assert offer['num_gpus']==1 and offer['gpu_ram']>=140000 and offer['cuda_max_good']>=13
        disk=200
        hourly=max(offer['dph_total'],offer['dph_base']+offer['storage_cost']*disk/720)*1.1
        assert hourly<4.5
        # $130 budgeted to time by default, leaving $20 for network and cleanup; a run may opt to spend it.
        prior=json.loads((ROOT/'restart_accounting.json').read_text())['prior_gpu_reserved_usd'] if (ROOT/'restart_accounting.json').exists() else 0.
        seconds=min(40*3600,max(0.,float(os.environ.get('ANTISLOP_TIME_CAP_USD',130))-prior)/hourly*3600)
        created=time.time();label='antislop-longrun-'+str(int(created))
        state={'status':'creation_pending','created':created,'label':label,'offer':a.offer,'gpu_budget':150.,'hourly_bound':hourly,'deadline':created+seconds,'disk_gb':disk}
        save(state_path,state)
        body={'image':'nvidia/cuda:13.0.2-devel-ubuntu24.04','disk':disk,'runtype':'ssh','target_state':'running','cancel_unavail':True,'label':label,'onstart':'mkdir -p /workspace/longrun /workspace/longrun/output'}
        try:
            result=request('PUT',f'/asks/{a.offer}/',json=body)
            assert result.get('success') and result.get('new_contract')
            ident=result['new_contract']
        except Exception:
            # Creation can succeed despite a lost response. Reconcile by unique label.
            matches=[x for x in request('GET','/instances/')['instances'] if x.get('label')==label]
            if len(matches)!=1:raise
            ident=matches[0]['id']
        state.update(id=ident,status='created');save(state_path,state);print(json.dumps(state),flush=True)
    reason='deadline'
    while time.time()<state['deadline']:
        if (ROOT/'STOP').exists():reason='requested_cleanup';break
        heartbeat=ROOT/'heartbeat.json'
        if heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>1200:
            reason='orchestrator_heartbeat_stale';break
        if not heartbeat.exists() and time.time()-created>3600:
            reason='setup_deadline';break
        if time.time()>state['deadline']-1800:(ROOT/'REQUEST_STOP').touch()
        try:
            instances=request('GET','/instances/')['instances']
            own=next((x for x in instances if x['id']==ident),None)
            if own is None:state['status']='absent';reason='already_absent';break
            save(ROOT/'instance_status.json',{k:own.get(k) for k in ['id','actual_status','status_msg','ssh_host','ssh_port','dph_total','start_date']})
        except Exception as exc:print(type(exc).__name__+': '+str(exc),flush=True)
        time.sleep(30)
    if state['status']!='absent':
        while True:
            try:
                result=request('DELETE',f'/instances/{ident}/')
                if not result.get('success'):raise RuntimeError('Destroy not confirmed')
                state.update(status='destroyed',destroy_response=result);break
            except Exception as exc:
                print('Cleanup retry: '+type(exc).__name__+': '+str(exc),flush=True);time.sleep(15)
    state.update(finished=time.time(),cleanup_reason=reason);save(state_path,state)
    for attempt in range(10):
        try:
            absent=all(x['id']!=ident for x in request('GET','/instances/')['instances'])
            if absent:
                save(ROOT/'rental_cleanup_verification.json',{'id':ident,'absent':True,'checked_at':time.time()});break
        except Exception:pass
        time.sleep(10)
    print('Rental cleanup: '+state['status'],flush=True)

if __name__=='__main__':main()
