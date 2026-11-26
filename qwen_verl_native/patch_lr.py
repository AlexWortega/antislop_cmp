"""Rewrite the LR stored in a verl checkpoint copy; load_state_dict would otherwise restore the old value."""
import json,os,sys
import torch
def save(obj,path):
 temp=path+'.tmp';torch.save(obj,temp);os.replace(temp,path)
def patch(root,old,new):
 optim=root+'/actor/optim_world_size_1_rank_0.pt';extra=root+'/actor/extra_state_world_size_1_rank_0.pt'
 state=torch.load(optim,map_location='cpu',weights_only=False)
 for group in state['param_groups']:
  if group['lr'] not in (old,new):raise ValueError('Unexpected optimizer LR: '+str(group['lr']))
  group['lr']=new;group['initial_lr']=new
 save(state,optim)
 data=torch.load(extra,map_location='cpu',weights_only=False);sched=data['lr_scheduler']
 if sched['base_lrs'][0] not in (old,new):raise ValueError('Unexpected scheduler LR: '+str(sched['base_lrs']))
 sched['base_lrs']=[new for _ in sched['base_lrs']];sched['_last_lr']=[new for _ in sched['_last_lr']]
 save(data,extra)
 check=torch.load(optim,map_location='cpu',weights_only=False)['param_groups'];after=torch.load(extra,map_location='cpu',weights_only=False)['lr_scheduler']
 if any(g['lr']!=new for g in check) or after['base_lrs']!=[new]*len(after['base_lrs']):raise ValueError('LR patch did not persist')
 return {'optimizer_lr':check[0]['lr'],'scheduler_base_lrs':after['base_lrs'],'scheduler_last_epoch':after['last_epoch']}
if __name__=='__main__':print(json.dumps(patch(sys.argv[1],float(sys.argv[2]),float(sys.argv[3]))))
