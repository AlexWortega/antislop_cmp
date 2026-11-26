"""Expand LoRA rank16->32 at alpha512 while preserving the policy delta."""
import json,math
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
root=Path(__file__).resolve().parent
src=root/'checkpoints/step0001';dst=root/'checkpoints/rank32-step0001';dst.mkdir(exist_ok=True)
assert not (dst/'adapter_model.safetensors').exists(), 'Do not overwrite expanded weights'
config_path=src/'adapter_config.server.json'
if not config_path.exists():config_path=src/'adapter_config.json'
cfg=json.loads(config_path.read_text());assert cfg['r']==16 and cfg['lora_alpha']==512
assert not cfg.get('rank_pattern') and not cfg.get('alpha_pattern') and not cfg.get('use_rslora') and not cfg.get('use_dora')
torch.manual_seed(20261102);old=load_file(str(src/'adapter_model.safetensors'));new={};max_error=0
for key,value in old.items():
 if 'lora_A' in key:
  assert value.shape[0]==16
  extra=torch.empty_like(value);torch.nn.init.kaiming_uniform_(extra,a=math.sqrt(5));new[key]=torch.cat([value,extra],dim=0)
 elif 'lora_B' in key:
  assert value.shape[1]==16
  new[key]=torch.cat([value*2,torch.zeros_like(value)],dim=1)
 else:raise ValueError('Unexpected adapter tensor '+key)
for key,a in old.items():
 if 'lora_A' not in key:continue
 bkey=key.replace('lora_A','lora_B');b=old[bkey];x=torch.randn(a.shape[1],3)
 before=b.float()@(a.float()@x)*32
 after=new[bkey].float()@(new[key].float()@x)*16
 err=float((before-after).abs().max());max_error=max(err,max_error)
 torch.testing.assert_close(before,after,atol=1e-5,rtol=1e-5)
cfg['r']=32
save_file(new,str(dst/'adapter_model.safetensors'));(dst/'adapter_config.json').write_text(json.dumps(cfg,indent=2))
(dst/'expansion_verification.json').write_text(json.dumps({'source_rank':16,'rank':32,'alpha':512,'maximum_probe_error':max_error,'optimizer_reset':True,'weights_preserve_previous_delta':True},indent=2))
print('Verified expansion, maximum error',max_error)
