"""Verify a real verl FSDP checkpoint changed the imported PEFT adapter."""
import json,math
from pathlib import Path
import torch
from safetensors.torch import load_file
ROOT=Path('/workspace/native')
p=ROOT/'output/checkpoints/global_step_1/actor'
meta=json.loads((p/'lora_train_meta.json').read_text())
assert int(meta.get('lora_rank',meta.get('r',0)))==32,meta
assert meta['lora_alpha']==512,meta
before=load_file(str(ROOT/'adapter-step0010/adapter_model.safetensors'))
state=torch.load(p/'model_world_size_1_rank_0.pt',map_location='cpu',weights_only=False)
def canonical(key):return key.replace('.default.','.').replace('_fsdp_wrapped_module.','')
lookup={canonical(k):v for k,v in before.items()};matched=0;changed=0;maxdelta=0.
for key,value in state.items():
 key=canonical(key)
 if key not in lookup:raise ValueError('Unmatched adapter tensor '+key)
 if hasattr(value,'to_local'):value=value.to_local()
 if hasattr(value,'local_shards'):
  shards=value.local_shards();assert len(shards)==1;value=shards[0].tensor
 original=lookup[key];assert value.shape==original.shape
 delta=float((value.float()-original.float()).abs().max());maxdelta=max(maxdelta,delta);matched+=1;changed+=int(delta>0)
assert matched==len(before) and changed>0,(matched,len(before),changed)
metrics_path=Path('/workspace/longrun/vendor/verl/antislop_verl_native/rank32-alpha512-batch16.jsonl')
if not metrics_path.exists():metrics_path=ROOT/'output/metrics.jsonl'
metrics=json.loads(metrics_path.read_text().splitlines()[-1])
norms={k:v for k,v in metrics['data'].items() if 'grad_norm' in k and 'actor' in k}
assert norms and all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in norms.values()),norms
(ROOT/'output/proof_metrics.json').write_text(json.dumps(metrics,indent=2))
report={'actor_gradient_norms':norms,'trainer':'verl.trainer.ppo.ray_trainer.RayPPOTrainer','native_global_step':1,'rank':32,'alpha':512,'matched_adapter_tensors':matched,'changed_adapter_tensors':changed,'maximum_weight_delta':maxdelta,'status':'verified_native_update'}
(ROOT/'output/native_step_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
