"""Export our single-rank LoRA checkpoint without loading the 27B base."""
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from peft import LoraConfig

source = Path(sys.argv[1])
destination = source.parent / 'adapter'
destination.mkdir(exist_ok=True)
state = torch.load(source/'model_world_size_1_rank_0.pt', map_location='cpu', weights_only=False)
weights = {}
for name, value in state.items():
    assert 'lora_' in name, 'Expected LoRA-only checkpoint'
    if hasattr(value, 'to_local'):
        value = value.to_local()
    assert torch.isfinite(value).all(), name
    weights[name.replace('.default.weight', '.weight')] = value.detach().cpu().contiguous().clone()
meta = json.loads((source/'lora_train_meta.json').read_text())
config = LoraConfig(r=meta['r'], lora_alpha=meta['lora_alpha'],
                    task_type=meta['task_type'], inference_mode=True,
                    target_modules=sorted({k.split('.')[-3] for k in weights}),
                    base_model_name_or_path='Qwen/Qwen3.8-27B',
                    revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
config.save_pretrained(destination)
save_file(weights, destination/'adapter_model.safetensors')
loaded = load_file(destination/'adapter_model.safetensors')
assert all(torch.equal(value, loaded[key]) for key, value in weights.items())
b = [value.float() for key, value in weights.items() if '.lora_B.' in key]
assert b and any(torch.count_nonzero(value).item() for value in b), 'No learned LoRA B update'
stats = {'tensors': len(weights), 'parameters': sum(v.numel() for v in weights.values()),
         'lora_B_nonzero': sum(torch.count_nonzero(v).item() for v in b),
         'lora_B_l2': sum(v.square().sum().item() for v in b)**0.5,
         'safetensors_roundtrip_exact': True}
(destination/'verification.json').write_text(json.dumps(stats, indent=2)+'\n')
print(json.dumps(stats))
