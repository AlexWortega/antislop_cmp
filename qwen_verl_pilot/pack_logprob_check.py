"""Packed vs single forward of a tiny random Qwen3.5 with verl's patches (CPU, transformers 5.16.1)."""
import sys, torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
import types, importlib.util
for name in ["verl", "verl.utils"]:
    sys.modules[name] = types.ModuleType(name)
u = types.ModuleType("verl.utils.ulysses")
u.get_ulysses_sequence_parallel_world_size = lambda *a, **k: 1
u.get_ulysses_sequence_parallel_group = lambda *a, **k: None
u.__getattr__ = lambda n: (lambda *a, **k: None)
sys.modules["verl.utils.ulysses"] = u
spec = importlib.util.spec_from_file_location("vq35", "/Users/alexwortega/Desktop/side_projects/antislop_cmp/qwen_verl_pilot/vendor/verl/verl/models/transformers/qwen3_5.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)

M.Qwen3_5Model.forward = V.qwen3_5_base_forward
M.Qwen3_5DecoderLayer.forward = V.qwen3_5_decoder_layer_forward
M.Qwen3_5GatedDeltaNet.forward = V.qwen3_5_gated_delta_net_forward
M.Qwen3_5ForConditionalGeneration.forward = V.forward_with_normal_backend

attn = sys.argv[1] if len(sys.argv) > 1 else "sdpa"
LT = {"lin": ["linear_attention"]*4, "full": ["full_attention"]*4, "mix": ["linear_attention"]*3+["full_attention"]}[sys.argv[2] if len(sys.argv)>2 else "mix"]
torch.manual_seed(0)
cfg = Qwen3_5Config(
    text_config=dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                     layer_types=LT,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                     linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                     linear_conv_kernel_dim=4, max_position_embeddings=256),
    vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64),
)
cfg._attn_implementation = attn
cfg.text_config._attn_implementation = attn
model = M.Qwen3_5ForConditionalGeneration(cfg).eval().float()
print("forward has cu_seqlens:", "cu_seqlens" in __import__("inspect").signature(model.forward).parameters)

import os; lens = [int(x) for x in os.environ.get("LENS","7,11,5").split(",")]
seqs = [torch.randint(0, 128, (n,)) for n in lens]

def pos(n_list):
    t = torch.cat([torch.arange(n) for n in n_list])
    return t.view(1, 1, -1).expand(4, 1, -1).contiguous()

@torch.no_grad()
def run(ids, n_list, cu):
    kw = dict(input_ids=ids.view(1, -1), attention_mask=None, use_cache=False, position_ids=pos(n_list))
    if cu:
        c = torch.tensor([0] + list(torch.tensor(n_list).cumsum(0)), dtype=torch.long)
        kw.update(cu_seqlens=c, cu_seqlens_cpu=c)
    return model(**kw).logits[0].log_softmax(-1)

single = torch.cat([run(s, [len(s)], True) for s in seqs])
packed = run(torch.cat(seqs), lens, True)
nocu = run(torch.cat(seqs), lens, False)
print(f"[{attn}] packed+cu_seqlens vs single: max|dlogp| = {(packed - single).abs().max():.2e}")
print(f"[{attn}] packed without cu_seqlens vs single: max|dlogp| = {(nocu - single).abs().max():.2e}")
off = 0
for n in lens:
    print(f"   seq len {n}: with cu {(packed[off:off+n]-single[off:off+n]).abs().max():.2e}  without {(nocu[off:off+n]-single[off:off+n]).abs().max():.2e}")
    off += n
