"""PEFT 0.19.1 compatibility with Transformers 5.16 on non-TP models only."""
import peft.utils.save_and_load as serialization
if not getattr(serialization._maybe_shard_state_dict_for_tp,'_antislop_no_tp_guard',False):
 original=serialization._maybe_shard_state_dict_for_tp
 def guarded(model,state_dict,adapter_name):
  has_tp=any(getattr(m,'_hf_tp_plan',None) is not None and getattr(m,'_hf_device_mesh',None) is not None for m in model.modules())
  if has_tp:return original(model,state_dict,adapter_name)
  # The original implementation also does nothing when no layer has both
  # attributes. Avoid importing removed Transformers TP symbols in that case.
  return None
 guarded._antislop_no_tp_guard=True
 serialization._maybe_shard_state_dict_for_tp=guarded
