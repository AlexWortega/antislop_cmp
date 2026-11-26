"""Merge a trained GRPO LoRA adapter into the policy weights, for serving and evaluation.

python 13_merge.py --model $POLICY --adapter runs/pilot/adapter --out /mnt/antislop-grpo/models/grpo
"""
import argparse
import os
import shutil
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("POLICY", ""))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoConfig, AutoTokenizer

    # The 27B line is text-only and the 4B one is a multimodal shell, and each loader refuses the
    # other's checkpoint, so pick by what the config says rather than by which model we had in mind.
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    arch = (cfg.architectures or [""])[0]
    if "ConditionalGeneration" in arch or "ImageText" in arch:
        from transformers import AutoModelForImageTextToText as Loader
    else:
        from transformers import AutoModelForCausalLM as Loader
    print("base arch:", arch or "?", flush=True)

    model = Loader.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu",
                                   trust_remote_code=True)
    model = PeftModel.from_pretrained(model, a.adapter, device_map="cpu")
    model = model.merge_and_unload()
    out = Path(a.out)
    # 5GB shards: transformers 5 defaults to shards large enough to exceed the per-file limit on the
    # Hub, which only bites at upload time, long after the merge looked like it had worked.
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")

    # Runs save a tokenizer next to the adapter; prefer it, since it is the one training used.
    tok_src = a.adapter if (Path(a.adapter) / "tokenizer.json").exists() else a.model
    AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True).save_pretrained(out)
    for name in os.listdir(a.model):
        if name.endswith((".json", ".jinja", ".txt")) and not (out / name).exists():
            shutil.copy(Path(a.model) / name, out / name)
    print("merged ->", out)


if __name__ == "__main__":
    main()
