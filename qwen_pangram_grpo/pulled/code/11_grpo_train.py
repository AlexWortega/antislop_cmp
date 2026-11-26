"""GRPO against AI-slop. What the run IS lives in a YAML config, not in this file.

    python 11_grpo_train.py --config configs/semdiv1.yaml
    python 11_grpo_train.py --config configs/semdiv1.yaml --set train.steps=10 --set train.out=/tmp/x

The reward terms, their weights, the dataset and its column names, the guard corridor, the judge
backend and the model all come from the config; see `trainconfig.py` for the full key list and
`configs/` for the runs that produced the published models.
"""
import argparse
import json
import os
import statistics as st
from pathlib import Path

import torch

import trainconfig


def _compliance_stats() -> dict:
    """Compliance labels for the harmful slice of the step, if there was one."""
    try:
        import compliance_reward as C
        labs = [l for l in C.LAST_LABELS if l not in ("n/a", "unknown")]
    except Exception:
        return {}
    if not labs:
        return {}
    return {"harmful_n": len(labs),
            "comply_rate": round(labs.count("comply") / len(labs), 3),
            "refuse_rate": round(labs.count("refuse") / len(labs), 3)}


def _judge_axes() -> dict:
    """Per-axis judge means for the step just scored.

    The judge reward collapses three axes into one number, so without this the reasoning score is
    invisible and there is no way to tell a run that improved reasoning from one that only got
    politer answers.
    """
    try:
        import judge_reward as J
        rows = [r for r in J.LAST_SCORES if r]
    except Exception:
        return {}
    if not rows:
        return {}
    out = {}
    for axis in ("helpful", "faithful", "reasoning"):
        vals = [float(r[axis]) for r in rows if axis in r]
        if vals:
            out[f"judge_{axis}"] = round(st.mean(vals), 3)
    return out


class SampleLogger:
    """Dumps completions and per-component rewards so a human can actually read them."""

    def __init__(self, out_dir: Path, meld, every: int = 5, n: int = 8,
                 harmful_task: str = "harmful"):
        self.path = out_dir / "samples.jsonl"
        self.stats_path = out_dir / "step_stats.jsonl"
        self.meld = meld
        self.every = every
        self.n = n
        self.harmful_task = harmful_task
        self.step = 0

    def __call__(self, prompts, completions, **kwargs):
        """Registered as a zero-weight reward function so it sees every batch."""
        import guards as G
        from thinking import split_thinking
        self.step += 1
        raw = [G._text(c) for c in completions]
        if G.EXPECT_THINKING:
            parts = [split_thinking(t) for t in raw]
            reasonings = [r for r, _, _ in parts]
            texts = [a for _, a, _ in parts]
            closed = [c for _, _, c in parts]
        else:
            reasonings = ["" for _ in raw]
            texts = raw
            closed = [True] * len(raw)
        qs = [str(q) for q in kwargs.get("question", prompts)]
        details = self.meld.last_details or [{}] * len(texts)
        words = [len(t.split()) for t in texts]

        def num(d, *keys, default=float("nan")):
            """Detectors disagree on field names and return None for unscored items."""
            for k in keys:
                v = d.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
            return default

        def mean_ok(vals):
            ok = [v for v in vals if isinstance(v, float) and v == v]
            return round(st.mean(ok), 4) if ok else None

        p_ai = [num(d, "p_ai", "p") for d in details]
        logits = [num(d, "score", "logit", default=0.0) for d in details]
        gd = [G.guard_detail(q, t) for q, t in zip(qs, texts)]
        from diversity_reward import diversity_reward as _div
        div = _div(prompts, completions, **kwargs)
        try:
            from semantic_diversity import semantic_diversity_reward as _sdiv
            sdiv = _sdiv(prompts, completions, **kwargs)
        except Exception:
            sdiv = []
        row = {"step": self.step,
               "diversity_mean": round(st.mean(div), 4),
               **({"semdiv_mean": round(st.mean(sdiv), 4)} if sdiv else {}),
               "words_mean": round(st.mean(words), 1),
               "p_ai_mean": mean_ok(p_ai),
               "meld_logit_mean": round(st.mean(logits), 3),
               "meld_logit_sd": round(st.pstdev(logits), 3),
               "scored_frac": round(sum(1 for v in p_ai if v == v) / len(p_ai), 3),
               "fallback_frac": round(sum(d.get("via") == "meld" for d in details) / len(details), 3),
               **_compliance_stats(),
               "loop_rate": round(sum(g["loop"] < 0 for g in gd) / len(gd), 3),
               "markdown_rate": round(sum(g["markdown"] < 0 for g in gd) / len(gd), 3),
               "unfinished_rate": round(sum(g["unfinished"] < 0 for g in gd) / len(gd), 3),
               "short_rate": round(sum(w < G.MIN_WORDS for w in words) / len(words), 3),
               "reasoning_words_mean": round(st.mean([len(r.split()) for r in reasonings]), 1),
               "closed_frac": round(sum(closed) / len(closed), 3),
               **_judge_axes()}
        with self.stats_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if self.step % self.every == 1:
            with self.path.open("a") as f:
                tasks = [str(t) for t in kwargs.get("task", ["" for _ in texts])]
                for i in range(min(self.n, len(texts))):
                    row = {"step": self.step, "question": qs[i], "words": words[i],
                           "p_ai": p_ai[i], "guards": gd[i], "task": tasks[i]}
                    # The compliance reward is careful never to persist an answer to a harmful
                    # prompt, and this logger was quietly undoing that for every fifth prompt in the
                    # batch. Metrics are what the samples file is read for; the text of a compliant
                    # answer to a harmful request adds nothing to that and should not sit on disk.
                    if tasks[i] != self.harmful_task:
                        row["text"] = texts[i]
                        row["reasoning"] = reasonings[i]
                        row["closed"] = closed[i]
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return [0.0] * len(texts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML run config; see configs/")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                    help="override one config key, e.g. --set train.steps=10")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the config and build the dataset, then stop without touching a GPU")
    a = ap.parse_args()

    # Must happen before guards/judge/semantic_diversity are imported: they read their settings
    # from the environment at import time.
    cfg = trainconfig.load_and_apply(a.config, a.sets)
    print(trainconfig.describe(cfg), flush=True)

    out = Path(cfg["train"]["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.resolved.yaml").write_text(json.dumps(cfg, indent=1, default=str))

    ds = trainconfig.build_dataset(cfg)
    print(f"dataset {len(ds)} rows; columns {ds.column_names}", flush=True)
    if a.dry_run:
        print("sample prompt:\n" + repr(ds[0]["prompt"]), flush=True)
        return

    from peft import LoraConfig
    from transformers import AutoConfig, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    reward_funcs, reward_weights = trainconfig.build_rewards(cfg)
    detector = next((f for f in reward_funcs if getattr(f, "__name__", "") in ("meld", "pangram", "emu")),
                    None)
    harmful = next((r["task"] for r in cfg["rewards"] if r["name"] == "compliance"), "harmful")
    logger = SampleLogger(out, detector, every=cfg["train"]["log_every"],
                          n=cfg["train"]["log_samples"], harmful_task=harmful)
    logger.__name__ = "log"
    reward_funcs.append(logger)
    reward_weights.append(0.0)

    m = cfg["model"]
    loader_name = m["loader"]
    if loader_name == "auto":
        arch = (AutoConfig.from_pretrained(m["path"], trust_remote_code=True).architectures or [""])[0]
        loader_name = "image_text_to_text" if ("ConditionalGeneration" in arch or "ImageText" in arch) \
            else "causal_lm"
        print(f"model arch {arch or '?'} -> loader {loader_name}", flush=True)
    if loader_name == "image_text_to_text":
        from transformers import AutoModelForImageTextToText as Loader
    else:
        from transformers import AutoModelForCausalLM as Loader

    tok = AutoTokenizer.from_pretrained(m["path"], trust_remote_code=True)
    model = Loader.from_pretrained(m["path"], dtype=torch.bfloat16, device_map=m["device_map"],
                                   trust_remote_code=True)
    model.config.use_cache = False

    lora = m["lora"]
    if lora.get("init"):
        # Continue a previous run's adapter (new optimizer state): round 2 starts from pgemu1 ck120.
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora["init"], is_trainable=True)
        model.print_trainable_parameters()
    peft_config = None if lora.get("init") else LoraConfig(
        r=lora["rank"], lora_alpha=lora["alpha"] or 2 * lora["rank"],
        lora_dropout=lora["dropout"],   # must be 0: dropout makes the policy logprob stochastic and
                                        # corrupts both the importance ratio and the reference logprob
        bias="none", task_type="CAUSAL_LM", target_modules=lora["target_modules"],
    )

    t = cfg["train"]
    gen_batch = t["prompts_per_step"] * t["num_generations"]
    grpo = GRPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=t["micro_batch"],
        gradient_accumulation_steps=gen_batch // t["micro_batch"],
        num_generations=t["num_generations"],
        max_completion_length=t["max_completion_length"],
        temperature=t["temperature"], top_p=1.0, top_k=None,   # any truncation breaks the ratio
        learning_rate=t["lr"], lr_scheduler_type="constant_with_warmup",
        warmup_steps=t["warmup_steps"], max_grad_norm=t["max_grad_norm"],
        beta=t["beta"], epsilon=t["epsilon"], epsilon_high=t["epsilon_high"],
        loss_type="dr_grpo",        # no length bias, no divide-by-group-std
        scale_rewards="none",
        reward_weights=reward_weights,
        # A separate rollout server means a second full copy of the weights. On one 141 GB card
        # that is 54 GB for the server on top of ~70 GB for policy plus optimizer plus
        # activations, which does not leave room for the KV cache. Generating from the policy
        # itself is slower but keeps one copy and cannot OOM halfway through a paid run.
        # TRL 1.13 defaults to sequence_mask: one ratio per completion, exp(sum of per-token logprob
        # gaps between vLLM and the trainer), multiplied into the loss. On ~470-token thinking
        # rollouts the bf16 gap alone (0.03/token, at lr=0 on step 1) put that ratio at 0.036 on
        # average and 0.39 at most, i.e. the run trained at ~4% strength. Token-level ratios stay
        # near 1 and only the rare outlier token is clipped.
        **({"use_vllm": True, "vllm_mode": "server", "vllm_server_port": t["vllm_port"],
            "vllm_server_timeout": 600.0,
            "vllm_importance_sampling_mode": "token_truncate"} if t["vllm"] else {}),
        bf16=True, gradient_checkpointing=True,
        max_steps=t["steps"], save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        logging_steps=1, report_to=[], seed=t["seed"], log_completions=False,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=reward_funcs,
                          args=grpo, train_dataset=ds, peft_config=peft_config)
    print("trainable:", sum(p.numel() for p in trainer.model.parameters() if p.requires_grad) / 1e6,
          "M", flush=True)
    # RESUME=1 continues from the newest checkpoint in `out`: an unattended night run must survive a
    # crashed rollout server or a restarted pod without starting the schedule over.
    ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    resume = str(ckpts[-1]) if ckpts and os.environ.get("RESUME") == "1" else None
    if resume:
        print("resuming from", resume, flush=True)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out / "adapter"))
    print("saved", out / "adapter", flush=True)


if __name__ == "__main__":
    main()
