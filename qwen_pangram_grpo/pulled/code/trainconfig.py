"""One YAML file decides what a GRPO run is: model, data, columns, reward terms, hyperparameters.

Runs used to be defined by a shell script that exported eleven environment variables and passed
fourteen flags, which meant the record of what a run actually was lived in three places and agreed
in none of them. Here the file is the run.

Two things this deliberately does NOT do quietly:

* an unknown key is an error, not a shrug. A typo in `semdiv_weight` vs `semdiv-weight` used to mean
  the term silently stayed at zero and the run looked fine for six hours.
* a reward term with weight 0 is dropped and announced. Keeping it would cost a forward pass, and a
  judge term costs money, to multiply the result by nothing.

The reward modules read their settings from the environment at import time, so `apply_env` has to
run before they are imported. `load_and_apply` does both in the right order.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

#: Every key a config may set, with its default. The tree doubles as the schema: anything not in
#: here is rejected, so a misspelled key fails at startup rather than at hour six.
DEFAULTS: dict[str, Any] = {
    "model": {
        "path": "",
        # auto reads config.architectures; the 27B line is text-only and the 4B one is a
        # multimodal shell, and each loader refuses the other's checkpoint.
        "loader": "auto",                 # auto | causal_lm | image_text_to_text
        "device_map": {"": 0},
        "lora": {
            "rank": 32,
            "alpha": 0,                   # 0 means 2 * rank
            "dropout": 0.0,               # must stay 0, see the comment where it is used
            "target_modules": r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            "init": "",                   # adapter dir to continue from (trainable); empty = fresh LoRA
        },
    },
    "data": {
        "path": "",                       # local .jsonl
        "hf": {"id": "", "split": "train", "config": ""},
        "limit": 0,
        "shuffle_seed": None,
        "columns": {
            # left: the name the trainer and the reward functions use.
            # right: the column in YOUR file that carries it.
            "prompt": "prompt",
            "question": "question",
            "task": "task",
            "keep": ["lang", "source", "source_text"],
        },
        # Used when the data has questions but no rendered prompt. Training format is not cosmetic:
        # the rewards were computed on whatever this produces.
        "prompt_template": "",
    },
    "thinking": False,
    "guards": {
        "min_words": 80,
        "target_words": 110,
        "max_words": 400,
        "length_scale": 0.5,
        "max_sentence_words": 30,
        "long_sentence_scale": 0.0,
    },
    "judge": {
        "backend": "openrouter",          # openrouter | azure
        "model": "google/gemini-3.8-flash",
        "think_model": "",                # judge for antislop_think; empty = same as model
    },
    "rewards": [],                        # list of {name, weight, ...}; see REWARD_KEYS
    "train": {
        "out": "",
        "steps": 150,
        "prompts_per_step": 6,
        "num_generations": 8,
        "micro_batch": 1,
        "max_completion_length": 640,
        "lr": 5e-5,
        "beta": 0.02,
        "warmup_steps": 10,
        "max_grad_norm": 0.2,
        "epsilon": 0.2,
        "epsilon_high": 0.28,
        "temperature": 1.0,
        "save_steps": 25,
        "save_total_limit": 10,           # not 2: it silently ate the checkpoints of a run whose
                                          # whole point was comparing early against late
        "seed": 0,
        "log_every": 5,                   # sample dumps, in steps
        "log_samples": 8,
        "vllm": False,
        "vllm_port": 8000,
    },
}

#: Per-reward keys, beyond `name` and `weight`. Same strictness as the tree above.
REWARD_KEYS: dict[str, dict[str, Any]] = {
    "meld":       {"dir": "", "device": "cuda:0", "temp": 6.0},
    "pangram":    {"model": "default", "workers": 6, "temp": 6.0,
                   "fallback_dir": "", "fallback_device": "cuda:0"},
    "judge":      {},                     # two-axis judge: helpful + faithful
    "rubric":     {},                     # three-axis readability rubric
    "antislop":   {},                     # antislop-copywriting rulebook judge (astra1)
    "antislop_think": {},                 # the same rulebook on the private reasoning (astra2)
    "dlg_antislop": {},                   # calibrated dialogue instrument on a chat's final reply (chat1)
    "continuation": {},                   # continuity + voice against data.columns.keep source_text
    "source_sim": {"embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                   "embed_device": "cuda:0", "floor": 0.35},
    "guards":     {},
    "diversity":  {},                     # lexical; kept only to reproduce old runs
    "semdiv":     {"embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                   "embed_device": "cuda:0"},
    "compliance": {"task": "harmful"},
    "emu":        {},                     # Opus-5.5 Pangram emulator scalar score (emu_reward.py)
    "task":       {},                     # DeepSeek task-fulfilment check
    "length":     {},                     # per-task target_words corridor
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        where = f"{path}.{k}" if path else k
        if k not in out:
            raise ConfigError(f"unknown config key: {where}\n  known here: {sorted(out)}")
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v, where)
        else:
            out[k] = v
    return out


def _check_rewards(rewards: list) -> list[dict]:
    if not isinstance(rewards, list) or not rewards:
        raise ConfigError("rewards: expected a non-empty list of {name, weight, ...}")
    seen, out = set(), []
    for i, r in enumerate(rewards):
        if not isinstance(r, dict) or "name" not in r:
            raise ConfigError(f"rewards[{i}]: expected a mapping with a `name`")
        name = r["name"]
        if name not in REWARD_KEYS:
            raise ConfigError(f"rewards[{i}]: unknown reward {name!r}; "
                              f"known: {sorted(REWARD_KEYS)}")
        if name in seen:
            raise ConfigError(f"rewards[{i}]: {name!r} listed twice")
        seen.add(name)
        allowed = {"name", "weight", *REWARD_KEYS[name]}
        extra = set(r) - allowed
        if extra:
            raise ConfigError(f"rewards[{i}] ({name}): unknown keys {sorted(extra)}; "
                              f"allowed: {sorted(allowed)}")
        merged = {"weight": 1.0, **REWARD_KEYS[name], **r}
        out.append(merged)
    if {"judge", "rubric"} <= seen:
        raise ConfigError("rewards: `judge` and `rubric` both present — they are two versions of "
                          "the same judge call and would double the API bill for one signal")
    if {"antislop", "rubric"} <= seen or {"antislop", "judge"} <= seen:
        raise ConfigError("rewards: `antislop` replaces the judge rubric in astra1; using both "
                          "pays twice for overlapping judge signals")
    if {"diversity", "semdiv"} <= seen:
        raise ConfigError("rewards: `diversity` and `semdiv` both present — semdiv replaces the "
                          "lexical term rather than joining it")
    return out


def _coerce(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def load(path: str | Path, sets: list[str] | None = None) -> dict:
    """Read a YAML config, apply `a.b=c` overrides, validate, return the resolved tree."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    cfg = _merge(DEFAULTS, {k: v for k, v in raw.items() if k != "rewards"})
    cfg["rewards"] = _check_rewards(raw.get("rewards", []))

    for s in sets or []:
        if "=" not in s:
            raise ConfigError(f"--set {s!r}: expected key=value")
        key, value = s.split("=", 1)
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise ConfigError(f"--set {key}: no such section {p!r}")
            node = node[p]
        if parts[-1] not in node:
            raise ConfigError(f"--set {key}: unknown key {parts[-1]!r}")
        node[parts[-1]] = _coerce(value)

    if not cfg["model"]["path"]:
        raise ConfigError("model.path is required")
    if not cfg["train"]["out"]:
        raise ConfigError("train.out is required")
    if not cfg["data"]["path"] and not cfg["data"]["hf"]["id"]:
        raise ConfigError("set either data.path (a .jsonl) or data.hf.id (a Hub dataset)")
    t = cfg["train"]
    gen_batch = t["prompts_per_step"] * t["num_generations"]
    if any(r["name"] == "antislop_think" for r in cfg["rewards"]) and not cfg["thinking"]:
        raise ConfigError("rewards: `antislop_think` scores the reasoning, so it needs thinking: true; "
                          "without it every rollout has no thought and the term is a constant 0")
    if gen_batch % t["micro_batch"]:
        raise ConfigError(f"prompts_per_step * num_generations ({gen_batch}) must divide by "
                          f"micro_batch ({t['micro_batch']})")
    return cfg


def apply_env(cfg: dict) -> None:
    """Push the config into the environment the reward modules read at import time.

    Call this BEFORE importing guards, judge_reward, semantic_diversity or compliance_reward.
    """
    g = cfg["guards"]
    env = {
        "EXPECT_THINKING": "1" if cfg["thinking"] else "0",
        "GUARD_MIN_WORDS": g["min_words"],
        "GUARD_TARGET_WORDS": g["target_words"],
        "GUARD_MAX_WORDS": g["max_words"],
        "GUARD_LENGTH_SCALE": g["length_scale"],
        "GUARD_MAX_SENTENCE_WORDS": g["max_sentence_words"],
        "GUARD_LONG_SENTENCE_SCALE": g["long_sentence_scale"],
        "JUDGE_BACKEND": cfg["judge"]["backend"],
        "JUDGE_MODEL": cfg["judge"]["model"],
        "ANTISLOP_JUDGE_MODEL": cfg["judge"]["model"],
        "ANTISLOP_THINK_JUDGE_MODEL": cfg["judge"]["think_model"] or cfg["judge"]["model"],
    }
    for r in cfg["rewards"]:
        if r["name"] == "semdiv":
            env["DIVERSITY_EMBED_MODEL"] = r["embed_model"]
            env["DIVERSITY_EMBED_DEVICE"] = r["embed_device"]
        if r["name"] == "compliance":
            env["COMPLIANCE_TASK"] = r["task"]
        if r["name"] == "source_sim":
            env["SOURCE_SIM_MODEL"] = r["embed_model"]
            env["SOURCE_SIM_DEVICE"] = r["embed_device"]
            env["SOURCE_SIM_FLOOR"] = r["floor"]
    for k, v in env.items():
        os.environ[k] = str(v)


def load_and_apply(path: str | Path, sets: list[str] | None = None) -> dict:
    cfg = load(path, sets)
    apply_env(cfg)
    return cfg


def build_dataset(cfg: dict):
    """Rename the user's columns to the names the trainer and the reward functions expect."""
    from datasets import Dataset, load_dataset

    d = cfg["data"]
    cols = d["columns"]
    if d["path"]:
        rows = [json.loads(l) for l in Path(d["path"]).read_text().splitlines() if l.strip()]
    else:
        ds = load_dataset(d["hf"]["id"], d["hf"]["config"] or None, split=d["hf"]["split"])
        rows = list(ds)

    if d["shuffle_seed"] is not None:
        import random
        random.Random(d["shuffle_seed"]).shuffle(rows)
    if d["limit"]:
        rows = rows[: d["limit"]]
    if not rows:
        raise ConfigError("dataset is empty")

    have = set(rows[0])
    if not d["prompt_template"] and cols["prompt"] not in have:
        raise ConfigError(
            f"data.columns.prompt = {cols['prompt']!r} is not in the data (columns: {sorted(have)}). "
            f"Either point it at the right column or set data.prompt_template to render one.")
    if cols["question"] not in have and not d["prompt_template"]:
        raise ConfigError(f"data.columns.question = {cols['question']!r} is not in the data; the "
                          f"judges score the answer against the question and need it")

    out = []
    for r in rows:
        question = str(r.get(cols["question"], ""))
        prompt = (d["prompt_template"].format(**r) if d["prompt_template"]
                  else str(r.get(cols["prompt"], "")))
        row = {"prompt": prompt, "question": question, "task": str(r.get(cols["task"], ""))}
        # A ragged set of columns breaks Dataset.from_list, so every row carries every key.
        for k in cols["keep"]:
            row[k] = r.get(k, "")
        out.append(row)
    return Dataset.from_list(out)


def build_rewards(cfg: dict) -> tuple[list, list[float]]:
    """Instantiate the reward terms in config order. Returns (functions, weights).

    A term at weight 0 is dropped: GRPO only ever sees within-group spread times weight, so a zero
    weight buys nothing and a judge term at zero weight buys nothing for money.
    """
    import guards as G
    funcs: list = []
    weights: list[float] = []

    for r in cfg["rewards"]:
        name, w = r["name"], float(r["weight"])
        if w == 0.0:
            print(f"[config] dropping reward {name!r}: weight 0", flush=True)
            continue

        if name == "meld":
            from meld_reward import MeldReward
            fn = MeldReward(r["dir"], device=r["device"], min_words=G.MIN_WORDS, temp=r["temp"])
            fn.__name__ = "meld"
        elif name == "pangram":
            from meld_reward import MeldReward
            from pangram_reward import PangramReward
            # MELD is loaded as the fallback so an exhausted API key degrades the run instead of
            # pinning every completion to the floor, which has killed two runs silently.
            fb = MeldReward(r["fallback_dir"], device=r["fallback_device"],
                            min_words=G.MIN_WORDS, temp=r["temp"]) if r["fallback_dir"] else None
            fn = PangramReward(model=r["model"], temp=r["temp"], workers=r["workers"],
                               min_words=G.MIN_WORDS, fallback=fb)
            fn.__name__ = "pangram"
        elif name == "judge":
            from judge_reward import judge_reward as fn
        elif name == "rubric":
            from judge_reward import rubric_reward as fn
        elif name == "antislop":
            from judge_reward import antislop_reward as fn
        elif name == "antislop_think":
            from judge_reward import antislop_think_reward as fn
        elif name == "dlg_antislop":
            from judge_reward import dlg_antislop_reward as fn
        elif name == "continuation":
            from judge_reward import continuation_reward as fn
        elif name == "source_sim":
            from source_similarity import source_similarity_reward as fn
        elif name == "guards":
            fn = G.guards_reward
        elif name == "diversity":
            from diversity_reward import diversity_reward as fn
        elif name == "semdiv":
            from semantic_diversity import semantic_diversity_reward as fn
        elif name == "compliance":
            from compliance_reward import compliance_reward as fn
        elif name == "emu":
            from emu_reward import EmuReward
            fn = EmuReward()
        elif name == "task":
            from emu_reward import TaskReward
            fn = TaskReward()
        elif name == "length":
            from emu_reward import length_reward as fn
        else:                                            # unreachable: load() checked the name
            raise ConfigError(f"unknown reward {name!r}")

        funcs.append(fn)
        weights.append(w)

    if not funcs:
        raise ConfigError("every reward has weight 0; there is nothing to optimise")
    return funcs, weights


def describe(cfg: dict) -> str:
    terms = "  ".join(f"{r['name']}={r['weight']}" for r in cfg["rewards"] if r["weight"])
    t = cfg["train"]
    return (f"model   {cfg['model']['path']}\n"
            f"data    {cfg['data']['path'] or cfg['data']['hf']['id']}\n"
            f"rewards {terms}\n"
            f"train   {t['steps']} steps x {t['prompts_per_step']} prompts x "
            f"{t['num_generations']} samples, lr {t['lr']}, beta {t['beta']}\n"
            f"guards  {cfg['guards']['min_words']}-{cfg['guards']['target_words']}-"
            f"{cfg['guards']['max_words']} words, sentence cap "
            f"{cfg['guards']['max_sentence_words']}\n"
            f"out     {t['out']}")
