"""Diversity of what the answers SAY, not of how they spell it.

The lexical measure this replaces compares character 4-grams and word bigrams. Eight samples of one
prompt already score about 0.85 on it, so it sits near its ceiling and its within-group spread is
0.01-0.03 against the detector's 0.14-0.6. At weight 4.0 it held four to six percent of the gradient
and moved in none of three runs. Raising the weight multiplies a number that is almost constant.

Two answers can be worded completely differently and make exactly the same point, which is the case
the lexical measure cannot see and the one that matters: the failure being fixed is eight rephrasings
of one idea. Embedding distance sees content, so it has headroom where the lexical measure does not.
"""
from __future__ import annotations

import os
import threading

_MODEL = None
_LOCK = threading.Lock()
MODEL_NAME = os.environ.get("DIVERSITY_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = os.environ.get("DIVERSITY_EMBED_DEVICE", "cuda:0")


def _model():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer(MODEL_NAME, device=DEVICE)
        return _MODEL


def _text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x:
        return x[-1].get("content", "") if isinstance(x[-1], dict) else str(x[-1])
    if isinstance(x, dict):
        return x.get("content", "")
    return str(x)


def group_diversity(texts: list[str]) -> list[float]:
    """1 - mean cosine similarity to the other members, per text. Range roughly [0, 1]."""
    n = len(texts)
    if n < 2:
        return [0.5] * n
    import numpy as np
    emb = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    sim = np.asarray(emb) @ np.asarray(emb).T
    np.fill_diagonal(sim, 0.0)
    mean_sim = sim.sum(axis=1) / (n - 1)
    return [float(1.0 - s) for s in mean_sim]


def semantic_diversity_reward(prompts, completions, **kwargs) -> list[float]:
    """TRL reward function: how much each answer differs in content from its siblings."""
    from thinking import answer_of
    import guards as G

    texts = [answer_of(_text(c), G.EXPECT_THINKING) for c in completions]
    keys = [str(q) for q in kwargs["question"]] if "question" in kwargs else [_text(p) for p in prompts]
    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    out = [0.5] * len(texts)
    for idxs in groups.values():
        vals = group_diversity([texts[i] or " " for i in idxs])
        for pos, i in enumerate(idxs):
            out[i] = vals[pos]
    return out


semantic_diversity_reward.__name__ = "semdiv"
