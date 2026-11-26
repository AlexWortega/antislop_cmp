"""Reward answers for differing from the other completions of the SAME question.

GRPO already samples a group per prompt, so diversity is free to measure: each completion is
scored by how unlike its siblings it is. Similarity is lexical (character 4-gram and word
bigram Jaccard, averaged), which needs no model and costs microseconds.

This term is easy to game with gibberish, so it must always be combined with the quality guards
and the judge; on its own it would reward noise.
"""
from __future__ import annotations

import os

import re
from collections import Counter

_WORD = re.compile(r"\w+", re.UNICODE)


def _text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x:
        return x[-1].get("content", "") if isinstance(x[-1], dict) else str(x[-1])
    if isinstance(x, dict):
        return x.get("content", "")
    return str(x)

# Rollouts may carry private reasoning before the answer. Everything in this file describes what a
# reader gets, so it must see the answer only: scoring the reasoning would let the policy hide slop
# in the answer behind a long tidy monologue, and would spend the reward on text nobody reads.
EXPECT_THINKING = os.environ.get("EXPECT_THINKING", "0") == "1"


def _reader_text(x) -> str:
    from thinking import answer_of
    return answer_of(_text(x), EXPECT_THINKING)


def _shingles(text: str) -> tuple[set, set]:
    t = " ".join(_WORD.findall(text.lower()))
    chars = {t[i:i + 4] for i in range(max(0, len(t) - 3))}
    words = t.split()
    bigrams = {(words[i], words[i + 1]) for i in range(max(0, len(words) - 1))}
    return chars, bigrams


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def pairwise_similarity(t1: str, t2: str) -> float:
    c1, w1 = _shingles(t1)
    c2, w2 = _shingles(t2)
    return 0.5 * _jaccard(c1, c2) + 0.5 * _jaccard(w1, w2)


def group_diversity(texts: list[str]) -> list[float]:
    """1 - mean similarity to siblings, per completion. Range [0, 1], higher = more distinct."""
    n = len(texts)
    if n < 2:
        return [0.5] * n
    shing = [_shingles(t) for t in texts]
    out = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            sims.append(0.5 * _jaccard(shing[i][0], shing[j][0]) + 0.5 * _jaccard(shing[i][1], shing[j][1]))
        out.append(1.0 - sum(sims) / len(sims))
    return out


def diversity_reward(prompts, completions, **kwargs) -> list[float]:
    """TRL reward function: groups the flat batch by prompt and scores distinctness within group."""
    texts = [_reader_text(c) for c in completions]
    keys = [str(q) for q in kwargs["question"]] if "question" in kwargs else [_text(p) for p in prompts]
    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    out = [0.5] * len(texts)
    for idxs in groups.values():
        for pos, i in enumerate(idxs):
            out[i] = group_diversity([texts[j] for j in idxs])[pos]
    return out


diversity_reward.__name__ = "diversity"


if __name__ == "__main__":
    a = "Birds are warm-blooded vertebrates that evolved from dinosaurs and have feathers."
    b = "Birds are warm-blooded vertebrates that evolved from dinosaurs and have feathers!"
    c = "The short answer is oil glands. Your palms simply do not have them, so no pimples."
    print([round(x, 3) for x in group_diversity([a, b, c])])
