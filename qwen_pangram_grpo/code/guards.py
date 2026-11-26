"""Cheap dependency-free guard penalties for GRPO, plus the independent eval metrics.

These keep the policy from gaming the AI-detector reward with rambling, looping, truncated or
wrong-language text. Everything here is pure python and runs in microseconds.

`slop_lexicon.score` is deliberately NOT part of the reward: it stays an independent held-out
metric, so a drop in slop-lexicon rate after training is evidence rather than tautology.
"""
from __future__ import annotations

import os

import re
from collections import Counter

# Length targets, overridable per run. The defaults below are for SHORT answers without thinking.
# A thinking run wants a longer corridor: reasoning absorbs the "work" the model used to do out loud
# in the answer, so the answer can afford to be fuller without padding. Set them from the launcher.
MIN_WORDS = int(os.environ.get("GUARD_MIN_WORDS", "80"))
TARGET_WORDS = int(os.environ.get("GUARD_TARGET_WORDS", "110"))
MAX_WORDS = int(os.environ.get("GUARD_MAX_WORDS", "400"))
#: how hard the length term pushes. It has to be large enough to out-spread a detector
#: whose own within-group spread is ~0.7, or the corridor is decorative.
LENGTH_SCALE = float(os.environ.get("GUARD_LENGTH_SCALE", "0.5"))
MAX_SENTENCE_WORDS = int(os.environ.get("GUARD_MAX_SENTENCE_WORDS", "30"))
LONG_SENTENCE_SCALE = float(os.environ.get("GUARD_LONG_SENTENCE_SCALE", "0.0"))
NEVER_FINISHED_THINKING = -2.0   # ran out of budget mid-thought: the user gets nothing
_SENT_SPLIT = re.compile(r"[.!?…]+")
_MD = re.compile(r"^#+ |^\s*([-*]|\d+[.)]) |\*\*", re.M)
_CYR = re.compile(r"[а-яё]", re.I)
_FINISHED = re.compile(r"[.!?…\"'»)]\s*$")


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


def markdown_marks(text: str) -> int:
    return len(_MD.findall(text))


def max_ngram_repeat(text: str, n: int = 4) -> int:
    w = text.split()
    if len(w) <= n:
        return 0
    c = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return max(c.values())


def is_russian(text: str) -> bool:
    letters = sum(ch.isalpha() for ch in text)
    return letters > 0 and len(_CYR.findall(text)) / letters > 0.2


_STOP_EN = set("the a an and or but to of in on at for with is are was were be been it its this that i you he she we they my your his her our their me him them not no so as if do does did have has had will would can could just very".split())
_LETTER_RUNS = re.compile(r"[^\W\d_]+", re.U)


def coherence_penalty(prompt: str, text: str) -> float:
    """Catch the token-salad collapse seen after ~130 steps in pgemu1 (multilingual word soup the emulator
    scored as human). English text has >=25% function words; salad has few and mixes scripts."""
    toks = [t.lower() for t in _LETTER_RUNS.findall(text)]
    if len(toks) < 15:
        return 0.0
    pen = 0.0
    if not is_russian(prompt):
        stop = sum(t in _STOP_EN for t in toks) / len(toks)
        if stop < 0.18:
            pen -= 1.5
        elif stop < 0.24:
            pen -= 0.5
        letters = [c for c in text if c.isalpha()]
        foreign = sum(ord(c) > 0x24F for c in letters) / max(1, len(letters))   # beyond Latin Extended
        if foreign > 0.01:
            pen -= min(1.5, 30 * foreign)
    weird = sum(len(t) > 18 for t in toks) / len(toks)
    if weird > 0.02:
        pen -= 0.5
    return max(-2.0, pen)


def guard_detail(prompt: str, text: str) -> dict:
    """Per-completion penalty breakdown. Total is <= 0."""
    words = len(text.split())
    d = {"words": words}
    if words < TARGET_WORDS:
        # The ramp width must follow the corridor, not a constant. With a hardcoded 80 and a target of
        # 250 the term saturated for every answer under 170 words: the whole group got the identical
        # penalty, within-group spread collapsed to 0.02, and the length signal did nothing at all
        # while the detector took 83% of the gradient. Scaling by the corridor keeps it informative.
        d["length"] = -LENGTH_SCALE * min(1.0, (TARGET_WORDS - words) / max(1.0, TARGET_WORDS - MIN_WORDS))
    elif words > MAX_WORDS:
        d["length"] = -LENGTH_SCALE * min(1.0, (words - MAX_WORDS) / max(1.0, 0.5 * MAX_WORDS))
    else:
        d["length"] = 0.0
    # The previous run kept a decent average Flesch while emitting single sentences of up to 75
    # words: an average cannot see one monster among ten short ones, so this penalises the longest
    # sentence rather than the mean.
    longest = max((len(x.split()) for x in _SENT_SPLIT.split(text) if x.strip()), default=0)
    d["long_sentence"] = -LONG_SENTENCE_SCALE * min(1.0, max(0, longest - MAX_SENTENCE_WORDS) / 25.0)
    d["markdown"] = -0.3 * min(markdown_marks(text), 3)
    d["loop"] = -1.0 if max_ngram_repeat(text) >= 4 else 0.0
    d["unfinished"] = 0.0 if _FINISHED.search(text.strip()) else -0.3
    d["language"] = -1.0 if is_russian(prompt) != is_russian(text) else 0.0
    d["empty"] = -1.0 if words < 15 else 0.0
    letters = [c for c in text if c.isalpha()]
    upper = sum(c.isupper() for c in letters) / max(1, len(letters))
    d["shouting"] = -1.0 if upper > 0.4 else 0.0        # temp-1.0 rollouts sometimes go all-caps
    d["coherence"] = coherence_penalty(prompt, text)
    d["total"] = sum(v for k, v in d.items() if k != "words")
    return d


def guards_reward(prompts, completions, **kwargs) -> list[float]:
    """TRL reward function: the (negative) sum of guard penalties, measured on the answer.

    With thinking on, a rollout that never closed its thought has no answer at all. That is worse
    than a bad answer, because the user is left with nothing, so it takes a flat penalty instead of
    being scored as an empty string.
    """
    from thinking import split_thinking
    qs = [str(q) for q in kwargs["question"]] if "question" in kwargs else [_text(p) for p in prompts]
    out = []
    for q, c in zip(qs, completions):
        raw = _text(c)
        if EXPECT_THINKING:
            _, answer, closed = split_thinking(raw)
            if not closed:
                out.append(NEVER_FINISHED_THINKING)
                continue
        else:
            answer = raw
        out.append(guard_detail(q, answer)["total"])
    return out


guards_reward.__name__ = "guards"


def length_ok(text: str) -> bool:
    return MIN_WORDS <= len(text.split()) <= MAX_WORDS
