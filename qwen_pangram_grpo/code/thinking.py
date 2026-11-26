"""Split a rollout into its reasoning and its answer.

With thinking enabled the prompt ends on an open `<think>` tag, so a completion looks like

    <reasoning ...>\n</think>\n\n<answer ...>

and the two halves must be scored by different things. The detector, the guards and the diversity
term all describe what a *reader* gets, so they see the answer only; feeding them the reasoning
would train the model to think human-sounding thoughts, which is not the goal and would also let it
hide slop in the answer behind a long tidy monologue. The judge is the one term that sees both,
because reasoning quality is worth rewarding on its own.

A completion with no closing tag never finished thinking. Its answer is empty, which the length
guard already punishes, and `closed=False` lets a caller punish it explicitly.
"""
from __future__ import annotations

CLOSE = "</think>"
OPEN = "<think>"


def split_thinking(text: str) -> tuple[str, str, bool]:
    """Return (reasoning, answer, closed)."""
    if not text:
        return "", "", True
    body = text
    # the template usually leaves the opening tag in the prompt, but a model that re-emits it
    # should not have it counted as reasoning content
    stripped = body.lstrip()
    if stripped.startswith(OPEN):
        body = stripped[len(OPEN):]
    idx = body.find(CLOSE)
    if idx < 0:
        return body.strip(), "", False
    return body[:idx].strip(), body[idx + len(CLOSE):].strip(), True


def answer_of(text: str, expect_thinking: bool = True) -> str:
    """The part a reader sees.

    A missing closing tag is ambiguous on its own: it means either that thinking was never turned
    on, or that the model never stopped thinking. Those deserve opposite treatment, so the caller
    states which mode the rollout came from rather than letting this guess.
    """
    if not expect_thinking:
        return text
    _, answer, closed = split_thinking(text)
    return answer if closed else ""


def reasoning_of(text: str, expect_thinking: bool = True) -> str:
    return split_thinking(text)[0] if expect_thinking else ""


if __name__ == "__main__":
    cases = [
        "Let me work it out. Step one.\n</think>\n\nThe answer is oil glands.",
        "No thinking here at all, just an answer.",
        "I am still thinking and never stopped",
        "<think>\nrestated tag\n</think>\n\nDone.",
    ]
    for c in cases:
        r, a, closed = split_thinking(c)
        print(f"closed={closed} | reasoning={r[:28]!r} | answer={a[:34]!r}"
              f" | answer_of(no-thinking)={answer_of(c, False)[:20]!r}")
