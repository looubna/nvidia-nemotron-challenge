"""Gravity: d = k * t^2 reasoning generator."""

from __future__ import annotations

import math
from collections import Counter

from reasoners.store_types import Problem


def _dp(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def reasoning_gravity(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append(
        "We need to determine the falling distance using d = k*t^2. "
        "Let me find k from the examples."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    out_dp = max((_dp(str(ex.output_value)) for ex in problem.examples), default=2)

    pairs = [
        (float(ex.input_value), float(ex.output_value))
        for ex in problem.examples
        if float(ex.input_value) > 0
    ]
    if not pairs:
        return None

    raw_ks = sorted(d / (t * t) for t, d in pairs)
    for ex in problem.examples:
        t = float(ex.input_value)
        if t > 0:
            k = float(ex.output_value) / (t * t)
            lines.append(
                f"t = {ex.input_value}s, d = {ex.output_value}m: "
                f"k = {ex.output_value} / {ex.input_value}^2 = {k:.6f}"
            )

    lines.append(f"\nk values (sorted): {', '.join(f'{k:.6f}' for k in raw_ks)}")
    n = len(raw_ks)
    k_lo = raw_ks[n // 2 - 1] if n % 2 == 0 else raw_ks[n // 2]
    k_hi = raw_ks[n // 2]
    lines.append(f"Median k = {k_lo:.6f}")

    q = float(problem.question)
    q2 = q * q

    # Generate candidates: both medians × (round, floor)
    candidates: list[str] = []
    for k in (k_lo, k_hi):
        val = k * q2
        candidates.append(f"{round(val, out_dp):.{out_dp}f}")
        truncated = math.floor(val * 10**out_dp) / 10**out_dp
        candidates.append(f"{truncated:.{out_dp}f}")

    # Pick the most common candidate (prefer round on tie by order)
    votes = Counter(candidates)
    boxed_answer = votes.most_common(1)[0][0]

    lines.append(f"\nFor t = {problem.question}:")
    lines.append(f"t^2 = {problem.question}^2 = {q2:.4f}")
    lines.append(f"d = {k_lo:.6f} * {q2:.4f} = {boxed_answer}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{boxed_answer}}}")
    return "\n".join(lines)
