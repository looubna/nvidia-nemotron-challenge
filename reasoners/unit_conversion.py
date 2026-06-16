"""Unit conversion: output = factor * input reasoning generator."""

from __future__ import annotations

import math
from collections import Counter

from reasoners.store_types import (
    Problem,
    cast_dp_pair,
    long_division_lines,
    long_multiplication_lines,
    truncate_3dp,
)


def _dp(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def reasoning_unit_conversion(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append(
        "We need to find a conversion rule that maps the inputs to outputs. "
        "Let me check if it's a linear factor."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    out_dp = max((_dp(str(ex.output_value)) for ex in problem.examples), default=2)

    factor_strs: list[str] = []
    for ex in problem.examples:
        inp = float(ex.input_value)
        if inp != 0:
            out_str = truncate_3dp(ex.output_value)
            inp_str = truncate_3dp(ex.input_value)
            lines.append(f"{ex.input_value} -> {ex.output_value}")
            inp_cast, out_cast, inp_dp, cast_out_dp = cast_dp_pair(inp_str, out_str)
            lines.append(
                f"Casting input to {inp_dp} decimal places, "
                f"output to {cast_out_dp} decimal places: "
                f"{inp_cast} -> {out_cast}"
            )
            lines.append(f"factor = {out_cast} / {inp_cast}")
            div_lines, factor_str = long_division_lines(out_cast, inp_cast)
            lines.extend(div_lines)
            lines.append(f"= {factor_str}")
            factor_strs.append(factor_str)
            lines.append("")

    if not factor_strs:
        return None

    factors = [float(s) for s in factor_strs]

    # List factor values and pick median (for even count, use the smaller middle value)
    f_list_str = ", ".join(factor_strs)
    lines.append(f"factor values: {f_list_str}")
    paired = sorted(zip(factors, factor_strs))
    sorted_str = ", ".join(s for _, s in paired)
    lines.append(f"factor values (sorted): {sorted_str}")
    n = len(paired)
    if n % 2 == 0 and n >= 2:
        f_lo, med_factor_str_lo = paired[n // 2 - 1]
        f_hi, med_factor_str_hi = paired[n // 2]
    else:
        mid = n // 2
        f_lo, med_factor_str_lo = paired[mid]
        f_hi, med_factor_str_hi = paired[mid]
    lines.append(f"The median factor is {med_factor_str_lo}.")

    q = float(problem.question)
    q_str = problem.question

    # Generate candidates: both medians × (round, floor)
    candidates: list[str] = []
    for f in (f_lo, f_hi):
        val = f * q
        candidates.append(f"{round(val, out_dp):.{out_dp}f}")
        truncated = math.floor(val * 10**out_dp) / 10**out_dp
        candidates.append(f"{truncated:.{out_dp}f}")

    votes = Counter(candidates)
    boxed_answer = votes.most_common(1)[0][0]

    med_display = med_factor_str_lo.rstrip("0").rstrip(".")
    lines.append("")
    lines.append(f"Converting {q_str}:")
    lines.append(f"{q_str} * {med_display} = {boxed_answer}")

    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{boxed_answer}}}")
    return "\n".join(lines)
