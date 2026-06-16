"""Equation symbolic reasoning generator.

Handles:
  1. All concatenation/reordering patterns of (A1, A2, B1, B2) — 40 patterns
  2. Arithmetic: addition, subtraction, multiplication, absolute-difference,
     with optional reversed-operands and/or reversed-result variants.

Input format: every example input is exactly 5 chars  A1 A2 OP B1 B2.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iproduct
from math import gcd
from typing import Callable

from reasoners.store_types import Problem


@dataclass
class _Ex:
    a: tuple[str, str]  # (A1, A2)
    op: str
    b: tuple[str, str]  # (B1, B2)
    out: str


def _box(s: str) -> str:
    return "".join(f"【{c}】" for c in s)


def _q(s: str) -> str:
    return f"【{s}】"


# ── Concat family ─────────────────────────────────────────────────────────────
# Every meaningful permutation/subset of (A1, A2, B1, B2).

_CONCAT_PATTERNS: list[tuple[str, Callable[[str, str, str, str], str]]] = [
    # 4-char (all 24 permutations; only the distinct ones listed)
    ("A1A2B1B2", lambda a1, a2, b1, b2: a1 + a2 + b1 + b2),
    ("B1B2A1A2", lambda a1, a2, b1, b2: b1 + b2 + a1 + a2),
    ("A1B1A2B2", lambda a1, a2, b1, b2: a1 + b1 + a2 + b2),
    ("A1B2B1A2", lambda a1, a2, b1, b2: a1 + b2 + b1 + a2),
    ("A1B1B2A2", lambda a1, a2, b1, b2: a1 + b1 + b2 + a2),
    ("A1B2A2B1", lambda a1, a2, b1, b2: a1 + b2 + a2 + b1),
    ("A2A1B1B2", lambda a1, a2, b1, b2: a2 + a1 + b1 + b2),
    ("A2A1B2B1", lambda a1, a2, b1, b2: a2 + a1 + b2 + b1),
    ("A2B1A1B2", lambda a1, a2, b1, b2: a2 + b1 + a1 + b2),
    ("A2B2A1B1", lambda a1, a2, b1, b2: a2 + b2 + a1 + b1),
    ("A2B1B2A1", lambda a1, a2, b1, b2: a2 + b1 + b2 + a1),
    ("A2B2B1A1", lambda a1, a2, b1, b2: a2 + b2 + b1 + a1),
    ("B1A1A2B2", lambda a1, a2, b1, b2: b1 + a1 + a2 + b2),
    ("B1A1B2A2", lambda a1, a2, b1, b2: b1 + a1 + b2 + a2),
    ("B1A2A1B2", lambda a1, a2, b1, b2: b1 + a2 + a1 + b2),
    ("B1A2B2A1", lambda a1, a2, b1, b2: b1 + a2 + b2 + a1),
    ("B1B2A2A1", lambda a1, a2, b1, b2: b1 + b2 + a2 + a1),
    ("B2A1A2B1", lambda a1, a2, b1, b2: b2 + a1 + a2 + b1),
    ("B2A1B1A2", lambda a1, a2, b1, b2: b2 + a1 + b1 + a2),
    ("B2A2A1B1", lambda a1, a2, b1, b2: b2 + a2 + a1 + b1),
    ("B2A2B1A1", lambda a1, a2, b1, b2: b2 + a2 + b1 + a1),
    ("B2B1A1A2", lambda a1, a2, b1, b2: b2 + b1 + a1 + a2),
    ("B2B1A2A1", lambda a1, a2, b1, b2: b2 + b1 + a2 + a1),
    ("A1A2B2B1", lambda a1, a2, b1, b2: a1 + a2 + b2 + b1),
    # 2-char
    ("A1A2",     lambda a1, a2, b1, b2: a1 + a2),
    ("A2A1",     lambda a1, a2, b1, b2: a2 + a1),
    ("B1B2",     lambda a1, a2, b1, b2: b1 + b2),
    ("B2B1",     lambda a1, a2, b1, b2: b2 + b1),
    ("A1B1",     lambda a1, a2, b1, b2: a1 + b1),
    ("A1B2",     lambda a1, a2, b1, b2: a1 + b2),
    ("A2B1",     lambda a1, a2, b1, b2: a2 + b1),
    ("A2B2",     lambda a1, a2, b1, b2: a2 + b2),
    ("B1A1",     lambda a1, a2, b1, b2: b1 + a1),
    ("B1A2",     lambda a1, a2, b1, b2: b1 + a2),
    ("B2A1",     lambda a1, a2, b1, b2: b2 + a1),
    ("B2A2",     lambda a1, a2, b1, b2: b2 + a2),
    # 1-char
    ("A1",       lambda a1, a2, b1, b2: a1),
    ("A2",       lambda a1, a2, b1, b2: a2),
    ("B1",       lambda a1, a2, b1, b2: b1),
    ("B2",       lambda a1, a2, b1, b2: b2),
]

_CONCAT_MAP: dict[str, Callable[[str, str, str, str], str]] = {
    name: fn for name, fn in _CONCAT_PATTERNS
}


def _is_concat_eligible(cases: list[_Ex]) -> bool:
    """
    A concat output can only contain characters drawn from {a1, a2, b1, b2}.
    If any output character falls outside that set, the rule cannot be a
    simple rearrangement — rule it out early to avoid false positives.
    """
    for ex in cases:
        allowed = {ex.a[0], ex.a[1], ex.b[0], ex.b[1]}
        if any(ch not in allowed for ch in ex.out):
            return False
    return True


def _detect_concat(cases: list[_Ex]) -> str | None:
    """Return first matching concat pattern name, or None."""
    if not _is_concat_eligible(cases):
        return None
    for name, fn in _CONCAT_PATTERNS:
        if all(fn(ex.a[0], ex.a[1], ex.b[0], ex.b[1]) == ex.out for ex in cases):
            return name
    return None


# ── Arithmetic family ─────────────────────────────────────────────────────────

_ARITH_OPS: list[tuple[str, Callable[[int, int], int]]] = [
    ("add",      lambda a, b: a + b),
    ("sub",      lambda a, b: a - b),
    ("mul",      lambda a, b: a * b),
    ("abs_diff", lambda a, b: abs(a - b)),
    ("div",      lambda a, b: a // b if b != 0 else 1_000_000),
    ("mod",      lambda a, b: a % b  if b != 0 else 1_000_000),
    ("max_op",   lambda a, b: max(a, b)),
    ("min_op",   lambda a, b: min(a, b)),
    ("gcd_op",   lambda a, b: gcd(a, b)),
]

_ARITH_OP_MAP: dict[str, Callable[[int, int], int]] = {
    name: fn for name, fn in _ARITH_OPS
}

_ARITH_OP_SYM: dict[str, str] = {
    "add": "+", "sub": "-", "mul": "×", "abs_diff": "|A-B|",
    "div": "÷", "mod": "%", "max_op": "max", "min_op": "min", "gcd_op": "gcd",
}

_MAX_CANDIDATES = 500


def _num_str(r: int) -> str:
    return str(r) if r >= 0 else f"-{-r}"


def _apply_result(r: int, rev_res: bool) -> str:
    s = _num_str(r)
    if rev_res:
        return ("-" + s[1:][::-1]) if s.startswith("-") else s[::-1]
    return s


def _get_operands(
    a1: str, a2: str, b1: str, b2: str,
    dmap: dict[str, int],
    rev_ops: bool,
) -> tuple[int, int]:
    if rev_ops:
        return dmap[a2] * 10 + dmap[a1], dmap[b2] * 10 + dmap[b1]
    return dmap[a1] * 10 + dmap[a2], dmap[b1] * 10 + dmap[b2]


def _match_result(
    r_str: str, out: str, dmap: dict[str, int]
) -> dict[str, int] | None:
    """
    Check whether numeric result string r_str is consistent with the output
    symbol string `out` under dmap.  Returns an extended map on success, None
    on mismatch.
    """
    neg = r_str.startswith("-")
    if neg != out.startswith("-"):
        return None
    r_body = r_str[1:] if neg else r_str
    o_body = out[1:] if neg else out
    if len(r_body) != len(o_body):
        return None
    m = dict(dmap)
    for d_ch, sym in zip(r_body, o_body):
        d = int(d_ch)
        if sym in m:
            if m[sym] != d:
                return None
        else:
            m[sym] = d
    return m


def _arith_solve(
    cases: list[_Ex],
    op_fn: Callable[[int, int], int],
    rev_ops: bool,
    rev_res: bool,
) -> dict[str, int] | None:
    """
    Find a symbol→digit mapping (0–9, many-to-one allowed) consistent with all
    cases under the given arithmetic operation and reversal flags.
    Returns the first valid mapping found, or None.
    """
    if not cases:
        return None

    first = cases[0]
    a1f, a2f, b1f, b2f = first.a[0], first.a[1], first.b[0], first.b[1]

    # Generate candidate mappings from the first example (100×100 = 10 000 tries)
    candidates: list[dict[str, int]] = []
    for va in range(100):
        for vb in range(100):
            d1, d2 = va // 10, va % 10
            d3, d4 = vb // 10, vb % 10
            dmap: dict[str, int] = {}
            valid = True
            for sym, d in zip([a1f, a2f, b1f, b2f], [d1, d2, d3, d4]):
                if sym in dmap and dmap[sym] != d:
                    valid = False
                    break
                dmap[sym] = d
            if not valid:
                continue
            va_use, vb_use = _get_operands(a1f, a2f, b1f, b2f, dmap, rev_ops)
            r_str = _apply_result(op_fn(va_use, vb_use), rev_res)
            ext = _match_result(r_str, first.out, dmap)
            if ext is not None:
                candidates.append(ext)

    if not candidates:
        return None

    # Filter through remaining examples
    for ex in cases[1:]:
        a1, a2, b1, b2 = ex.a[0], ex.a[1], ex.b[0], ex.b[1]
        surviving: list[dict[str, int]] = []
        for dmap in candidates[:_MAX_CANDIDATES]:
            unmapped = [s for s in [a1, a2, b1, b2] if s not in dmap]
            if unmapped:
                if len(unmapped) > 3:
                    continue
                for new_ds in iproduct(range(10), repeat=len(unmapped)):
                    ext = dict(dmap)
                    for s, d in zip(unmapped, new_ds):
                        ext[s] = d
                    va_use, vb_use = _get_operands(a1, a2, b1, b2, ext, rev_ops)
                    r_str = _apply_result(op_fn(va_use, vb_use), rev_res)
                    ext2 = _match_result(r_str, ex.out, ext)
                    if ext2 is not None:
                        surviving.append(ext2)
            else:
                va_use, vb_use = _get_operands(a1, a2, b1, b2, dmap, rev_ops)
                r_str = _apply_result(op_fn(va_use, vb_use), rev_res)
                ext = _match_result(r_str, ex.out, dmap)
                if ext is not None:
                    surviving.append(ext)
        candidates = surviving
        if not candidates:
            return None

    return candidates[0] if candidates else None


def _detect_arith(
    cases: list[_Ex],
    q_a: tuple[str, str] | None = None,
    q_b: tuple[str, str] | None = None,
) -> tuple[str, dict[str, int], bool, bool] | None:
    """
    Try all (op × rev_ops × rev_res) combinations.
    If q_a/q_b are given, collect all variants that can answer the question,
    then return the answer that has the most votes (convergence heuristic).
    Fall back to first match overall.
    """
    if q_a is None:
        for op_name, op_fn in _ARITH_OPS:
            for rev_ops in (False, True):
                for rev_res in (False, True):
                    dmap = _arith_solve(cases, op_fn, rev_ops, rev_res)
                    if dmap is not None:
                        return op_name, dmap, rev_ops, rev_res
        return None

    fallback: tuple[str, dict[str, int], bool, bool] | None = None
    # answer → (natural_votes, total_votes, first_tuple)
    # natural_votes: count of variants with rev_res=False (more canonical)
    answer_votes: dict[str, tuple[int, int, tuple[str, dict[str, int], bool, bool]]] = {}
    q_syms = [q_a[0], q_a[1], q_b[0], q_b[1]]

    for op_name, op_fn in _ARITH_OPS:
        for rev_ops in (False, True):
            for rev_res in (False, True):
                dmap = _arith_solve(cases, op_fn, rev_ops, rev_res)
                if dmap is None:
                    continue
                if fallback is None:
                    fallback = (op_name, dmap, rev_ops, rev_res)
                if all(s in dmap for s in q_syms):
                    va, vb = _get_operands(q_a[0], q_a[1], q_b[0], q_b[1], dmap, rev_ops)
                    r_str = _apply_result(op_fn(va, vb), rev_res)
                    ans = _encode_result(r_str, dmap)
                    if ans is not None:
                        nat = 0 if rev_res else 1
                        entry = answer_votes.get(ans)
                        if entry is None:
                            answer_votes[ans] = (nat, 1, (op_name, dmap, rev_ops, rev_res))
                        else:
                            answer_votes[ans] = (entry[0] + nat, entry[1] + 1, entry[2])

    if answer_votes:
        # Prefer answers with most natural (rev_res=False) votes, then total votes
        best_ans = max(answer_votes, key=lambda a: (answer_votes[a][0], answer_votes[a][1]))
        return answer_votes[best_ans][2]
    return fallback


def _encode_result(r_str: str, dmap: dict[str, int]) -> str | None:
    """Re-encode a numeric result string back into symbols using dmap."""
    rev: dict[int, str] = {}
    for sym, d in dmap.items():
        if d not in rev:
            rev[d] = sym
    result = ""
    for ch in r_str:
        if ch == "-":
            result += "-"
        else:
            d = int(ch)
            if d not in rev:
                return None
            result += rev[d]
    return result


# ── Trace builders ────────────────────────────────────────────────────────────

def _trace_concat(
    exs: list[_Ex],
    op_rules: dict[str, _Rule],
    q_a: tuple[str, str],
    q_b: tuple[str, str],
    q_op: str,
    q_pattern: str,
    answer: str,
) -> list[str]:
    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    for ex in exs:
        a1, a2 = ex.a
        b1, b2 = ex.b
        orig_inp = a1 + a2 + ex.op + b1 + b2
        fwd = a1 + a2 + b1 + b2
        rev = b1 + b2 + a1 + a2
        lines.append(f"{_q(orig_inp)} = {_q(ex.out)}")
        lines.append(f"  input: {_box(a1+a2)}{_q(ex.op)}{_box(b1+b2)}")
        lines.append(f"  left:{_box(a1+a2)}")
        lines.append(f"  operator: {_q(ex.op)}")
        lines.append(f"  right:{_box(b1+b2)}")
        lines.append(f"  output: {_box(ex.out)}")
        lines.append(
            f"  concatenation: {_box(fwd)} {'match' if ex.out == fwd else 'mismatch'}"
        )
        lines.append(
            f"  reverse concatenation: {_box(rev)} {'match' if ex.out == rev else 'mismatch'}"
        )
        rule = op_rules.get(ex.op)
        if rule and rule[0] == "concat":
            lines.append(f"  operator: {_q(ex.op)} pattern={rule[1]}")
        else:
            lines.append(f"  operator: {_q(ex.op)} unknown")
        lines.append("")

    qa0, qa1 = _q(q_a[0]), _q(q_a[1])
    qb0, qb1 = _q(q_b[0]), _q(q_b[1])
    lines.append(f"Question{_q(q_a[0]+q_a[1]+q_op+q_b[0]+q_b[1])}")
    lines.append(f"  input: {qa0}{qa1}{_q(q_op)}{qb0}{qb1}")
    lines.append(f"  left:{qa0}{qa1}")
    lines.append(f"  operator:{_q(q_op)}")
    lines.append(f"  right:{qb0}{qb1}")
    lines.append("")
    lines.append(f"  pattern {q_pattern}: {_box(answer)}")
    lines.append(f"  output: {_q(answer)} -> {_q('{' + answer + '}')}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
    return lines


def _trace_arith(
    op_exs: list[_Ex],
    op_name: str,
    op_fn: Callable[[int, int], int],
    dmap: dict[str, int],
    rev_ops: bool,
    rev_res: bool,
    q_a: tuple[str, str],
    q_b: tuple[str, str],
    q_op: str,
    answer: str,
) -> list[str]:
    sym = _ARITH_OP_SYM[op_name]
    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    lines.append("Symbol-to-digit mapping:")
    for s in sorted(dmap, key=lambda x: dmap[x]):
        lines.append(f"  {_q(s)} = {dmap[s]}")
    lines.append("")
    desc_parts = [op_name]
    if rev_ops:
        desc_parts.append("operands reversed")
    if rev_res:
        desc_parts.append("result reversed")
    lines.append(f"Operator {_q(q_op)}: {', '.join(desc_parts)}")
    lines.append("")
    lines.append("Verifying examples:")
    for ex in op_exs:
        a1, a2 = ex.a
        b1, b2 = ex.b
        if any(s not in dmap for s in [a1, a2, b1, b2]):
            continue
        va, vb = _get_operands(a1, a2, b1, b2, dmap, rev_ops)
        r = op_fn(va, vb)
        r_str = _apply_result(r, rev_res)
        lines.append(
            f"  {_box(a1+a2)} {sym} {_box(b1+b2)}"
            f" = {va} {sym} {vb} = {r} -> {r_str} -> {_q(ex.out)}"
        )
    lines.append("")
    a1, a2 = q_a
    b1, b2 = q_b
    va = dmap.get(a2 if rev_ops else a1, 0) * 10 + dmap.get(a1 if rev_ops else a2, 0)
    vb = dmap.get(b2 if rev_ops else b1, 0) * 10 + dmap.get(b1 if rev_ops else b2, 0)
    r = op_fn(va, vb)
    r_str = _apply_result(r, rev_res)
    lines.append(
        f"Question: {_box(a1+a2)} {sym} {_box(b1+b2)}"
        f" = {va} {sym} {vb} = {r} -> {r_str} -> {_q(answer)}"
    )
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
    return lines


def _trace_generic(
    exs: list[_Ex],
    dsl_rules: dict,
    q_rule: object,
    q_a: tuple[str, str],
    q_b: tuple[str, str],
    q_op: str,
    answer: str,
) -> list[str]:
    """Generic chain-of-thought trace for DSL operator families beyond concat/arith."""
    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    rule_name = type(q_rule).__name__
    lines.append(f"Operator {_q(q_op)}: rule family = {rule_name}")
    lines.append("")
    lines.append("Verifying examples:")
    for ex in exs:
        if ex.op != q_op:
            continue
        from reasoners.cryptarithm_dsl import apply_to_question
        pred = apply_to_question(q_rule, ex.a, ex.b, ex.op)
        match = "✓" if pred == ex.out else "✗"
        lines.append(
            f"  {_box(ex.a[0]+ex.a[1])} {_q(ex.op)} {_box(ex.b[0]+ex.b[1])} "
            f"= {_q(ex.out)} {match}"
        )
    lines.append("")
    qa0, qa1 = _q(q_a[0]), _q(q_a[1])
    qb0, qb1 = _q(q_b[0]), _q(q_b[1])
    lines.append(f"Question: {qa0}{qa1}{_q(q_op)}{qb0}{qb1}")
    lines.append(f"  Applying {rule_name} → {_q(answer)}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
    return lines


# ── Main entry point ──────────────────────────────────────────────────────────

def reasoning_cryptarithm(problem: Problem) -> str | None:
    """Generate reasoning for cryptarithm problems."""
    exs: list[_Ex] = []
    for ex in problem.examples:
        inp = str(ex.input_value)
        if len(inp) != 5:
            return None
        exs.append(_Ex(
            a=(inp[0], inp[1]),
            op=inp[2],
            b=(inp[3], inp[4]),
            out=str(ex.output_value),
        ))

    q = str(problem.question)
    if len(q) != 5:
        return None
    q_a = (q[0], q[1])
    q_op = q[2]
    q_b = (q[3], q[4])

    # Group examples by operator
    by_op: dict[str, list[_Ex]] = {}
    for ex in exs:
        by_op.setdefault(ex.op, []).append(ex)

    op_rules: dict[str, tuple] = {}  # populated by detection then used by trace builders

    # ── Fast path: per-operator detection (concat + per-op arithmetic) ─────────
    for op, cases in by_op.items():
        pat = _detect_concat(cases)
        if pat is not None:
            op_rules[op] = ("concat", pat)
        else:
            hint_a = q_a if op == q_op else None
            hint_b = q_b if op == q_op else None
            arith = _detect_arith(cases, hint_a, hint_b)
            if arith is not None:
                op_name, dmap, rev_ops, rev_res = arith
                op_rules[op] = ("arith", op_name, dmap, rev_ops, rev_res)

    def _arith_dmaps_consistent() -> bool:
        global_dmap: dict[str, int] = {}
        for rule in op_rules.values():
            if rule[0] != "arith":
                continue
            for sym, d in rule[2].items():
                if sym in global_dmap and global_dmap[sym] != d:
                    return False
                global_dmap[sym] = d
        return True

    _fast_path_ok = (
        q_op in op_rules
        and _arith_dmaps_consistent()
    )

    if _fast_path_ok:
        # Use existing tuple-based path for speed
        q_rule: tuple = op_rules[q_op]
        if q_rule[0] == "concat":
            pattern = q_rule[1]
            answer = _CONCAT_MAP[pattern](q_a[0], q_a[1], q_b[0], q_b[1])
            return "\n".join(_trace_concat(exs, op_rules, q_a, q_b, q_op, pattern, answer))
        else:
            _, op_name, dmap, rev_ops, rev_res = q_rule
            op_fn = _ARITH_OP_MAP[op_name]
            # For q_syms not in the per-op dmap, check if they appear in other
            # arith operators' dmaps (all consistent because _fast_path_ok).
            # Extend the per-op dmap only with those missing symbols so that
            # _encode_result still uses the per-op symbol set for the output.
            missing_q = [s for s in [q_a[0], q_a[1], q_b[0], q_b[1]] if s not in dmap]
            if missing_q:
                ext: dict[str, int] = dict(dmap)
                for r in op_rules.values():
                    if r[0] != "arith":
                        continue
                    for s in missing_q:
                        if s in r[2] and s not in ext:
                            ext[s] = r[2][s]
                use_dmap = ext
            else:
                use_dmap = dmap
            if not any(s not in use_dmap for s in [q_a[0], q_a[1], q_b[0], q_b[1]]):
                va, vb = _get_operands(q_a[0], q_a[1], q_b[0], q_b[1], use_dmap, rev_ops)
                r_str = _apply_result(op_fn(va, vb), rev_res)
                answer = _encode_result(r_str, use_dmap)
                if answer is not None:
                    op_exs = by_op.get(q_op, exs)
                    lines = _trace_arith(op_exs, op_name, op_fn, use_dmap, rev_ops, rev_res,
                                         q_a, q_b, q_op, answer)
                    return "\n".join(lines)
            # Fast path can't answer — fall through to slow path DSL

    # ── Slow path: full DSL cascade (set ops, char map, global arith, bit ops) ─
    from reasoners.cryptarithm_dsl import (
        solve_global,
        apply_to_question,
        RuleConcat, RuleArith,
    )

    # Save fast-path q_op arith rule (may be overwritten/replaced by DSL)
    _saved_fast_q_rule = op_rules.get(q_op)

    dsl_rules = solve_global(by_op, q_op=q_op, q_a=q_a, q_b=q_b)

    # Translate DSL rules back to legacy tuple format for existing trace builders,
    # and extract the rule for the question operator.
    if dsl_rules is not None:
        # Back-fill op_rules for trace builder compatibility
        for op, rule in dsl_rules.items():
            if isinstance(rule, RuleConcat):
                op_rules[op] = ("concat", rule.pattern)
            elif isinstance(rule, RuleArith):
                op_rules[op] = ("arith", rule.op_name, rule.dmap, rule.rev_ops, rule.rev_res)
            else:
                op_rules[op] = ("dsl", rule)

    # Determine the rule for the question operator
    q_dsl_rule = dsl_rules.get(q_op) if dsl_rules else None

    if q_dsl_rule is None and q_op not in by_op:
        # Operator never seen in any example → default to fwd concat
        q_dsl_rule = RuleConcat("A1A2B1B2")
        op_rules[q_op] = ("concat", "A1A2B1B2")

    if q_dsl_rule is None:
        return None

    # Compute answer via DSL
    answer = apply_to_question(q_dsl_rule, q_a, q_b, q_op)
    if answer is None and _saved_fast_q_rule is not None and _saved_fast_q_rule[0] == "arith":
        # Fast path found a per-op arith for q_op; try it as fallback
        _, fn_op_name, fp_dmap, fp_rev_ops, fp_rev_res = _saved_fast_q_rule
        fp_op_fn = _ARITH_OP_MAP[fn_op_name]
        q_syms = [q_a[0], q_a[1], q_b[0], q_b[1]]
        if all(s in fp_dmap for s in q_syms):
            va, vb = _get_operands(q_a[0], q_a[1], q_b[0], q_b[1], fp_dmap, fp_rev_ops)
            r_str = _apply_result(fp_op_fn(va, vb), fp_rev_res)
            _fb_ans = _encode_result(r_str, fp_dmap)
            if _fb_ans is not None:
                answer = _fb_ans
                q_dsl_rule = RuleArith(fn_op_name, fp_rev_ops, fp_rev_res, fp_dmap)
    if answer is None:
        return None

    # Generate trace
    if isinstance(q_dsl_rule, RuleConcat):
        lines = _trace_concat(exs, op_rules, q_a, q_b, q_op, q_dsl_rule.pattern, answer)

    elif isinstance(q_dsl_rule, RuleArith):
        dmap = q_dsl_rule.dmap
        op_name = q_dsl_rule.op_name
        op_fn = _ARITH_OP_MAP[op_name]
        rev_ops = q_dsl_rule.rev_ops
        rev_res = q_dsl_rule.rev_res
        op_exs = by_op.get(q_op, exs)
        lines = _trace_arith(op_exs, op_name, op_fn, dmap, rev_ops, rev_res, q_a, q_b, q_op, answer)

    else:
        # Generic trace for new operator families
        lines = _trace_generic(exs, dsl_rules or {}, q_dsl_rule, q_a, q_b, q_op, answer)

    return "\n".join(lines)
