"""DSL-based cascade solver for cryptarithm A1A2 OP B1B2 → output puzzles.

For each operator's examples, we walk through transformation families in order.
Each family generates candidate rules; for each candidate we immediately check
ALL examples — if any example fails we skip to the next candidate.
The first rule that passes every example is returned.

Cascade order (cheapest first):
  1. Concat       – 40 rearrangement patterns of (A1,A2,B1,B2)
  2. SetOp        – Union / Intersection / Difference of char sets
  3. PosOp        – fixed-index subsets of [A1,A2,B1,B2]
  4. UnaryOp      – Reverse / Sort applied to left or right string
  5. CharMap      – learned bijective sym→sym substitution
  6. BitOp        – XOR / AND / OR on ASCII codes
  7. Arithmetic   – add / sub / mul / abs_diff with symbol→digit mapping
                    (per-operator first; global joint search if inconsistent)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iproduct
from typing import Callable

from reasoners.cryptarithm import (
    _ARITH_OP_MAP,
    _ARITH_OPS,
    _CONCAT_MAP,
    _CONCAT_PATTERNS,
    _Ex,
    _apply_result,
    _encode_result,
    _get_operands,
    _is_concat_eligible,
    _match_result,
)


# ══════════════════════════════════════════════════════════════════════════════
# Rule dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RuleConcat:
    pattern: str

@dataclass
class RuleSetOp:
    name: str

@dataclass
class RulePosOp:
    name: str

@dataclass
class RuleUnaryOp:
    name: str

@dataclass
class RuleCharMap:
    pattern_name: str
    smap: dict[str, str]
    key_fn: Callable[[_Ex], list[str]]

@dataclass
class RuleBitOp:
    pair_name: str
    fn_name: str

@dataclass
class RuleTernaryBitOp:
    triple_name: str
    fn_name: str

@dataclass
class RuleArith:
    op_name: str
    rev_ops: bool
    rev_res: bool
    dmap: dict[str, int]

    @property
    def fn(self) -> Callable[[int, int], int]:
        return _ARITH_OP_MAP[self.op_name]


Rule = RuleConcat | RuleSetOp | RulePosOp | RuleUnaryOp | RuleCharMap | RuleBitOp | RuleTernaryBitOp | RuleArith


# ══════════════════════════════════════════════════════════════════════════════
# Rule application
# ══════════════════════════════════════════════════════════════════════════════

def _apply_rule(rule: Rule, ex: _Ex) -> str | None:
    a1, a2, b1, b2 = ex.a[0], ex.a[1], ex.b[0], ex.b[1]

    if isinstance(rule, RuleConcat):
        return _CONCAT_MAP[rule.pattern](a1, a2, b1, b2)

    if isinstance(rule, RuleSetOp):
        return _set_op(a1, a2, b1, b2, rule.name)

    if isinstance(rule, RulePosOp):
        return _pos_op(a1, a2, b1, b2, rule.name)

    if isinstance(rule, RuleUnaryOp):
        return _unary_op(a1, a2, b1, b2, rule.name)

    if isinstance(rule, RuleCharMap):
        keys = rule.key_fn(ex)
        result = ""
        for k in keys:
            if k not in rule.smap:
                return None
            result += rule.smap[k]
        return result

    if isinstance(rule, RuleBitOp):
        fn = _BIT_FN_MAP[rule.fn_name]
        pair_fn = _BIT_PAIR_MAP[rule.pair_name]
        pairs = pair_fn(ex)
        result = ""
        for a_ch, b_ch in pairs:
            r = fn(ord(a_ch), ord(b_ch))
            if not (0 <= r < 128):
                return None
            result += chr(r)
        return result

    if isinstance(rule, RuleTernaryBitOp):
        fn3 = _BIT3_FN_MAP[rule.fn_name]
        triple_fn = _BIT3_TRIPLE_MAP[rule.triple_name]
        triples = triple_fn(ex)
        result = ""
        for ac, bc, cc in triples:
            r = fn3(ord(ac), ord(bc), ord(cc)) & 0x7F
            if not (0 <= r < 128):
                return None
            result += chr(r)
        return result

    if isinstance(rule, RuleArith):
        if any(s not in rule.dmap for s in [a1, a2, b1, b2]):
            return None
        va, vb = _get_operands(a1, a2, b1, b2, rule.dmap, rule.rev_ops)
        r_str = _apply_result(rule.fn(va, vb), rule.rev_res)
        return _encode_result(r_str, rule.dmap)

    return None


def _check_all(rule: Rule, cases: list[_Ex]) -> bool:
    """Return True iff rule correctly predicts every example."""
    return all(_apply_rule(rule, ex) == ex.out for ex in cases)


# ══════════════════════════════════════════════════════════════════════════════
# Family 1: Concat
# ══════════════════════════════════════════════════════════════════════════════

def _try_family_concat(cases: list[_Ex]) -> Rule | None:
    if not _is_concat_eligible(cases):
        return None
    for name, fn in _CONCAT_PATTERNS:
        rule = RuleConcat(name)
        if _check_all(rule, cases):
            return rule
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Family 2: Set operations
# ══════════════════════════════════════════════════════════════════════════════

def _unique_ordered(s: str) -> str:
    seen: set[str] = set()
    out = ""
    for c in s:
        if c not in seen:
            seen.add(c)
            out += c
    return out


def _set_op(a1: str, a2: str, b1: str, b2: str, name: str) -> str:
    la, lb = [a1, a2], [b1, b2]
    sa, sb = set(la), set(lb)
    ops = {
        "union_ab":   _unique_ordered(a1 + a2 + b1 + b2),
        "union_ba":   _unique_ordered(b1 + b2 + a1 + a2),
        "inter_a":    "".join(c for c in la if c in sb),
        "inter_b":    "".join(c for c in lb if c in sa),
        "diff_ab":    "".join(c for c in la if c not in sb),
        "diff_ba":    "".join(c for c in lb if c not in sa),
        "sym_diff":   "".join(c for c in la if c not in sb) + "".join(c for c in lb if c not in sa),
        "common":     _unique_ordered("".join(c for c in a1+a2 if c in sb)),
    }
    return ops[name]


_SET_OP_NAMES = ["union_ab", "union_ba", "inter_a", "inter_b",
                  "diff_ab", "diff_ba", "sym_diff", "common"]


def _try_family_set_op(cases: list[_Ex]) -> Rule | None:
    for name in _SET_OP_NAMES:
        rule = RuleSetOp(name)
        if _check_all(rule, cases):
            return rule
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Family 3: Position ops (subsets of [A1,A2,B1,B2])
# ══════════════════════════════════════════════════════════════════════════════

def _pos_op(a1: str, a2: str, b1: str, b2: str, name: str) -> str:
    ch = [a1, a2, b1, b2]
    ops = {
        "even":   ch[0]+ch[2],
        "odd":    ch[1]+ch[3],
        "outer":  ch[0]+ch[3],
        "inner":  ch[1]+ch[2],
        "first1": ch[0],
        "last1":  ch[3],
        "first3": ch[0]+ch[1]+ch[2],
        "last3":  ch[1]+ch[2]+ch[3],
        "no_a1":  ch[1]+ch[2]+ch[3],
        "no_a2":  ch[0]+ch[2]+ch[3],
        "no_b1":  ch[0]+ch[1]+ch[3],
        "no_b2":  ch[0]+ch[1]+ch[2],
    }
    return ops[name]


_POS_OP_NAMES = list(_pos_op("A","B","C","D", n) and n  # just to verify keys exist
                      for n in ["even","odd","outer","inner","first1","last1",
                                "first3","last3","no_a1","no_a2","no_b1","no_b2"])


def _try_family_pos_op(cases: list[_Ex]) -> Rule | None:
    if not _is_concat_eligible(cases):
        return None
    for name in _POS_OP_NAMES:
        rule = RulePosOp(name)
        if _check_all(rule, cases):
            return rule
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Family 4: Unary ops
# ══════════════════════════════════════════════════════════════════════════════

def _unary_op(a1: str, a2: str, b1: str, b2: str, name: str) -> str:
    la, lb = a1+a2, b1+b2
    ops = {
        "rev_left":       la[::-1],
        "rev_right":      lb[::-1],
        "rev_left+right": la[::-1]+lb,
        "left+rev_right": la+lb[::-1],
        "rev_both":       (la+lb)[::-1],
        "sort_left":      "".join(sorted(la)),
        "sort_right":     "".join(sorted(lb)),
        "sort_both":      "".join(sorted(la+lb)),
        "sort_both_rev":  "".join(sorted(la+lb, reverse=True)),
        "dedup_left":     _unique_ordered(la),
        "dedup_right":    _unique_ordered(lb),
        "dedup_all":      _unique_ordered(la+lb),
    }
    return ops[name]


_UNARY_OP_NAMES = ["rev_left","rev_right","rev_left+right","left+rev_right",
                    "rev_both","sort_left","sort_right","sort_both","sort_both_rev",
                    "dedup_left","dedup_right","dedup_all"]


def _try_family_unary_op(cases: list[_Ex]) -> Rule | None:
    if not _is_concat_eligible(cases):
        return None
    for name in _UNARY_OP_NAMES:
        rule = RuleUnaryOp(name)
        if _check_all(rule, cases):
            return rule
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Family 5: Character map (learned sym→sym substitution)
# ══════════════════════════════════════════════════════════════════════════════

_CHAR_MAP_PATTERNS: list[tuple[str, Callable[[_Ex], list[str]]]] = [
    # 1-char
    ("map_a1",      lambda ex: [ex.a[0]]),
    ("map_a2",      lambda ex: [ex.a[1]]),
    ("map_b1",      lambda ex: [ex.b[0]]),
    ("map_b2",      lambda ex: [ex.b[1]]),
    # 2-char
    ("map_left",    lambda ex: [ex.a[0], ex.a[1]]),
    ("map_right",   lambda ex: [ex.b[0], ex.b[1]]),
    ("map_a1b1",    lambda ex: [ex.a[0], ex.b[0]]),
    ("map_a2b2",    lambda ex: [ex.a[1], ex.b[1]]),
    ("map_a1b2",    lambda ex: [ex.a[0], ex.b[1]]),
    ("map_a2b1",    lambda ex: [ex.a[1], ex.b[0]]),
    ("map_b1a2",    lambda ex: [ex.b[0], ex.a[1]]),
    ("map_b2a1",    lambda ex: [ex.b[1], ex.a[0]]),
    # 3-char (all 4-choose-3 orderings of inputs)
    ("map_a1a2b1",  lambda ex: [ex.a[0], ex.a[1], ex.b[0]]),
    ("map_a1a2b2",  lambda ex: [ex.a[0], ex.a[1], ex.b[1]]),
    ("map_a1b1b2",  lambda ex: [ex.a[0], ex.b[0], ex.b[1]]),
    ("map_a2b1b2",  lambda ex: [ex.a[1], ex.b[0], ex.b[1]]),
    ("map_b1b2a1",  lambda ex: [ex.b[0], ex.b[1], ex.a[0]]),
    ("map_b1b2a2",  lambda ex: [ex.b[0], ex.b[1], ex.a[1]]),
    ("map_a1b1a2",  lambda ex: [ex.a[0], ex.b[0], ex.a[1]]),
    ("map_a1b2a2",  lambda ex: [ex.a[0], ex.b[1], ex.a[1]]),
    ("map_b1a1b2",  lambda ex: [ex.b[0], ex.a[0], ex.b[1]]),
    ("map_b1a2b2",  lambda ex: [ex.b[0], ex.a[1], ex.b[1]]),
    ("map_a2a1b1",  lambda ex: [ex.a[1], ex.a[0], ex.b[0]]),
    ("map_a2a1b2",  lambda ex: [ex.a[1], ex.a[0], ex.b[1]]),
    # 4-char
    ("map_all",     lambda ex: [ex.a[0], ex.a[1], ex.b[0], ex.b[1]]),
    ("map_all_rev", lambda ex: [ex.b[0], ex.b[1], ex.a[0], ex.a[1]]),
    ("map_zip",     lambda ex: [ex.a[0], ex.b[0], ex.a[1], ex.b[1]]),
    ("map_zip_rev", lambda ex: [ex.b[0], ex.a[0], ex.b[1], ex.a[1]]),
]


def _learn_char_map(cases: list[_Ex], key_fn: Callable[[_Ex], list[str]]) -> dict[str, str] | None:
    """Learn sym→sym mapping from examples; return None on any conflict."""
    smap: dict[str, str] = {}
    for ex in cases:
        keys = key_fn(ex)
        if len(keys) != len(ex.out):
            return None                       # wrong output length → fail immediately
        for k, v in zip(keys, list(ex.out)):
            if k in smap:
                if smap[k] != v:
                    return None               # conflict → fail immediately
            else:
                smap[k] = v
    return smap or None


def _try_family_char_map(cases: list[_Ex]) -> Rule | None:
    for name, key_fn in _CHAR_MAP_PATTERNS:
        smap = _learn_char_map(cases, key_fn)
        if smap is not None:
            rule = RuleCharMap(name, smap, key_fn)
            if _check_all(rule, cases):       # extra safety check
                return rule
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Family 6: Bit ops (XOR / AND / OR on ASCII codes)
# ══════════════════════════════════════════════════════════════════════════════

_BIT_FN_MAP: dict[str, Callable[[int, int], int]] = {
    # binary operations
    "xor":     lambda a, b: a ^ b,
    "and":     lambda a, b: a & b,
    "or":      lambda a, b: a | b,
    "xnor":    lambda a, b: (~(a ^ b)) & 0x7F,
    "add_mod": lambda a, b: (a + b) % 128,
    "sub_mod": lambda a, b: (a - b) % 128,
    "max_chr": lambda a, b: max(a, b),
    "min_chr": lambda a, b: min(a, b),
    # unary operations (b is ignored; use self-pair functions)
    "not7":    lambda a, b: (~a) & 0x7F,
    "lshift1": lambda a, b: (a << 1) & 0x7F,
    "rshift1": lambda a, b: a >> 1,
    "rotl7":   lambda a, b: ((a << 1) | (a >> 6)) & 0x7F,
    "rotr7":   lambda a, b: ((a >> 1) | (a << 6)) & 0x7F,
    "lshift2": lambda a, b: (a << 2) & 0x7F,
    "rshift2": lambda a, b: a >> 2,
}

_BIT_PAIR_MAP: dict[str, Callable[[_Ex], list[tuple[str, str]]]] = {
    "a1_b1":     lambda ex: [(ex.a[0], ex.b[0])],
    "a2_b2":     lambda ex: [(ex.a[1], ex.b[1])],
    "a1b1_a2b2": lambda ex: [(ex.a[0], ex.b[0]), (ex.a[1], ex.b[1])],
    "a1_b2":     lambda ex: [(ex.a[0], ex.b[1])],
    "a2_b1":     lambda ex: [(ex.a[1], ex.b[0])],
    "a1a2_b1b2": lambda ex: list(zip(ex.a[0]+ex.a[1], ex.b[0]+ex.b[1])),
    # reversed-direction pairs (matter for sub_mod)
    "b1_a1":     lambda ex: [(ex.b[0], ex.a[0])],
    "b2_a2":     lambda ex: [(ex.b[1], ex.a[1])],
    "b1_a2":     lambda ex: [(ex.b[0], ex.a[1])],
    "b2_a1":     lambda ex: [(ex.b[1], ex.a[0])],
    # same-side pairs (combine chars from one side)
    "a1_a2":     lambda ex: [(ex.a[0], ex.a[1])],
    "b1_b2":     lambda ex: [(ex.b[0], ex.b[1])],
    "a2_a1":     lambda ex: [(ex.a[1], ex.a[0])],
    "b2_b1":     lambda ex: [(ex.b[1], ex.b[0])],
    # cross pairs producing 2-char output
    "a1b2_a2b1": lambda ex: [(ex.a[0], ex.b[1]), (ex.a[1], ex.b[0])],
    "b1a2_b2a1": lambda ex: [(ex.b[0], ex.a[1]), (ex.b[1], ex.a[0])],
    # self-pairs (for unary ops — b argument is same char, ignored by unary fns)
    "a1_a1":     lambda ex: [(ex.a[0], ex.a[0])],
    "a2_a2":     lambda ex: [(ex.a[1], ex.a[1])],
    "b1_b1":     lambda ex: [(ex.b[0], ex.b[0])],
    "b2_b2":     lambda ex: [(ex.b[1], ex.b[1])],
    # self-pairs producing 2-char output
    "a1a1_a2a2": lambda ex: [(ex.a[0], ex.a[0]), (ex.a[1], ex.a[1])],
    "b1b1_b2b2": lambda ex: [(ex.b[0], ex.b[0]), (ex.b[1], ex.b[1])],
    "a1a1_b1b1": lambda ex: [(ex.a[0], ex.a[0]), (ex.b[0], ex.b[0])],
    "a2a2_b2b2": lambda ex: [(ex.a[1], ex.a[1]), (ex.b[1], ex.b[1])],
}

# ── Ternary bit operations (majority, choice) ─────────────────────────────────

_BIT3_FN_MAP: dict[str, Callable[[int, int, int], int]] = {
    "maj":  lambda a, b, c: (a & b) | (b & c) | (a & c),
    "ch":   lambda a, b, c: (a & b) | ((~a) & c),
    "xor3": lambda a, b, c: a ^ b ^ c,
    "med":  lambda a, b, c: sorted([a, b, c])[1],  # median
}

_BIT3_TRIPLE_MAP: dict[str, Callable[[_Ex], list[tuple[str, str, str]]]] = {
    "a1_a2_b1": lambda ex: [(ex.a[0], ex.a[1], ex.b[0])],
    "a1_a2_b2": lambda ex: [(ex.a[0], ex.a[1], ex.b[1])],
    "a1_b1_b2": lambda ex: [(ex.a[0], ex.b[0], ex.b[1])],
    "a2_b1_b2": lambda ex: [(ex.a[1], ex.b[0], ex.b[1])],
    "b1_b2_a1": lambda ex: [(ex.b[0], ex.b[1], ex.a[0])],
    "b1_b2_a2": lambda ex: [(ex.b[0], ex.b[1], ex.a[1])],
    "a1_b1_a2": lambda ex: [(ex.a[0], ex.b[0], ex.a[1])],
    "a2_b2_b1": lambda ex: [(ex.a[1], ex.b[1], ex.b[0])],
}


def _try_family_bit_op(cases: list[_Ex]) -> Rule | None:
    for pair_name in _BIT_PAIR_MAP:
        for fn_name in _BIT_FN_MAP:
            rule = RuleBitOp(pair_name, fn_name)
            if _check_all(rule, cases):
                return rule
    return None


def _try_family_ternary_bit_op(cases: list[_Ex]) -> Rule | None:
    for triple_name in _BIT3_TRIPLE_MAP:
        for fn_name in _BIT3_FN_MAP:
            rule = RuleTernaryBitOp(triple_name, fn_name)
            if _check_all(rule, cases):
                return rule
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Family 7: Arithmetic (symbol→digit mapping + math operation)
# ══════════════════════════════════════════════════════════════════════════════

_MAX_ARITH = 500        # per-op candidate cap


def _arith_seed(cases: list[_Ex], op_name: str, rev_ops: bool, rev_res: bool) -> list[dict[str, int]]:
    """Generate candidate dmaps consistent with the FIRST example."""
    if not cases:
        return []
    fn = _ARITH_OP_MAP[op_name]
    ex0 = cases[0]
    a1, a2, b1, b2 = ex0.a[0], ex0.a[1], ex0.b[0], ex0.b[1]
    out: list[dict[str, int]] = []
    for va in range(100):
        for vb in range(100):
            dm: dict[str, int] = {}
            ok = True
            for sym, d in zip([a1, a2, b1, b2], [va//10, va%10, vb//10, vb%10]):
                if sym in dm and dm[sym] != d:
                    ok = False; break
                dm[sym] = d
            if not ok:
                continue
            va_u, vb_u = _get_operands(a1, a2, b1, b2, dm, rev_ops)
            r = _apply_result(fn(va_u, vb_u), rev_res)
            ext = _match_result(r, ex0.out, dm)
            if ext is not None:
                out.append(ext)
    return out


def _arith_filter(
    candidates: list[dict[str, int]], ex: _Ex,
    op_name: str, rev_ops: bool, rev_res: bool,
) -> list[dict[str, int]]:
    """Keep only candidates consistent with this example; extend mapping as needed."""
    fn = _ARITH_OP_MAP[op_name]
    a1, a2, b1, b2 = ex.a[0], ex.a[1], ex.b[0], ex.b[1]
    surviving: list[dict[str, int]] = []
    for dm in candidates[:_MAX_ARITH]:
        unmapped = [s for s in [a1, a2, b1, b2] if s not in dm]
        if len(unmapped) > 3:
            continue
        if unmapped:
            for new_ds in iproduct(range(10), repeat=len(unmapped)):
                ext = dict(dm)
                for s, d in zip(unmapped, new_ds):
                    ext[s] = d
                va_u, vb_u = _get_operands(a1, a2, b1, b2, ext, rev_ops)
                r = _apply_result(fn(va_u, vb_u), rev_res)
                ext2 = _match_result(r, ex.out, ext)
                if ext2 is not None:
                    surviving.append(ext2)
        else:
            va_u, vb_u = _get_operands(a1, a2, b1, b2, dm, rev_ops)
            r = _apply_result(fn(va_u, vb_u), rev_res)
            ext2 = _match_result(r, ex.out, dm)
            if ext2 is not None:
                surviving.append(ext2)
    return surviving



def _try_family_arith(
    cases: list[_Ex],
    cache: dict | None = None,
) -> Rule | None:
    for op_name, _ in _ARITH_OPS:
        for rev_ops in (False, True):
            for rev_res in (False, True):
                key = (op_name, rev_ops, rev_res)
                if cache is not None and key in cache:
                    cands = cache[key]
                else:
                    cands = _arith_seed(cases, op_name, rev_ops, rev_res)
                    for ex in cases[1:]:
                        cands = _arith_filter(cands, ex, op_name, rev_ops, rev_res)
                        if not cands:
                            break
                    if cache is not None:
                        cache[key] = cands[:_MAX_ARITH]
                if cands:
                    return RuleArith(op_name, rev_ops, rev_res, cands[0])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CASCADE — try all families in order for one operator's examples
# ══════════════════════════════════════════════════════════════════════════════

_CASCADE: list[Callable[[list[_Ex]], Rule | None]] = [
    _try_family_concat,
    _try_family_set_op,
    _try_family_pos_op,
    _try_family_unary_op,
    _try_family_char_map,
    _try_family_bit_op,
    _try_family_ternary_bit_op,
    _try_family_arith,
]


def detect_rule(cases: list[_Ex]) -> Rule | None:
    """
    Try each transformation family in order.
    Within each family every variant is tested against ALL examples.
    Returns the first rule that satisfies every example, or None.
    """
    for family_fn in _CASCADE:
        rule = family_fn(cases)
        if rule is not None:
            return rule
    return None


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONSISTENCY — joint solve when per-op arith mappings conflict
# ══════════════════════════════════════════════════════════════════════════════

def _merge(a: dict[str, int], b: dict[str, int]) -> dict[str, int] | None:
    m = dict(a)
    for k, v in b.items():
        if k in m and m[k] != v:
            return None
        m[k] = v
    return m


def _dmaps_consistent(rules: dict[str, Rule]) -> bool:
    gm: dict[str, int] = {}
    for rule in rules.values():
        if not isinstance(rule, RuleArith):
            continue
        for sym, d in rule.dmap.items():
            if sym in gm and gm[sym] != d:
                return False
            gm[sym] = d
    return True


def _all_arith_variants(
    cases: list[_Ex],
    cache: dict | None = None,
) -> list[tuple[str, bool, bool, list[dict[str, int]]]]:
    """Return (op_name, rev_ops, rev_res, candidates) for every variant with ≥1 candidate.
    Reuses cached candidates from a prior cascade run when available."""
    result = []
    for op_name, _ in _ARITH_OPS:
        for rev_ops in (False, True):
            for rev_res in (False, True):
                key = (op_name, rev_ops, rev_res)
                if cache is not None and key in cache:
                    cands = cache[key]
                else:
                    cands = _arith_seed(cases, op_name, rev_ops, rev_res)
                    for ex in cases[1:]:
                        cands = _arith_filter(cands, ex, op_name, rev_ops, rev_res)
                        if not cands:
                            break
                    cands = cands[:_MAX_ARITH]
                    if cache is not None:
                        cache[key] = cands
                if cands:
                    result.append((op_name, rev_ops, rev_res, cands))
    return result


def _global_arith_solve(
    arith_by_op: dict[str, list[_Ex]],
    caches: dict[str, dict] | None = None,
) -> dict[str, RuleArith] | None:
    """
    Find a single globally-consistent symbol→digit mapping across all operators.

    Strategy: enumerate locally-valid variant combinations, then for each
    combination seed from the first example of the first operator and filter
    through ALL examples from ALL operators in sequence.  This avoids the
    per-op merge bottleneck and naturally produces a global dmap that covers
    every symbol seen across all operators.
    """
    ops = list(arith_by_op.keys())
    if not ops:
        return None

    # Pre-compute locally-valid variants per op (reuses cascade cache).
    # If any op has zero valid variants, no arithmetic solution exists.
    op_variants: dict[str, list[tuple[str, bool, bool, list[dict[str, int]]]]] = {}
    for op in ops:
        cache = caches.get(op, {}) if caches else {}
        variants = _all_arith_variants(arith_by_op[op], cache=cache)
        if not variants:
            return None
        op_variants[op] = variants

    # Flatten all (op, example) pairs; first entry is the global seed.
    ordered: list[tuple[str, _Ex]] = [
        (op, ex) for op in ops for ex in arith_by_op[op]
    ]
    seed_op = ordered[0][0]
    seed_ex = ordered[0][1]
    rest = ordered[1:]

    # Enumerate all combinations of locally-valid variants across operators.
    for combo in iproduct(*[op_variants[op] for op in ops]):
        rule_for_op: dict[str, tuple[str, bool, bool]] = {
            ops[i]: (combo[i][0], combo[i][1], combo[i][2])
            for i in range(len(ops))
        }
        seed_name, seed_rv, seed_rr = rule_for_op[seed_op]

        # Seed from the very first example only (all 100×100 candidates).
        cands = _arith_seed([seed_ex], seed_name, seed_rv, seed_rr)
        if not cands:
            continue

        # Filter through every remaining example using that example's operator rule.
        ok = True
        for op, ex in rest:
            name, rv, rr = rule_for_op[op]
            cands = _arith_filter(cands, ex, name, rv, rr)
            if not cands:
                ok = False
                break

        if ok and cands:
            # Found a globally-consistent mapping.
            dmap = cands[0]
            return {
                op: RuleArith(rule_for_op[op][0], rule_for_op[op][1], rule_for_op[op][2], dmap)
                for op in ops
            }

    return None


def _joint_arith_solve(
    arith_by_op: dict[str, list[_Ex]],
    caches: dict[str, dict] | None = None,
) -> dict[str, RuleArith] | None:
    """
    Find arithmetic rules for each operator such that a SINGLE global dmap
    is consistent across all of them.

    Strategy: pick one operator, seed global candidates from it, then for each
    remaining operator try all its arithmetic variants and intersect the dmaps.
    Uses _MAX_ARITH (not _MAX_JOINT) so all candidates are considered.
    """
    ops = list(arith_by_op.keys())
    if not ops:
        return None

    # Pre-compute all viable (variant, candidates) per op (reuses cascade cache)
    op_variants: dict[str, list[tuple[str, bool, bool, list[dict[str, int]]]]] = {}
    for op in ops:
        cache = caches.get(op, {}) if caches else {}
        variants = _all_arith_variants(arith_by_op[op], cache=cache)
        if not variants:
            return None
        op_variants[op] = variants

    # Backtrack: pick one variant per op, merge globally
    def _bt(
        i: int,
        global_cands: list[dict[str, int]],
        chosen: dict[str, tuple[str, bool, bool, dict[str, int]]],
    ) -> dict[str, tuple[str, bool, bool, dict[str, int]]] | None:
        if i == len(ops):
            return chosen if global_cands else None
        op = ops[i]
        for op_name, rev_ops, rev_res, cands in op_variants[op]:
            merged: list[dict[str, int]] = []
            for base in global_cands:
                for ext in cands:
                    m = _merge(base, ext)
                    if m is not None:
                        merged.append(m)
                        if len(merged) >= _MAX_ARITH:
                            break
                if len(merged) >= _MAX_ARITH:
                    break
            if not merged:
                if not global_cands:
                    merged = list(cands[:_MAX_ARITH])
                else:
                    continue
            r = _bt(i + 1, merged, {**chosen, op: (op_name, rev_ops, rev_res, merged[0])})
            if r is not None:
                return r
        return None

    first_op = ops[0]
    for op_name, rev_ops, rev_res, cands in op_variants[first_op]:
        r = _bt(1, list(cands[:_MAX_ARITH]), {first_op: (op_name, rev_ops, rev_res, cands[0])})
        if r is not None:
            return {
                op: RuleArith(v[0], v[1], v[2], v[3])
                for op, v in r.items()
            }
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def solve_global(
    by_op: dict[str, list[_Ex]],
    q_op: str | None = None,
    q_a: tuple[str, str] | None = None,
    q_b: tuple[str, str] | None = None,
) -> dict[str, Rule] | None:
    """
    Detect the rule for each operator. Returns {op_char: Rule} or None.

    1. Run cascade detection per operator, caching arithmetic variant results.
    2. If arithmetic operators have conflicting digit mappings, re-solve them
       jointly (reusing cached variants — no duplicate seeding).

    When q_op/q_a/q_b are given, prefer rules for q_op that can answer the question.
    Non-arith rules for q_op that can't answer are discarded so arith gets a chance.
    """
    rules: dict[str, Rule] = {}
    # arith_caches[op] = {(op_name, rev_ops, rev_res): [dmap, ...]}
    arith_caches: dict[str, dict] = {}
    _q_ex = _Ex(a=q_a, op=q_op, b=q_b, out="") if (q_op and q_a and q_b) else None

    for op, cases in by_op.items():
        # Cheap non-arith families first
        rule = (
            _try_family_concat(cases)
            or _try_family_set_op(cases)
            or _try_family_pos_op(cases)
            or _try_family_unary_op(cases)
            or _try_family_char_map(cases)
            or _try_family_bit_op(cases)
            or _try_family_ternary_bit_op(cases)
        )
        # For the question operator, skip non-arith rules that can't answer
        if rule is not None and op == q_op and _q_ex is not None:
            if _apply_rule(rule, _q_ex) is None:
                rule = None  # fall through to arith
        if rule is not None:
            rules[op] = rule
        else:
            # Pre-compute ALL arith variants into cache (joint solver reuses this)
            cache: dict = {}
            arith_caches[op] = cache
            _all_arith_variants(cases, cache=cache)   # populates cache, result discarded
            if op == q_op and _q_ex is not None:
                # Prefer variant that can answer the question; use voting to pick best
                _fallback_arith: RuleArith | None = None
                # answer → (natural_votes, total_votes, first_candidate)
                _votes: dict[str, tuple[int, int, RuleArith]] = {}
                for (op_name, rev_ops, rev_res), cands in cache.items():
                    if not cands:
                        continue
                    candidate = RuleArith(op_name, rev_ops, rev_res, cands[0])
                    if _fallback_arith is None:
                        _fallback_arith = candidate
                    ans = _apply_rule(candidate, _q_ex)
                    if ans is not None:
                        nat = 0 if rev_res else 1
                        entry = _votes.get(ans)
                        if entry is None:
                            _votes[ans] = (nat, 1, candidate)
                        else:
                            _votes[ans] = (entry[0] + nat, entry[1] + 1, entry[2])
                if _votes:
                    best = max(_votes, key=lambda a: (_votes[a][0], _votes[a][1]))
                    rules[op] = _votes[best][2]
                elif _fallback_arith is not None:
                    rules[op] = _fallback_arith
            else:
                # Pick first match from cache
                for (op_name, rev_ops, rev_res), cands in cache.items():
                    if cands:
                        rules[op] = RuleArith(op_name, rev_ops, rev_res, cands[0])
                        break

    # Joint arithmetic re-solve if any arith rules conflict or an op is unsolved.
    # Try global seed-filter first (finds full global dmap covering all symbols),
    # then fall back to the merge-based joint solver.
    needs_joint = not _dmaps_consistent(rules) or any(op not in rules for op in by_op)
    if needs_joint:
        arith_ops_cases = {
            op: by_op[op]
            for op in by_op
            if op not in rules or isinstance(rules.get(op), RuleArith)
        }
        if arith_ops_cases:
            shared_caches = {op: arith_caches.get(op, {}) for op in arith_ops_cases}
            result = _global_arith_solve(arith_ops_cases, caches=shared_caches)
            if result is None:
                result = _joint_arith_solve(arith_ops_cases, caches=shared_caches)
            if result is not None:
                rules.update(result)

    return rules if rules else None


def apply_to_question(rule: Rule, q_a: tuple[str, str], q_b: tuple[str, str], q_op: str) -> str | None:
    """Apply a detected rule to the question input."""
    return _apply_rule(rule, _Ex(a=q_a, op=q_op, b=q_b, out=""))
