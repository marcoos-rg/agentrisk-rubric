"""Per-item answer probabilities from token logprobs (the "soft" f(T)).

With structured JSON output, each item is written as {"step": …, "quote": "…", "answer": "YES"}.
For every binary/graded item we locate the token where its "answer" value starts and read the
top-k alternatives the model considered there → P(YES), P(NO), P(UNKNOWN), P(NA) or P(0..3).

Measured on Qwen3-32B (2026-11-25/26): the top-5 alternatives cover ≈ 100 % of the mass, and two
identical runs differ by a mean total-variation distance of ≈ 0.002 — one call per trace is enough.
"""
from __future__ import annotations

import math
import re

from .rubric import BINARY, GRADED, Rubric


def item_probs(tokens: list[str], tops: list[list[tuple[str, float]]], rb: Rubric) -> dict:
    """tokens: generated tokens; tops: for each token, its top-k (token, logprob) alternatives.

    Returns {item_id: {"answer": str, "mass": float, "probs": {option: p}}} (probs renormalised
    over the options; "mass" = how much of the top-k fell on valid options), or {"error": …}.
    """
    text = "".join(tokens)
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t)
    out = {}
    for it in rb.items:
        if it["kind"] == "multi":
            continue
        options = BINARY if it["kind"] == "binary" else GRADED
        m = re.search(r'"%s"\s*:\s*\{.*?"answer"\s*:\s*"' % re.escape(it["id"]), text, re.S)
        if not m:
            out[it["id"]] = {"error": "answer not found"}
            continue
        vpos = m.end()                                   # first character of the answer value
        k = max(i for i, s in enumerate(starts) if s <= vpos)
        prefix = text[starts[k]:vpos]                    # part of the token before the value (e.g. '"')
        probs = {o: 0.0 for o in options}
        for alt, lp in tops[k]:
            if not alt.startswith(prefix):
                continue
            rest = alt[len(prefix):]
            if not rest:
                continue
            match = [o for o in options if o.startswith(rest) or rest.startswith(o)]
            for o in match:                              # ambiguous prefixes ("N" → NO/NA) are split
                probs[o] += math.exp(lp) / len(match)
        mass = sum(probs.values())
        end = text.find('"', vpos)
        chosen = text[vpos:end] if end != -1 else text[vpos:]
        out[it["id"]] = {"answer": chosen, "mass": round(mass, 4),
                         "probs": {o: round(p / mass, 4) for o, p in probs.items() if p} if mass else {}}
    return out


def doubtful(probs: dict, lo: float = 0.1, hi: float = 0.9) -> list[str]:
    """Items whose chosen answer had probability strictly between lo and hi."""
    out = []
    for iid, v in probs.items():
        p = v.get("probs") or {}
        if p and lo < p.get(v.get("answer"), 0.0) < hi:
            out.append(iid)
    return out
