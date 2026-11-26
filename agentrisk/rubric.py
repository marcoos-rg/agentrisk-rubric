"""Load and validate a rubric file (rubrics/<name>.py).

A rubric file is a plain Python module with three module-level names:

  VERSION : str             e.g. "v1-factors-draft"; stored in every result file
  BLOCKS  : dict[str, str]  block id → heading shown in the prompt, in prompt order
  RUBRIC  : list[dict]      the items, in prompt order (see rubrics/TEMPLATE.py)

Everything downstream (prompt, JSON schema, validation, features, logprob extraction) is
generated from RUBRIC, so a rubric file is the ONLY thing you edit to change the questionnaire.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path

from .taxonomy import DOMAINS, DRIVERS, FACTORS, GATE, OFFICIAL

REPO = Path(__file__).resolve().parent.parent
RUBRICS_DIR = REPO / "rubrics"

BINARY = ["YES", "NO", "UNKNOWN", "NA"]
GRADED = ["0", "1", "2", "3", "UNKNOWN", "NA"]
KINDS = ("binary", "graded", "multi")

_COMMON = {"id", "kind", "block", "question", "official", "note"}
_ALLOWED = {"binary": _COMMON | {"yes", "no"},
            "graded": _COMMON | {"levels", "tie_break"},
            "multi": _COMMON | {"options"}}
_ID = re.compile(r"^[A-Z][A-Z0-9]*_[a-z0-9_]+$")          # e.g. "O2_user_confirmed"
_OPTION = re.compile(r"^[a-z][a-z0-9_]*$")


class RubricError(ValueError):
    """The rubric file breaks the contract; the message lists every problem found."""


@dataclass
class Rubric:
    name: str                      # file stem, e.g. "v0_alpha"
    version: str
    blocks: dict[str, str]
    items: list[dict]
    path: Path
    sha256: str                    # hash of the file: results record it so runs stay comparable
    warnings: list[str] = field(default_factory=list)

    @property
    def gate(self) -> dict | None:
        first = self.items[0]
        return first if first["block"] == GATE else None

    def item(self, item_id: str) -> dict:
        return next(i for i in self.items if i["id"] == item_id)


def available() -> list[str]:
    return sorted(p.stem for p in RUBRICS_DIR.glob("*.py") if not p.stem.startswith("_"))


def load(name_or_path: str) -> Rubric:
    """Load rubrics/<name>.py (or a path to any .py file) and validate it. Raises RubricError."""
    p = Path(name_or_path)
    if p.suffix != ".py":
        p = RUBRICS_DIR / f"{name_or_path}.py"
    if not p.exists():
        raise RubricError(f"rubric not found: {p} (available: {', '.join(available())})")
    spec = importlib.util.spec_from_file_location(f"rubric_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    missing = [k for k in ("VERSION", "BLOCKS", "RUBRIC") if not hasattr(mod, k)]
    if missing:
        raise RubricError(f"{p.name}: missing module-level name(s): {', '.join(missing)}")
    rb = Rubric(name=p.stem, version=mod.VERSION, blocks=dict(mod.BLOCKS), items=list(mod.RUBRIC),
                # Line endings are normalised so Windows (CRLF) and Linux/Kaggle (LF) checkouts agree.
                path=p, sha256=hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12])
    errors, rb.warnings = validate_structure(rb)
    if errors:
        raise RubricError(f"{p.name}: {len(errors)} problem(s)\n  - " + "\n  - ".join(errors))
    return rb


def validate_structure(rb: Rubric) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors make the rubric unusable; warnings are style advice."""
    err, warn = [], []
    if not isinstance(rb.version, str) or not rb.version:
        err.append("VERSION must be a non-empty string")
    if not rb.blocks:
        err.append("BLOCKS must map at least one block id to a heading")
    if not rb.items:
        err.append("RUBRIC is empty")
        return err, warn

    seen, block_order = set(), []
    for n, it in enumerate(rb.items):
        where = f"item #{n} ({it.get('id', '?') if isinstance(it, dict) else '?'})"
        if not isinstance(it, dict):
            err.append(f"{where}: must be a dict")
            continue
        iid, kind = it.get("id"), it.get("kind")
        if not isinstance(iid, str) or not _ID.match(iid):
            err.append(f"{where}: id must look like 'O2_user_confirmed' (PREFIX_snake_case)")
        elif iid in seen:
            err.append(f"{where}: duplicate id")
        seen.add(iid)
        if kind not in KINDS:
            err.append(f"{where}: kind must be one of {KINDS}")
            continue
        unknown = set(it) - _ALLOWED[kind]
        if unknown:
            err.append(f"{where}: unknown key(s) {sorted(unknown)} for kind '{kind}' (typo?)")
        if it.get("block") not in rb.blocks:
            err.append(f"{where}: block '{it.get('block')}' is not declared in BLOCKS")
        elif not block_order or block_order[-1] != it["block"]:
            if it["block"] in block_order:
                err.append(f"{where}: items of block '{it['block']}' must be contiguous")
            block_order.append(it["block"])
        q = it.get("question")
        if not isinstance(q, str) or not q.strip():
            err.append(f"{where}: question is empty")
        elif not q.rstrip().endswith("?"):
            warn.append(f"{where}: question does not end with '?'")
        off = it.get("official")
        if not isinstance(off, list) or not off:
            err.append(f"{where}: official must be a non-empty list of taxonomy names")
        else:
            bad = [o for o in off if o not in OFFICIAL]
            if bad:
                err.append(f"{where}: official {bad} not in agentrisk/taxonomy.py")

        if kind == "binary":
            for k in ("yes", "no"):
                if not isinstance(it.get(k), str) or not it[k].strip():
                    err.append(f"{where}: binary items need a non-empty '{k}' description")
        elif kind == "graded":
            lv = it.get("levels")
            if (not isinstance(lv, list) or len(lv) != 4
                    or any(not str(l).startswith(f"{d} =") for d, l in enumerate(lv))):
                err.append(f"{where}: graded items need levels = ['0 = …', '1 = …', '2 = …', '3 = …']")
            warn.append(f"{where}: graded item — prefer several binary YES/NO items (see TEMPLATE.py)")
        else:
            opts = it.get("options")
            if (not isinstance(opts, list) or not opts or len(set(opts)) != len(opts)
                    or any(not isinstance(o, str) or not _OPTION.match(o) for o in opts)):
                err.append(f"{where}: multi items need unique snake_case options")
            warn.append(f"{where}: multi-select has no per-option logprobs and was the least "
                        "stable output across runs — prefer one binary item per option")

    gates = [it for it in rb.items if isinstance(it, dict) and it.get("block") == GATE]
    if gates:
        if rb.items[0].get("block") != GATE or len(gates) != 1:
            err.append("the gate block must contain exactly one item and come first")
        elif rb.items[0].get("kind") != "binary":
            err.append("the gate item must be binary (YES = the action has an external effect)")
    if len(rb.items) > 45:
        warn.append(f"{len(rb.items)} items: output tokens (and Kaggle run time) grow with every item")
    return err, warn


def coverage(rb: Rubric) -> dict[str, dict[str, list[str]]]:
    """Which items observe each official factor / driver / domain (for the 2–5 items/factor goal)."""
    out = {"factors": {f: [] for f in FACTORS}, "drivers": {d: [] for d in DRIVERS},
           "domains": {d: [] for d in DOMAINS}}
    for it in rb.items:
        for o in it["official"]:
            for group in out.values():
                if o in group:
                    group[o].append(it["id"])
    return out
