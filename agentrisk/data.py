"""Download and load the AgentRisk splits (public on Hugging Face).

  python -m agentrisk.data              # downloads train, dev and the split manifest into data/

Fields of each line: example_id, formatted_interaction, proposed_action, risk_severity (1–5),
risk_domains (list), risk_drivers (list). The data is NOT committed to git (see .gitignore).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from .rubric import REPO

DATA = REPO / "data"
BASE = "https://huggingface.co/datasets/AgentRisk/agentrisk-data-2027/resolve/main"
FILES = {"train.jsonl": "data/train.jsonl", "dev.jsonl": "data/dev.jsonl",
         "split_manifest.jsonl": "metadata/split_manifest.jsonl"}


def download(force: bool = False) -> None:
    DATA.mkdir(exist_ok=True)
    for name, remote in FILES.items():
        dst = DATA / name
        if dst.exists() and not force:
            print(f"  ✓ {name} (already there)")
            continue
        try:
            urllib.request.urlretrieve(f"{BASE}/{remote}", dst)
            print(f"  ↓ {name}: {sum(1 for _ in dst.open(encoding='utf-8'))} lines")
        except Exception as e:  # the manifest is optional; the splits are not
            print(f"  ✗ {name}: {e}")
            if name != "split_manifest.jsonl":
                raise


def load(split: str) -> list[dict]:
    path = DATA / f"{split}.jsonl"
    if not path.exists():
        sys.exit(f"Missing {path}. Run: python -m agentrisk.data")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def pick(rows: list[dict], ids: list[str] | None = None, n: int | None = None) -> list[dict]:
    """Explicit ids, or the first n traces stratified by gold severity (1, 3, 4, 2, 5 in turn),
    so a small run already covers every level. n=None → all rows in file order."""
    if ids:
        by_id = {r["example_id"]: r for r in rows}
        missing = [i for i in ids if i not in by_id]
        if missing:
            sys.exit(f"Unknown ids: {missing}")
        return [by_id[i] for i in ids]
    if n is None or n >= len(rows):
        return list(rows)
    order, out = [1, 3, 4, 2, 5], []
    pools = {s: [r for r in rows if r["risk_severity"] == s] for s in order}
    while len(out) < n and any(pools.values()):
        for s in order:
            if pools[s] and len(out) < n:
                out.append(pools[s].pop(0))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    download(force="--force" in sys.argv)
