"""Validate a rubric file and preview what the model will see. No API calls, no GPU.

  python -m agentrisk.check_rubric                      # every rubric in rubrics/
  python -m agentrisk.check_rubric v0_alpha --prompt    # one rubric + full system prompt
  python -m agentrisk.check_rubric my_rubric --schema   # + JSON schema

Exit code 1 if any rubric is invalid, so it can run as a pre-commit check and in the notebook
BEFORE the 10–15 minute model load.
"""
from __future__ import annotations

import argparse
import json
import sys

from .prompt import feature_names, json_schema, system_prompt, user_prompt
from .rubric import RubricError, available, coverage, load


def report(name: str, show_prompt: bool, show_schema: bool) -> bool:
    try:
        rb = load(name)
    except RubricError as e:
        print(f"✗ {e}\n")
        return False
    sp = system_prompt(rb)
    kinds = {k: sum(i["kind"] == k for i in rb.items) for k in ("binary", "graded", "multi")}
    print(f"✓ {rb.name} · version {rb.version} · sha {rb.sha256} · {len(rb.items)} items {kinds} · "
          f"{len(feature_names(rb))} features · system prompt ≈ {len(sp) // 4} tokens")
    for b, title in rb.blocks.items():
        ids = [i["id"] for i in rb.items if i["block"] == b]
        print(f"  [{b}] {title}: {len(ids)} items")
    cov = coverage(rb)
    print("  coverage of the 7 risk factors (goal: 2–5 items each):")
    for f, ids in cov["factors"].items():
        flag = "  " if 2 <= len(ids) <= 5 else "⚠ "
        print(f"    {flag}{f:26s} {len(ids)}  {', '.join(ids)}")
    print("  coverage of the risk drivers (Subtask 2b):")
    for d, ids in cov["drivers"].items():
        print(f"    {'  ' if ids else '⚠ '}{d:38s} {len(ids)}  {', '.join(ids)}")
    dom = [d for d, ids in cov["domains"].items() if ids]
    print(f"  harm domains observed (Subtask 2a): {len(dom)}/{len(cov['domains'])}")
    for w in rb.warnings:
        print(f"  · {w}")
    # Output-length estimate: ≈ 35 tokens per item (evidence + answer), measured ≈ 660 for 16 items.
    est = 60 + 38 * len(rb.items)
    print(f"  expected output ≈ {est} tokens/trace (v0 with 16 items: ≈ 660 → 4.5 h per 711 traces "
          f"on Qwen3-32B) → ≈ {4.5 * est / 660:.1f} h per full pass\n")
    if show_prompt:
        print(sp)
        try:
            from .data import load as load_split
            print("\n----- USER (first train trace, truncated) -----\n" + user_prompt(load_split("train")[0])[:1500])
        except SystemExit:
            print("\n(no data/ yet — run python -m agentrisk.data to preview a user prompt)")
    if show_schema:
        print(json.dumps(json_schema(rb), indent=1))
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rubrics", nargs="*", help="rubric names (default: all in rubrics/)")
    ap.add_argument("--prompt", action="store_true", help="print the full system prompt")
    ap.add_argument("--schema", action="store_true", help="print the JSON schema")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    names = args.rubrics or available()
    ok = all([report(n, args.prompt, args.schema) for n in names])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
