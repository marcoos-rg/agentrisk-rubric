"""Summarise a run, or compare two runs of the same rubric.

  python -m agentrisk.analyze results/v0_alpha__qwen3-32b__train
  python -m agentrisk.analyze results/RUN_A results/RUN_B          # run-to-run comparison

Report (also saved as summary.json inside the run folder):
  * validity: valid JSON, failed traces, evidence warnings
  * gate: level-1 traces closed by the gate, and the dangerous error (gate=NO on gold ≥ 2)
  * per item: YES rate, mean P(chosen answer), doubtful answers (0.1 < P < 0.9) and, for items
    tagged with an official risk driver, precision / recall against the gold drivers
  * Spearman correlation between the summed feature vector (uniform weights) and gold severity —
    the baseline any learned weighting must beat
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .logprobs import doubtful
from .prompt import encode_hard, encode_soft
from .rubric import REPO, RubricError, load
from .run import read_records
from .taxonomy import DRIVERS, GATE


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** .5
    vy = sum((b - my) ** 2 for b in ry) ** .5
    return round(cov / (vx * vy), 3) if vx and vy else None


def _score(vec: list[float], names: list[str]) -> float:
    """Uniform-weight risk score: YES/UNK columns count 1, graded columns are scaled to [0, 1]."""
    s = 0.0
    for v, n in zip(vec, names):
        if n.endswith("=UNK"):
            continue
        s += v / 3 if "=" not in n else v
    return s


def summarise(run: Path) -> dict:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    rb = load(meta["rubric"])
    if rb.sha256 != meta["rubric_sha256"]:
        print(f"⚠ rubrics/{rb.name}.py changed since this run (sha {meta['rubric_sha256']} → {rb.sha256}); "
              "the report uses the current file — check out the run's commit if items were renamed.")
    recs = read_records(run / "run.jsonl")
    ok = [r for r in recs.values() if r.get("ok")]
    rep = {"run": run.name, "rubric": rb.name, "rubric_version": meta["rubric_version"], "model": meta["model"],
           "traces": len(recs), "valid": len(ok), "failed": sorted(r["id"] for r in recs.values() if not r.get("ok"))}
    warns = [w for r in ok for w in r["warnings"]]
    rep["warnings"] = {"total": len(warns), "quote_not_verbatim": sum("not verbatim" in w for w in warns),
                       "missing_quote": sum("without quote" in w for w in warns),
                       "gate_cleanups": sum("gate=NO" in w for w in warns)}
    if not ok:
        return rep

    if rb.gate:
        gid = rb.gate["id"]
        lvl1 = [r for r in ok if r["gold"] == 1]
        hi = [r for r in ok if (r["gold"] or 0) >= 2]
        rep["gate"] = {
            "level1_closed": f"{sum(r['items'][gid]['answer'] == 'NO' for r in lvl1)}/{len(lvl1)}",
            "dangerous_gate_no_on_gold_ge2": sorted(r["id"] for r in hi if r["items"][gid]["answer"] == "NO"),
            "gate_yes_on_level1": sorted(r["id"] for r in lvl1 if r["items"][gid]["answer"] != "NO")}

    items = {}
    for it in rb.items:
        iid = it["id"]
        ans = [r["items"][iid] for r in ok]
        row = {"kind": it["kind"]}
        if it["kind"] == "multi":
            row["selected_rate"] = {o: round(sum(o in a["selected"] for a in ans) / len(ans), 2) for o in it["options"]}
        else:
            vals = [a["answer"] for a in ans]
            row["dist"] = {v: vals.count(v) for v in sorted(set(vals))}
            pc = [r["probs"][iid]["probs"].get(r["items"][iid]["answer"], 0.0) for r in ok
                  if r.get("probs", {}).get(iid, {}).get("probs")]
            if pc:
                row["mean_p_chosen"] = round(sum(pc) / len(pc), 3)
                row["doubtful"] = sorted(r["id"] for r in ok if iid in doubtful(r.get("probs") or {}))
            drv = [o for o in it["official"] if o in DRIVERS]
            if drv and it["kind"] == "binary" and it["block"] != GATE:
                fired = [r for r in ok if r["items"][iid]["answer"] == "YES"]
                truth = [r for r in ok if set(drv) & set(r.get("gold_drivers") or [])]
                tp = sum(1 for r in fired if set(drv) & set(r.get("gold_drivers") or []))
                row["vs_gold_driver"] = {"drivers": drv, "precision": f"{tp}/{len(fired)}",
                                         "recall": f"{tp}/{len(truth)}"}
        items[iid] = row
    rep["items"] = items

    from .prompt import feature_names
    names = feature_names(rb)
    gold = [r["gold"] for r in ok]
    hard = [_score(encode_hard(r["items"], rb), names) for r in ok]
    soft = [_score(encode_soft(r["items"], r.get("probs") or {}, rb), names) for r in ok]
    eff = [i for i, r in enumerate(ok) if r["gold"] and r["gold"] >= 2]
    rep["spearman_uniform_sum_vs_gold"] = {
        "hard": spearman(hard, gold), "soft": spearman(soft, gold),
        "hard_gold_ge2": spearman([hard[i] for i in eff], [gold[i] for i in eff]),
        "soft_gold_ge2": spearman([soft[i] for i in eff], [gold[i] for i in eff])}
    return rep


def compare(a: Path, b: Path) -> dict:
    ra, rb_ = read_records(a / "run.jsonl"), read_records(b / "run.jsonl")
    common = sorted(i for i in ra if i in rb_ and ra[i].get("ok") and rb_[i].get("ok"))
    flips, tvs, multi = [], [], []
    for i in common:
        for iid, x in ra[i]["items"].items():
            y = rb_[i]["items"].get(iid)
            if y is None:
                continue
            if "selected" in x:
                if sorted(x["selected"]) != sorted(y["selected"]):
                    multi.append(f"{i} {iid}: {x['selected']} → {y['selected']}")
                continue
            if x["answer"] != y["answer"]:
                flips.append(f"{i} {iid}: {x['answer']} → {y['answer']}")
            px = (ra[i].get("probs") or {}).get(iid, {}).get("probs")
            py = (rb_[i].get("probs") or {}).get(iid, {}).get("probs")
            if px and py:
                tvs.append((0.5 * sum(abs(px.get(k, 0) - py.get(k, 0)) for k in set(px) | set(py)), f"{i} {iid}"))
    tvs.sort(reverse=True)
    return {"common_traces": len(common), "answer_flips": flips, "multi_select_changes": multi,
            "mean_tv": round(sum(t for t, _ in tvs) / len(tvs), 4) if tvs else None,
            "largest_tv": [f"{n}: {t:.3f}" for t, n in tvs[:5]]}


def _print(rep: dict) -> None:
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="one run folder, or two to compare")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    runs = [p if p.is_absolute() or p.exists() else REPO / p for p in args.runs]
    try:
        if len(runs) == 1:
            rep = summarise(runs[0])
            (runs[0] / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
        elif len(runs) == 2:
            rep = compare(*runs)
        else:
            sys.exit("give one run folder, or two to compare")
    except RubricError as e:
        sys.exit(str(e))
    _print(rep)


if __name__ == "__main__":
    main()
