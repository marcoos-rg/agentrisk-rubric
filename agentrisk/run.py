"""Run a rubric over AgentRisk traces against any OpenAI-compatible server (vLLM on Kaggle by
default) and save answers, per-item probabilities and feature vectors.

  python -m agentrisk.run --rubric v0_alpha --n 20                 # 20 stratified train traces
  python -m agentrisk.run --rubric v0_alpha --ids AGENTRISK-014 AGENTRISK-041
  python -m agentrisk.run --rubric v0_alpha --all --split dev      # every dev trace

Output: results/<rubric>__<model>__<split>[__<tag>]/
  meta.json          rubric name/version/sha, model, args, git commit, timestamps
  run.jsonl          one record per trace (append-only; the run RESUMES from it)
  features_hard.csv  0/1 features (YES = 1, graded 0..3)
  features_soft.csv  same columns, P(YES) and E[value] from logprobs
Re-running the same command skips traces already answered, so an interrupted Kaggle session
loses nothing. Changing the rubric file changes its sha → use a new --tag or --overwrite.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from . import data
from .logprobs import doubtful, item_probs
from .prompt import encode_hard, encode_soft, feature_names, json_schema, system_prompt, user_prompt, validate
from .rubric import REPO, Rubric, RubricError, load

# Kaggle sets AGENTRISK_RESULTS=/kaggle/working/results so the files appear in the notebook Output.
RESULTS = Path(os.environ.get("AGENTRISK_RESULTS", REPO / "results"))


def run_dir(rb: Rubric, model: str, split: str, tag: str | None) -> Path:
    name = f"{rb.name}__{model.replace('/', '-')}__{split}" + (f"__{tag}" if tag else "")
    return RESULTS / name


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "rubrics", "agentrisk"], cwd=REPO,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except Exception:
        return None


def read_records(path: Path) -> dict[str, dict]:
    """Last record per trace id (a failed trace that was retried keeps only the newest line)."""
    recs = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                recs[r["id"]] = r
    return recs


def answer_text(msg) -> str:
    text = (msg.content or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # thinking leaked into content
    return text.removeprefix("```json").removesuffix("```").strip()


def call_one(client, rb: Rubric, ex: dict, args) -> dict:
    rec = {"id": ex["example_id"], "gold": ex.get("risk_severity"), "gold_domains": ex.get("risk_domains"),
           "gold_drivers": ex.get("risk_drivers")}
    rf = {"type": "json_schema", "json_schema": {"name": "rubric", "schema": json_schema(rb), "strict": False}}
    kw = {"logprobs": True, "top_logprobs": args.top_logprobs} if args.top_logprobs else {}
    extra = None if args.no_think_kwarg else {"chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=args.model, temperature=args.temperature, max_tokens=args.max_tokens, response_format=rf,
            extra_body=extra, messages=[{"role": "system", "content": system_prompt(rb)},
                                        {"role": "user", "content": user_prompt(ex)}], **kw)
    except Exception as e:  # network / server errors: recorded, retried on the next run
        return {**rec, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}", "latency": time.time() - t0}
    ch = r.choices[0]
    rec.update(latency=round(time.time() - t0, 2), finish=ch.finish_reason,
               usage=r.usage.model_dump() if r.usage else None)
    if ch.finish_reason == "length":
        return {**rec, "ok": False, "error": f"truncated at max_tokens={args.max_tokens}"}
    try:
        raw = json.loads(answer_text(ch.message))
        items, warns = validate(raw, ex, rb)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return {**rec, "ok": False, "error": f"bad answer: {type(e).__name__}: {e}"}
    probs = {}
    lp = getattr(ch, "logprobs", None)
    if lp and lp.content:
        probs = item_probs([t.token for t in lp.content],
                           [[(a.token, a.logprob) for a in (t.top_logprobs or [])] for t in lp.content], rb)
    return {**rec, "ok": True, "summary": raw.get("action_summary", ""), "items": items, "probs": probs,
            "warnings": warns}


def write_features(rb: Rubric, recs: dict[str, dict], out: Path) -> int:
    names = feature_names(rb)
    ok = [r for r in recs.values() if r.get("ok")]
    ok.sort(key=lambda r: r["id"])
    for kind, enc in (("hard", lambda r: encode_hard(r["items"], rb)),
                      ("soft", lambda r: encode_soft(r["items"], r.get("probs") or {}, rb))):
        with (out / f"features_{kind}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["example_id", "gold", *names])
            for r in ok:
                w.writerow([r["id"], r["gold"], *enc(r)])
    return len(ok)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rubric", default="v0_alpha", help="file stem in rubrics/ (or a path to a .py file)")
    ap.add_argument("--base-url", default=os.environ.get("AGENTRISK_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--key-env", default="AGENTRISK_API_KEY", help="env var holding the server's API key")
    ap.add_argument("--model", default="qwen3-32b", help="served model name")
    ap.add_argument("--split", default="train", choices=["train", "dev"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--n", type=int, default=20, help="number of stratified traces (default 20)")
    g.add_argument("--ids", nargs="+", help="explicit example ids")
    g.add_argument("--all", action="store_true", help="every trace of the split")
    ap.add_argument("--conc", type=int, default=6, help="concurrent requests (32B on 2×T4: 6)")
    ap.add_argument("--top-logprobs", type=int, default=5, help="0 disables logprobs (soft = hard)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-think-kwarg", action="store_true", help="do not send enable_thinking=False")
    ap.add_argument("--tag", help="suffix for the results folder, e.g. your initials or an experiment name")
    ap.add_argument("--overwrite", action="store_true", help="discard previous results in the same folder")
    return ap


def main(argv: list[str] | None = None, client=None) -> Path:
    args = build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    try:
        rb = load(args.rubric)
    except RubricError as e:
        sys.exit(str(e))

    rows = data.load(args.split)
    todo_rows = data.pick(rows, args.ids, None if args.all else (None if args.ids else args.n))
    out = run_dir(rb, args.model, args.split, args.tag)
    out.mkdir(parents=True, exist_ok=True)
    meta_path, log_path = out / "meta.json", out / "run.jsonl"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    if meta and not args.overwrite and meta["rubric_sha256"] != rb.sha256:
        sys.exit(f"{out.name} holds results of another version of {rb.name}.py "
                 f"(sha {meta['rubric_sha256']} ≠ {rb.sha256}). Use --tag NAME or --overwrite.")
    if args.overwrite:
        log_path.unlink(missing_ok=True)
        meta = None
    meta = meta or {"created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    meta.update(rubric=rb.name, rubric_version=rb.version, rubric_sha256=rb.sha256, model=args.model,
                split=args.split, git_commit=git_commit(), base_url=args.base_url,
                args={k: v for k, v in vars(args).items() if k not in ("overwrite",)},
                updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    recs = read_records(log_path)
    todo = [ex for ex in todo_rows if not recs.get(ex["example_id"], {}).get("ok")]
    print(f"rubric {rb.name} ({rb.version}, {len(rb.items)} items, sha {rb.sha256}) · model {args.model}\n"
          f"{len(todo_rows)} traces requested · {len(todo_rows) - len(todo)} already done · {len(todo)} to run "
          f"· concurrency {args.conc} · logprobs {'top-' + str(args.top_logprobs) if args.top_logprobs else 'off'}\n"
          f"→ {out}")

    if todo:
        if client is None:
            import httpx
            from openai import OpenAI
            client = OpenAI(base_url=args.base_url, api_key=os.environ.get(args.key_env, "EMPTY"),
                            max_retries=2, timeout=900, http_client=httpx.Client(timeout=900))
        t0, done, failed = time.time(), 0, 0
        with ThreadPoolExecutor(args.conc) as pool, log_path.open("a", encoding="utf-8") as f:
            futs = [pool.submit(call_one, client, rb, ex, args) for ex in todo]
            for fut in as_completed(futs):
                rec = fut.result()
                recs[rec["id"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                failed += not rec["ok"]
                eta = (time.time() - t0) / done * (len(todo) - done)
                if rec["ok"]:
                    g = rec["items"][rb.gate["id"]]["answer"] if rb.gate else "-"
                    info = f"gate={g} · doubtful={len(doubtful(rec['probs']))} · warnings={len(rec['warnings'])}"
                else:
                    info = rec["error"][:120]
                print(f"  [{done}/{len(todo)}] {'✓' if rec['ok'] else '✗'} {rec['id']} gold {rec['gold']} · "
                      f"{rec.get('latency', 0):5.1f}s · {info} · ETA {eta / 60:.0f} min", flush=True)
        wall = time.time() - t0
        print(f"\nDone in {wall / 60:.1f} min · {failed} failed (rerun the same command to retry them) · "
              f"{len(todo) / wall * 3600:.0f} traces/hour")

    n = write_features(rb, recs, out)
    print(f"features for {n} traces → {out.name}/features_hard.csv, features_soft.csv\n"
          f"next: python -m agentrisk.analyze {out.as_posix()}")
    return out


if __name__ == "__main__":
    main()
