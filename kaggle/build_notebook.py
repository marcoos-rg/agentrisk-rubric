"""Generate kaggle/agentrisk_kaggle.ipynb.

The notebook does NOT embed the project code: it clones this repository, so a rubric pushed to
GitHub is used by the next Kaggle run without rebuilding the notebook. Rebuild only when this
builder changes (new cells, new presets, another repository URL):

  python kaggle/build_notebook.py --repo-url https://github.com/OWNER/REPO.git
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo-url", default="https://github.com/OWNER/agentrisk-rubric.git")
ap.add_argument("--branch", default="main")
args = ap.parse_args()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


cells = [
    md(f"""
# AgentRisk rubric · Qwen3 on vLLM (Kaggle 2×T4)
Clones **[the team repository]({args.repo_url.removesuffix('.git')})**, serves an open model with vLLM
inside this notebook, runs a rubric over AgentRisk traces and saves answers, per-item probabilities
and feature vectors to the **Output** panel (`/kaggle/working/results/…` and `results.zip`).

**Before running**
1. *Settings → Accelerator → GPU T4 ×2* · *Settings → Internet → On* (needs a phone-verified account).
2. Private repository only: *Add-ons → Secrets* → add `GITHUB_TOKEN` (fine-grained token, read-only
   *Contents* on this repository) and tick it for this notebook. Public repository: nothing to do.
3. Edit the **Configuration** cell below (rubric, branch, how many traces).
4. Short test (≤ 20 traces): *Run all* in the interactive session.
   Long run (hundreds of traces): *Save Version → Save & Run All (Commit)*; it keeps running after you
   close the tab (max 12 h) and the results appear in the version's Output.

First run: installing vLLM + downloading Qwen3-32B-AWQ (≈ 19 GB) and loading it takes ≈ 15–25 min.
"""),
    code(f"""
# ── Configuration ──────────────────────────────────────────────────────────────────────────
REPO_URL = "{args.repo_url}"
BRANCH = "{args.branch}"          # your branch, e.g. "rubric/v1-oversight"
RUBRIC = "v0_alpha"          # file stem in rubrics/ (python -m agentrisk.check_rubric lists them)
SPLIT = "train"              # "train" (568 traces) or "dev" (143)
N = 20                       # stratified traces (1, 3, 4, 2, 5 in turn); ignored if IDS or RUN_ALL
IDS = None                   # e.g. ["AGENTRISK-014", "AGENTRISK-041"]
RUN_ALL = False              # True → every trace of the split (≈ 4.5 h per 711 traces with v0 on 32B)
TAG = None                   # suffix for the results folder, e.g. your initials or experiment name
MODEL_PRESET = "32b"         # "32b" (default, Qwen3-32B-AWQ) or "14b" (faster, much weaker on drivers)
CONC = None                  # concurrent requests; None → preset value
"""),
    code("""
PRESETS = {
    # 32B AWQ ≈ 9.5 GB of weights per T4 → ≈ 15K tokens of KV cache → ≈ 6 traces in flight.
    "32b": dict(model="Qwen/Qwen3-32B-AWQ", served="qwen3-32b", max_len=8192, gpu_util=0.93, conc=6),
    # 14B AWQ ≈ 4.7 GB per T4 → ≈ 74K tokens of KV cache.
    "14b": dict(model="Qwen/Qwen3-14B-AWQ", served="qwen3-14b", max_len=16384, gpu_util=0.90, conc=8),
}
P = PRESETS[MODEL_PRESET]
CONC = CONC or P["conc"]
REPO_DIR = "/kaggle/working/repo"
RESULTS_DIR = "/kaggle/working/results"

import os, subprocess, sys

def sh(*cmd, cwd=REPO_DIR, check=True, env=None):
    \"\"\"Run a command and stream its output into the notebook (subprocess output is not shown otherwise).\"\"\"
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **(env or {})})
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() and check:
        raise RuntimeError(f"command failed (exit {p.returncode}): {' '.join(cmd[:4])} …")
    return p.returncode
print(P, "· concurrency", CONC)
"""),
    md("## 1 · Get the code (clone or update the repository)"),
    code("""
import base64
token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass  # public repository, or no secret attached
# The token is passed as an HTTP header for this command only: it is never written to .git/config.
auth = (["-c", "http.extraHeader=Authorization: Basic "
         + base64.b64encode(f"x-access-token:{token}".encode()).decode()] if token else [])
if os.path.isdir(f"{REPO_DIR}/.git"):
    sh("git", *auth, "fetch", "origin", BRANCH)
    sh("git", "checkout", "-B", BRANCH, f"origin/{BRANCH}")
else:
    sh("git", *auth, "clone", "--branch", BRANCH, REPO_URL, REPO_DIR, cwd="/kaggle/working")
sh("git", "log", "-1", "--format=commit %h · %an · %ad · %s", "--date=short")
"""),
    md("## 2 · Check the rubric (fails fast, before the long model load)"),
    code("""
sh(sys.executable, "-m", "agentrisk.check_rubric", RUBRIC)
sh(sys.executable, "-m", "agentrisk.data")
"""),
    md("## 3 · Install vLLM"),
    code("""
import time; _t = time.time()
# uv resolves and installs much faster than pip against Kaggle's preinstalled packages.
sh("pip", "install", "-q", "uv", cwd="/kaggle/working")
sh("uv", "pip", "install", "--system", "vllm", "openai", "httpx", cwd="/kaggle/working")
print(f"install took {(time.time() - _t)/60:.1f} min")
"""),
    md("## 4 · Start the vLLM server (background, localhost only)"),
    code("""
import secrets, time
API_KEY = secrets.token_hex(16)          # local only; never leaves the notebook
os.environ["AGENTRISK_API_KEY"] = API_KEY
log = open("/kaggle/working/vllm.log", "w")
server = subprocess.Popen(
    [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", P["model"],
     "--served-model-name", P["served"], "--tensor-parallel-size", "2", "--dtype", "half",
     "--max-model-len", str(P["max_len"]), "--gpu-memory-utilization", str(P["gpu_util"]),
     "--enable-prefix-caching", "--max-logprobs", "20", "--api-key", API_KEY,
     "--host", "127.0.0.1", "--port", "8000"], stdout=log, stderr=subprocess.STDOUT)
T_START = time.time()
print("vLLM starting in the background (pid", server.pid, ") — the next cell waits for it")
"""),
    md("Waits for the server, printing progress every minute. **Safe to interrupt (■) and rerun**: it never starts a second server."),
    code("""
import urllib.request

def last_log_line():
    lines = [l.strip() for l in open("/kaggle/working/vllm.log", errors="replace") if l.strip()]
    return lines[-1][-160:] if lines else "(log empty)"

last_print = 0
while True:
    if server.poll() is not None:
        print(open("/kaggle/working/vllm.log").read()[-4000:])
        raise RuntimeError("vLLM exited — see the log above (try MODEL_PRESET='14b' or a lower max_len)")
    try:
        if urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5).status == 200:
            break
    except Exception:
        pass
    elapsed = time.time() - T_START
    if elapsed > 2400:
        raise TimeoutError("server not ready after 40 min — check /kaggle/working/vllm.log")
    if time.time() - last_print >= 60:
        print(f"[{elapsed/60:4.1f} min] {last_log_line()}", flush=True)
        last_print = time.time()
    time.sleep(10)
print(f"Server ready after {(time.time() - T_START)/60:.1f} min")
sh("bash", "-c", "grep -iE 'KV cache|maximum concurrency' /kaggle/working/vllm.log | tail -3", check=False)
"""),
    md("## 5 · Run the rubric\nResumable: if the session stops, rerun this cell (same configuration) and it continues where it left off."),
    code("""
args = ["--rubric", RUBRIC, "--model", P["served"], "--split", SPLIT, "--conc", str(CONC)]
args += ["--ids", *IDS] if IDS else (["--all"] if RUN_ALL else ["--n", str(N)])
args += ["--tag", TAG] if TAG else []
ENV = {"AGENTRISK_RESULTS": RESULTS_DIR}
sh(sys.executable, "-m", "agentrisk.run", *args, env=ENV)
"""),
    md("## 6 · Analyse"),
    code("""
import glob
model_tag = P["served"]
RUN_DIR = f"{RESULTS_DIR}/{RUBRIC}__{model_tag}__{SPLIT}" + (f"__{TAG}" if TAG else "")
sh(sys.executable, "-m", "agentrisk.analyze", RUN_DIR)
"""),
    md("## 7 · Package the results (download `results.zip` from the Output panel)"),
    code("""
sh("bash", "-c", "cd /kaggle/working && zip -qr results.zip results vllm.log && ls -la results.zip", cwd="/kaggle/working")
"""),
    md("## Stop the server (frees the GPUs)"),
    code("""
server.terminate(); server.wait(timeout=60); print("stopped")
"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}}
out = Path(__file__).with_name("agentrisk_kaggle.ipynb")
with out.open("w", encoding="utf-8", newline="\n") as f:   # LF on every OS (see .gitattributes)
    f.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out.name}: {len(cells)} cells · repo {args.repo_url} · branch {args.branch}")
