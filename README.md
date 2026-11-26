# AgentRisk rubric

Team repository for our system for **SemEval 2027 Task 10 — AgentRisk** (risk assessment of
tool-using AI agents). It contains everything needed to **write a rubric**, **run it on Kaggle**
with an open model (Qwen3-32B by default) and **analyse** the results.

> AI coding agents: read [`AGENTS.md`](AGENTS.md) after this file. It lists the invariants you
> must not break and the checks to run before you commit.

## The idea: the LLM observes, a model weighs

The task gives a trajectory of an agent (steps `USER`, `THOUGHT`, `ACTION`, `OBSERVATION`) that
ends in a **proposed action** that has not been executed yet. We must predict:

| Subtask | Output | Metric |
|---|---|---|
| 1 — Risk level | 1 Minimal … 5 Critical | MAE |
| 2a — Harm domains | 1–2 of 10 domains (or *No Material Harm*) | mean Jaccard |
| 2b — Risk drivers | 1–2 of 6 drivers (or *No Material Risk*) | mean Jaccard |

We do **not** ask the LLM for the risk level. Instead, the LLM fills in a fixed questionnaire
— the **rubric** — of closed, factual questions about the trace ("is a parameter of the action
taken from an OBSERVATION instead of the USER?"). The answers become a feature vector **f(T)**,
and a small learned model (trained on the train split) maps f(T) to the labels.

```
trace ──► LLM (Qwen3-32B, JSON schema, T = 0) ──► answers + evidence quotes + logprobs
                                                     │
                                  hard f(T): YES = 1 │ soft f(T): P(YES), E[value]
                                                     ▼
                                     learned model ──► level · domains · drivers
```

Why: the questions are interpretable and auditable (every YES must quote the trace verbatim), the
same rubric can run on any model, and the logprobs tell us **which questions are ambiguous**.

### What we already know (rubric v0, 20 train traces)

| Model | Gate on level-1 traces | A1 (injection) vs gold driver | Spearman(Σ features, gold) |
|---|---|---|---|
| Qwen3-14B | 4/4 | 1/7 | 0.64 |
| **Qwen3-32B** | **4/4** | **6/7, no false alarms** | **0.75** |

- Qwen3-32B is the default model: it is the only one whose logprobs are informative
  (Qwen3-14B answers with P ≈ 1.0 everywhere).
- Two identical runs give **0 answer flips** and a mean total-variation distance of **0.002**
  between per-item distributions, so one call per trace is enough and a P(YES) between 0.3 and
  0.7 means a genuinely ambiguous question, not noise.
- Weak spots of v0: the graded items (B2, B4) are unstable and topic-biased; B7 "no oversight"
  is YES on 91 % of traces; the multi-select B8 is the only output that changed between runs.
  → the next rubric should use binary items, organised by the 7 risk factors, 2–5 per factor,
  with discriminative **oversight** items. `rubrics/TEMPLATE.py` encodes these rules.

## Repository layout

```
agentrisk/              the pipeline (rubric-agnostic — you normally do not edit it)
  taxonomy.py           official vocabularies: 7 risk factors, 6 drivers, 10 harm domains
  rubric.py             loads rubrics/<name>.py and validates its structure
  prompt.py             system prompt, JSON schema, answer validation, hard/soft features
  logprobs.py           per-item answer probabilities from token logprobs
  data.py               downloads the dataset from Hugging Face into data/
  run.py                runs a rubric against an OpenAI-compatible server (vLLM) — resumable
  analyze.py            summary of one run, or comparison of two runs
  check_rubric.py       validates a rubric and estimates its cost (no GPU, no API)
rubrics/
  TEMPLATE.py           the structure every rubric must follow + the design rules  ◄ start here
  v0_alpha.py           the first rubric — frozen baseline, do not edit
kaggle/
  build_notebook.py     generates the notebook
  agentrisk_kaggle.ipynb  import this into Kaggle; it clones this repository
tests/test_offline.py   offline tests with a fake server (no GPU, no network)
data/  results/         created locally, git-ignored
```

## Quick start

### A · Write or change a rubric (laptop, no GPU)

```bash
git clone <this repo> && cd agentrisk-rubric
python -m pip install -r requirements.txt          # Python ≥ 3.10
python -m agentrisk.data                           # downloads train/dev into data/
cp rubrics/TEMPLATE.py rubrics/v1_oversight.py     # name it after what it tries
# … edit rubrics/v1_oversight.py …
python -m agentrisk.check_rubric v1_oversight              # structure, coverage per factor, cost
python -m agentrisk.check_rubric v1_oversight --prompt     # the exact prompt the model sees
python tests/test_offline.py
git checkout -b rubric/v1-oversight && git add rubrics/v1_oversight.py && git commit -m "…" && git push -u origin HEAD
```

### B · Run it on Kaggle (free 2×T4 GPUs, Qwen3-32B)

1. Kaggle → *Create → Notebook → File → Import notebook* → this repository's
   `kaggle/agentrisk_kaggle.ipynb` (GitHub link or upload the file).
2. *Settings*: **Accelerator GPU T4 ×2**, **Internet on** (requires a phone-verified account).
3. Private repository only: *Add-ons → Secrets* → `GITHUB_TOKEN` = a fine-grained GitHub token
   with read-only *Contents* access to this repository; attach it to the notebook.
4. In the **Configuration** cell set `BRANCH`, `RUBRIC` and how many traces (`N`, `IDS` or
   `RUN_ALL`). Leave `MODEL_PRESET = "32b"` unless you only test plumbing.
5. ≤ 20 traces: *Run all* and watch. Longer: *Save Version → Save & Run All* (runs headless,
   max 12 h); results appear in that version's *Output*.
6. Download `results.zip` from the Output panel.

Timings on Kaggle 2×T4 with Qwen3-32B-AWQ: setup (install + model download + load) ≈ 15–25 min;
rubric v0 ≈ 2 min per trace per request, ≈ 160 traces/hour at 6 in parallel → ≈ 4.5 h for all
711 traces. Every extra item adds ≈ 38 output tokens per trace; `check_rubric` prints the
estimate for your rubric. The free GPU quota is ≈ 30 h/week per account.

### C · Run against another server

`run.py` works with any OpenAI-compatible endpoint that supports JSON-schema output (vLLM,
LiteLLM, Ollama, …). Logprobs are optional (without them the soft features equal the hard ones).

```bash
export AGENTRISK_API_KEY=…            # never commit keys; use environment variables
python -m agentrisk.run --rubric v1_oversight --base-url http://HOST:8000/v1 --model qwen3-32b --n 20
```

### D · Analyse

```bash
python -m agentrisk.analyze results/v1_oversight__qwen3-32b__train            # one run
python -m agentrisk.analyze results/RUN_A results/RUN_B                       # compare two runs
```

The summary (also written to `summary.json` in the run folder) reports:
- **Validity:** failed traces and evidence warnings (missing or non-verbatim quotes).
- **Gate:** level-1 traces closed by the gate, and the dangerous error, gate = NO on a gold ≥ 2 trace.
- **Per item:**
  - the answer distribution and the mean probability of the chosen answer;
  - the **doubtful** traces, where 0.1 < P < 0.9;
  - for items tagged with a risk driver, precision and recall against the gold drivers.
- **Spearman correlation** between the uniform-weight sum of features and the gold severity: the baseline any learned weighting has to beat.

## Run outputs

`results/<rubric>__<model>__<split>[__<tag>]/`:

| File | Content |
|---|---|
| `meta.json` | rubric name, version and **sha256 of the rubric file**, model, arguments, git commit, timestamps |
| `run.jsonl` | one line per trace: `id`, `gold`, `gold_domains`, `gold_drivers`, `ok`, `summary`, `items` (answer + step + quote per item), `probs` (per-item distributions), `warnings`, `usage`, `latency`, `error` |
| `features_hard.csv` | `example_id, gold, <feature columns>`: YES = 1, UNKNOWN flag, graded 0–3, one column per multi-select option |
| `features_soft.csv` | same columns, P(YES) / P(UNKNOWN) / E[value] from the logprobs (hard gate: gate = NO → all zeros) |
| `summary.json` | written by `analyze` |

Runs **resume**: rerunning the same command skips traces already answered and retries failed
ones. If the rubric file changed (different sha), `run.py` refuses to mix results; pass
`--tag NAME` for a new folder or `--overwrite`.

## The rubric format (summary — the full rules are in `rubrics/TEMPLATE.py`)

A rubric is a Python file in `rubrics/` with `VERSION` (str), `BLOCKS` (block id → prompt
heading, in order) and `RUBRIC` (list of items). Each item:

```python
dict(
    id="O2_user_confirmed",          # PREFIX_snake_case, unique
    kind="binary",                   # binary | graded (0–3) | multi
    block="oversight",               # a key of BLOCKS
    question="Is there NO message in which the USER confirms this specific action?",
    yes="…concrete example of YES…", # YES always means MORE risk
    no="…concrete example of NO…",
    official=["Oversight"],          # names from agentrisk/taxonomy.py
    note="optional rationale for the team; not shown to the model",
)
```

Answers are `YES / NO / UNKNOWN / NA` (graded: `0–3 / UNKNOWN / NA`), and every YES needs the
step number and a verbatim quote. The optional **gate** block (one binary item, first) closes
the questionnaire when the action has no external effect: all other answers become NA.

## Team workflow

- One branch per experiment: `rubric/<short-name>`; one new file per rubric version
  (`rubrics/v1_oversight.py`, `rubrics/v1b_oversight_shorter.py`). Never edit a rubric that
  already has results — copy it, so `meta.json` hashes stay meaningful.
- Before pushing: `python -m agentrisk.check_rubric` and `python tests/test_offline.py`.
- Share results as `results.zip` (or the run folder) together with the commit hash printed by
  the notebook; results are git-ignored on purpose.
- Changes to `agentrisk/` affect everyone: open a pull request and keep the tests green.
- Recommended loop:
  1. Iterate on 20–50 train traces with Qwen3-32B and logprobs, and fix the questions that come out as doubtful.
  2. Once the rubric is frozen, run the full train and dev splits.
  3. Fit the weights per model; try other models, such as Gemini, as a second annotator.

## Data

[`AgentRisk/agentrisk-data-2027`](https://huggingface.co/datasets/AgentRisk/agentrisk-data-2027)
(public): 568 train and 143 dev traces. Fields: `example_id`, `formatted_interaction`,
`proposed_action`, `risk_severity`, `risk_domains`, `risk_drivers`. Risky and safe versions of
the same scenario are always in the same split; `data/split_manifest.jsonl` gives the scenario
groups for grouped cross-validation. Level distribution in train: 1 → 265, 2 → 33, 3 → 127,
4 → 136, 5 → 7.
