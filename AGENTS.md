# Instructions for AI coding agents

Read `README.md` first for the purpose and layout of this repository. This file is the short list
of rules that keep the team's results comparable. Follow it unless a human teammate explicitly
tells you otherwise in the conversation.

## What you will usually be asked to do

| Request | Where to work | Do not touch |
|---|---|---|
| "Write / improve a rubric", "add questions about X" | a **new** file `rubrics/<version>_<name>.py`, copied from `rubrics/TEMPLATE.py` | `agentrisk/`, existing rubric files that have results |
| "Run it", "test it on N traces" | the Kaggle notebook configuration cell, or `python -m agentrisk.run …` against a server the human provides | — |
| "Why did item X fail?", "compare runs" | `python -m agentrisk.analyze …`, read `run.jsonl` | — |
| Change the pipeline itself | `agentrisk/` + `tests/`, on a branch, with a pull request | rubric files |

## Invariants (breaking one silently corrupts experiments)

1. **Rubric content lives only in `rubrics/*.py`.** `agentrisk/` must stay rubric-agnostic: never
   hard-code an item id there (the only special case is the `gate` block).
2. **Do not edit a rubric file that already has results** (including `v0_alpha.py`, the frozen
   baseline). Copy it to a new file with a new `VERSION`. Results store the sha256 of the rubric
   file, and `run.py` refuses to mix versions.
3. **Item semantics:** closed factual questions; **YES always means more risk**; never ask the
   model for a risk level or severity; prefer binary over graded and multi-select items. Every
   item has `official` tags from `agentrisk/taxonomy.py` (do not invent new tag strings; extend
   `taxonomy.py` only if the official task definition changes).
4. **The JSON schema puts `answer` after `step` and `quote` in every item.** `logprobs.py` finds
   the answer token by that order. Do not reorder these keys in `prompt.json_schema`.
5. **The gate, if present, is the first item, binary, YES = the action has an external effect.**
   A gate = NO answer forces every other item to NA (and zero features). Keep that behaviour.
6. **No secrets in the repository.** API keys come from environment variables
   (`AGENTRISK_API_KEY`) or Kaggle Secrets (`GITHUB_TOKEN`). Never write a key into code,
   notebooks, config files or commit messages. Do not print tokens.
7. **No data or results in git.** `data/` and `results/` are git-ignored; keep it that way.
8. Notebook changes go through `kaggle/build_notebook.py` (then regenerate the `.ipynb`), not by
   editing the JSON of `agentrisk_kaggle.ipynb` by hand.

## Checks to run before every commit

```bash
python -m agentrisk.check_rubric          # every rubric must load; read the coverage + cost lines
python tests/test_offline.py              # must print "all tests passed"
```

`check_rubric` exits with code 1 on any structural error. Treat its warnings (graded or
multi-select items, factors with fewer than 2 items, missing drivers) as design feedback to
discuss with the human, not as errors to silence.

## Writing good rubric items (checklist)

- One observable fact per question; name the actor (`the USER`, `an OBSERVATION`, `the agent`).
- Give a concrete `yes` and `no` example, ideally drawn from real train traces (cite the
  example id in `note`, never paste long trace text into the rubric).
- Avoid items that are YES on almost every trace (v0's "effect is immediate" was YES on 91 %):
  they carry no information. `analyze` shows each item's answer distribution.
- Look at the logprobs: an item whose chosen answer often has P between 0.3 and 0.7
  (`doubtful` in `analyze`) is ambiguous — rewrite its question or its yes/no anchors.
- Keep the total size in mind: each item costs ≈ 38 output tokens per trace (Kaggle time).

## Environment notes

- Python ≥ 3.10; runtime dependencies are only `openai` and `httpx`; everything else is stdlib.
- The default model is `qwen3-32b` served by vLLM inside the Kaggle notebook
  (`Qwen/Qwen3-32B-AWQ`, 2×T4, max_len 8192, 6 concurrent requests). Thinking mode is disabled
  with `chat_template_kwargs={"enable_thinking": False}`; `--no-think-kwarg` turns that off for
  servers that reject the field.
- You cannot reach the Kaggle GPU from a local machine. To test pipeline changes locally use the
  fake server in `tests/test_offline.py`; for real runs, hand the human the configuration values
  for the notebook.
