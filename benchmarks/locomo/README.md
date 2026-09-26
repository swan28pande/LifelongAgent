# LoCoMo

[LoCoMo](https://arxiv.org/abs/2402.17753) (Maharana et al., 2024): 10 long conversations
between two people, ~19 sessions each over several months, with ~200 questions per
conversation. Data: `baselines/locomo/data/locomo10.json`.

Question categories: 1 multi-hop, 2 temporal, 3 commonsense, 4 single-hop, 5
adversarial (the correct answer is to decline, because the question has a false premise).

## Files

| File | What it does |
|---|---|
| [common.py](common.py) | Shared by every runner: dataset parsing, the paper's scoring, the single-pass answer prompt, result/trace writing. |
| [run_v3_agentic.py](run_v3_agentic.py) | memory_v3 with its **tool-calling agent**: ingest every session, build summaries, then the agent decides which tools to call per question. |
| [run_v3_onepass.py](run_v3_onepass.py) | memory_v3's **store, read in one pass**: one fixed retrieval (top conversation chunks + top summaries + all preference rows) and one LLM call. Reads the stores built by `run_v3_agentic.py` and does not re-ingest. |
| [run_mem0.py](run_mem0.py) | **mem0** (open-source v2.x) baseline, single pass: mem0 extracts facts per session; each question retrieves the top-k memories and one LLM call answers. |

## Comparing systems: the 2×2

The agentic runner differs from the others in two ways at once: what it stores and how
it reads. These runners separate the two:

| | One-pass read | Agentic read |
|---|---|---|
| **mem0 store** | `run_mem0.py` | *(not built yet)* |
| **v3 store** | `run_v3_onepass.py` | `run_v3_agentic.py` |

`run_v3_onepass.py` and `run_mem0.py` use the same answer prompt (`ANSWER_SYSTEM` in
`common.py`), so between them only the retrieved context differs. Each trace line also
records `context_chars` and `seconds`, so accuracy can be compared against cost.

## Scoring

`common.score_answer` reproduces the LoCoMo paper's metric exactly:

- cat 1: split-F1 over comma-separated sub-answers
- cat 2, 3, 4: token F1 with Porter stemming
- cat 5: 1.0 only if the response contains an abstention phrase ("don't know",
  "not mentioned", ...)

Known limits of this metric (see `results/locomo/README.md`): long but correct answers
lose precision, "Three" vs "3" scores 0, and on cat 5 an answer that correctly points
out the false premise ("that was Caroline, not Melanie") scores 0. mem0's own benchmark
uses an LLM judge on categories 1–4 instead, so its published numbers are not on this scale.

## Usage

```bash
venv/bin/python benchmarks/locomo/run_v3_agentic.py --conversations 1    # conv-26 only
venv/bin/python benchmarks/locomo/run_v3_onepass.py --conversations 1
venv/bin/python benchmarks/locomo/run_mem0.py       --conversations 1
```

Common flags: `--conversations N` (first N conversations), `--questions N` (cap per
conversation), `--categories 1 2 3 4` (subset), `--model`, `--run NAME` (output folder
under `results/locomo/`). Runner-specific: `--no-summaries` (agentic),
`--k-chunks`/`--k-summaries`/`--store-run` (one-pass), `--top-k` (mem0).
