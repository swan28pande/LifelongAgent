# Experiments

One harness for comparing memory methods on memory benchmarks. Every method implements
the same interface, every benchmark is loaded into the same format, and one runner does
ingestion, answering, grading, resuming and cost logging — so the only thing that changes
between two rows of a results table is the memory method.

```bash
venv/bin/python -m experiments.run list
venv/bin/python -m experiments.run run --method ours_v3 --benchmark v2 --users u1
venv/bin/python -m experiments.run run --method mem0 --benchmark locomo
venv/bin/python -m experiments.run run --method naive_rag --benchmark longmemeval --sample 100
venv/bin/python -m experiments.run run --method full_context --benchmark v2 --users u1 --seed 1
venv/bin/python -m experiments.run report          # methods × benchmarks table, mean ± sd over seeds
```

Add `--limit 3` to any run for a smoke test. Rerun the same command after an interruption;
finished sessions, answers and grades are skipped.

## Plan

**Methods** (3 controls + 5 systems; see `reports/LLM agent memory baselines.md` for why):

| Method | Role | Status |
|---|---|---|
| `ours_v3` | the proposed method (memory_v3) | ready |
| `full_context` | control: whole history in the prompt | ready |
| `naive_rag` | control: raw-chunk vector retrieval | ready |
| Letta-style agent | control: same agent, raw-history search only | to add |
| `mem0` | Mem0 OSS (ECAI 2025), most-used baseline | ready |
| Zep / Graphiti | bi-temporal knowledge graph (needs Neo4j/FalkorDB) | to add |
| A-MEM | NeurIPS 2025, most-used academic baseline | to add |
| MemoryOS | EMNLP 2025, recency-based hierarchy | to add |
| APEX-MEM or TiMem | closest competitor (SQL + agent / calendar hierarchy) | to add |

Ablations of `ours_v3` (no SQL, no summaries, no agent, one summary level) are separate
methods registered the same way.

**Benchmarks:**

| Benchmark | Loader | Instances | Questions | Notes |
|---|---|---|---|---|
| `v2` (ours) | `benchmarks/v2.py` | one per user (u1–u5) | 50 curated per user | headline results; data in `data/synthetic_v2/` |
| `locomo` | `benchmarks/locomo.py` | 10 conversations | 1,540 (categories 1–4) | category 5 excluded; reports judge accuracy and paper F1 |
| `longmemeval` | `benchmarks/longmemeval.py` | one per question | 500, or `--sample N` stratified | cleaned LongMemEval-S; in `data/` |

## Protocol (held equal across methods)

- **Same models** (`config.py`): `gemini-3.5-flash` for every method's memory construction
  and answering, `nomic-embed-text-v1` embeddings, `gemini-3.1-pro-preview` as the judge.
- **Same question text:** every method gets `prompt_for(question)`, which prefixes the date
  the question is asked on when the benchmark defines one.
- **Same answer prompt** (`config.ANSWER_SYSTEM`) for every method that answers with one
  call over retrieved context. Agentic methods use their own loop.
- **Same judge prompt** (`core/grading.py`), published with the results, plus string
  checks for exact/date items and LoCoMo's paper F1.
- **Native dates:** sessions carry their date; methods pass it through their own fields
  where they have them, otherwise as a `[date]` prefix on each message.
- **Seeds:** run each configuration three times (`--seed 0/1/2`) and report mean ± sd.
- **Cost:** `usage.json` counts the LLM calls and tokens a method makes through its own
  models; library-internal calls (e.g. Mem0's extraction) are noted where they are missing.

## Layout

```
experiments/
  config.py            shared models, prompts and budgets
  run.py               CLI: list / run / report
  core/                types, method interface, runner, grading
  benchmarks/          loaders → Instance(sessions, questions)
  methods/             one file per method, registered in methods/__init__.py
  data/                benchmark datasets (locomo, longmemeval, synthetic_v2)
results/experiments/<benchmark>/<method>/<run>/
  stores/  ingest.jsonl  answers.jsonl  grades.jsonl  usage.json  summary.json
```

## Adding a method

Subclass `core.method.MemoryMethod`, implement `ingest(session)`, `finalize()` and
`answer(question)`, set `persistent = True` if its store survives a restart, and register
it in `methods/__init__.py`. Smoke-test with `--benchmark locomo --conversations 1 --limit 3`.
