# Memory v3 architecture and research-design audit

Research date: 2026-09-28. Status: in progress; the central claim and final experiment remain user decisions.

## Objective

Explain the current memory_v3 implementation well enough to trace a conversation from input through storage, summaries, retrieval, and answer. Assess what the repository's experiments establish about lifelong memory and changing preferences.

## Scope and definitions

- The implementation is memory_v3 plus its imported memory_v2 store and summarizer. There is no method_v3 directory in this checkout.
- A preference is a dated committed choice from a recurring category. A fact is a relatively stable attribute. An event is a dated occurrence. These are the extractor's terms, not labels assigned by this investigation.
- Cover both batch replay and live converse/flush. Use the 60-day synthetic dataset for the worked trace; use LoCoMo as a second evaluation setting.
- Examine key methods and data shapes, empirical validity, baseline fairness, reproducibility, and a concise comparison with primary prior work.
- Treat saved results as historical artifacts when their model, tool surface, or schema differs from current code.

## Questions and evidence

1. What is the actual agent structure? Inspect construction, prompts, tool surface, and call paths.
2. What data is written and where? Trace extraction, consolidation, deduplication, SQL rows, raw chunks, summaries, and dates.
3. How does a question reach evidence? Inspect reader prompt, tool implementations, and trace API. Distinguish an available route from an observed model route.
4. How do live and batch modes differ? Inspect buffer behavior, flush timing, summary cadence, and benchmark calls.
5. What do synthetic and LoCoMo results establish? Inspect generators, question mix, scorers, baselines, saved artifacts, and version drift.
6. Which parts overlap with prior art? Compare problem, memory representation, update semantics, retrieval, and evaluation setting using primary papers.

## Counter-evidence checks

- Find cases where the documentation describes behavior that current code does not perform.
- Check whether structured records and raw chunks are deduplicated the same way.
- Check whether a scheduled event's effective date and its transcript date trigger the same summary update.
- Check whether benchmark runners isolate store quality from reader prompt, retrieval budget, and answer model.
- Compare the local LoCoMo scorer with the bundled upstream evaluator.
- Test the generality of any cycle claim against the number and diversity of explicit pattern questions.

## Stopping condition

Stop when a reader can trace both paths and identify the source of every material claim in the report. Separate observed code, archived run results, illustrative traces, and inference. Do not claim a current-revision performance result without running that revision under a pinned protocol.
