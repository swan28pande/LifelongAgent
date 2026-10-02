# Research plan: is memory_v3 enough?

## Objective

Assess whether memory_v3 is suitable for complex lifelong memory tasks, whether full-context models could replace it, and whether stronger retrieval or memory designs are likely to outperform it.

## Scope and terms

- **Full context:** supplying the whole available history to the answer model for each question.
- **Simple RAG:** retrieving a few relevant raw conversation chunks.
- **Memory system:** any persistent structured, summarized, or indexed representation used across sessions.
- Compare systems only under matched datasets, models, answer prompts, scoring, context budgets, and costs. Separate local evidence from published results.
- Root CONTEXT.md is absent; docs/adr contains no ADR files.

## Questions and evidence

1. What can the current code reliably represent, retrieve, and update? Inspect source, tests, and archived runs.
2. How well does full context work on long histories, temporal updates, multi-hop questions, and cost? Prefer primary benchmark papers with full-context baselines.
3. What do stronger RAG and memory systems add, and what published comparisons actually isolate their benefits? Prefer primary papers, official repos, and matched ablations.
4. What experiment would fairly test whether v3's dated rows, hierarchy, and agentic reader add value over full context and simple RAG?

## Counter-evidence to seek

- Cases where full context beats memory systems.
- Cases where simple RAG equals or beats more elaborate memory.
- Cases where memory wins only because it uses a stronger model, more context, more calls, or a different scorer.
- Tasks where v3's exact dated ledger is useful even if broad LoCoMo scores are weak.

## Stopping conditions

Stop when primary sources establish the main tradeoffs, local evidence is checked, and remaining uncertainty is labeled rather than covered by cross-benchmark rankings.
