# LoCoMo results

One folder per system, each produced by the runner of the same name in
[benchmarks/locomo/](../../benchmarks/locomo/). All three runs cover conversation
conv-26 (199 questions) with Gemini 3.5 Flash and `nomic-embed-text-v1`.

## Scoreboard (paper token F1)

| Category | n | [v3_agentic](v3_agentic/) | [v3_onepass](v3_onepass/) | [mem0](mem0/) |
|---|---|---|---|---|
| single-hop | 70 | **0.667** | 0.617 | 0.464 |
| temporal | 37 | 0.605 | **0.634** | 0.617 |
| multi-hop | 32 | 0.394 | **0.422** | 0.348 |
| commonsense | 13 | **0.222** | 0.213 | 0.174 |
| adversarial | 47 | 0.149 | 0.809 | **0.894** |
| **overall** | 199 | 0.460 | **0.608** | 0.557 |
| **overall excluding adversarial** | 152 | **0.557** | 0.546 | 0.452 |
| LLM calls per question | | ~8 (7 tool calls) | 1 | 1 |
| seconds per question | | ~30 | 4.4 | 3.4 |

## How to read it

The runners are laid out as a 2×2 of store × read path (see
[benchmarks/locomo/README.md](../../benchmarks/locomo/README.md)):

- **Same prompt, different store** (`v3_onepass` vs `mem0`): 0.546 vs 0.452 excluding
  adversarial. v3's store (raw conversation chunks + summaries + preference rows) gives
  the answerer more to work with than mem0's ~6 extracted facts per session. mem0 says
  "I don't know" on 15/70 single-hop questions; v3 one-pass on 8/70.
  **Caveat:** the context budgets differ. v3 one-pass passes ~12k characters; mem0's top 10
  memories are far shorter. An equal-budget run is still to do.
- **Same store, different read path** (`v3_agentic` vs `v3_onepass`): nearly equal
  outside adversarial (0.557 vs 0.546), with the agent ahead on single-hop and behind on
  temporal and multi-hop, at about 7× the calls and latency. **Caveat:** the two also use
  different answer prompts (`memory_v3/prompts.py:CHAT_SYSTEM` vs `common.ANSWER_SYSTEM`).
- **Adversarial** mostly measures how readily the prompt says "I don't know". The agent's
  prompt leads it to correct false premises ("that was Caroline, not Melanie"), which
  this metric scores as 0 even when right.

Token F1 also under-rates correct long answers. Judged by meaning, v3_agentic is 83%
correct overall (`v3_agentic/manual_judgement.jsonl`). mem0's own benchmark
(github.com/mem0ai/memory-benchmarks) uses a lenient LLM judge on categories 1–4 with
top-200 retrieval and gpt-5, so its published 91.6% is on a different scale.

## Next

- Re-score all three with the same LLM judge (categories 1–4), so they sit on one scale
  with each other and with mem0's published numbers.
- Equal-context-budget one-pass runs.
- The missing cell: the agent with mem0's search as its only tool.
- The remaining 9 conversations.
