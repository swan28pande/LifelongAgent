# Is `memory_v3` enough for lifelong memory?

**Assessment, 2026-09-29.** Your skepticism is well founded. The current system is a useful prototype for dated choices and searchable conversation history. It has not shown that its summary hierarchy or tool-using reader improves answers enough to justify their cost. See the [evidence ledger](evidence.md) for source details and limits.

## 1. What the current design does well

1. It retains each dated choice instead of overwriting the old one. “Oat latte on March 1; black coffee on March 8” stays an inspectable timeline.
2. It also keeps raw dialogue. The reader can check exact wording when an extracted record is incomplete or unclear.
3. SQL date queries suit questions about order and transitions. A flat embedding search is less reliable for exact dates.

These are real capabilities of the [implemented store and tools](../../memory_v3/tools.py). The system also keeps weekly through lifetime summaries, but those are derived claims rather than ground truth.

## 2. Where it looks too simple

1. A structured row has one date and no direct link to the source utterance. It has no explicit way to represent “planned on Monday, cancelled on Wednesday” as a correction with a validity period. The reader may recover this from raw dialogue, but the structured timeline does not encode it. [Schema](../../memory_v2/store.py#L55)
2. The summarizer asks an LLM to infer cycles from dated choices. It does not independently test whether a proposed rule fits every observation or whether the evidence supports any rule at all. Its selected recurring categories can also stay stale during the life of a summarizer object. [Summarizer](../../memory_v2/summarizer.py#L658)
3. The read agent may make several model/tool calls for a question. On the one tested LoCoMo dialogue, its token F1 was 0.557 on non-adversarial questions versus 0.546 for a one-pass reader of the same store. It used roughly eight answer-time model calls and 30 seconds per question; the one-pass reader used one call and 4.4 seconds. Their prompts differ, so this is a cost signal rather than a clean accuracy ablation. [Local results](../../results/locomo/README.md)
4. The archived synthetic run answered **0 of 3** direct pattern questions correctly. That sample is tiny, but the claimed pattern skill is not yet demonstrated. [Saved run](../../results/synthetic/v3/README.md)

## 3. Could a stronger model just read the whole history?

1. **For these experiments, quite possibly.** The local 60-day transcript has 169,247 characters of raw turn text; the tested LoCoMo dialogue has 58,108. A full-history baseline is feasible and is missing from the saved comparisons. The model must still receive the history on each call or have it retained by the application: context-window capacity is not persistent memory. Repeatedly supplying history has a cost; prefix caching may reduce it, so measure the actual bill and latency.
2. Full context can win on bounded histories. In [Mem0's LoCoMo comparison](https://arxiv.org/pdf/2504.19413), full context scored 72.90% versus Mem0's 66.88% under its judge, but took much longer per answer. In [APEX-MEM's LoCoMo table](https://aclanthology.org/2026.acl-long.749.pdf), the GPT-4o full-context reader slightly beat the GPT-4o APEX reader on non-adversarial questions: 87.52% versus 86.75%.
3. Capacity alone does not guarantee faithful recall. On [LongMemEval-S](https://arxiv.org/pdf/2410.10813), a 2024 GPT-4o scored 0.606 from a roughly 115k-token history versus 0.870 with only the gold evidence sessions. Gold retrieval is an oracle, and newer models have improved simple long-context fact retrieval, as [Google's 2025 study](https://research.google/pubs/retrieval-quality-at-context-limit/) shows. Evolving facts and multi-session reasoning still need direct testing.

**Model-age caveat.** LongMemEval's full-context result used GPT-4o in 2024; the [2025 Mem0 paper](https://arxiv.org/abs/2504.19413) used GPT-4o-mini. [TiMem appeared in July 2026](https://aclanthology.org/2026.findings-acl.1091/) but its main controlled runs still used a 2024 GPT-4o-mini version. [APEX-MEM, also July 2026](https://aclanthology.org/2026.acl-long.749/), tested GPT-5 and Claude 4.5 Sonnet readers, though its GPT-5 results lack a same-model full-context baseline. A [March 2026 preprint](https://arxiv.org/pdf/2603.04814) directly compared full-history GPT-5-mini with a flat Mem0 pipeline: 92.85% versus 57.68% on LoCoMo, and 82.40% versus 49.00% on LongMemEval under its judge. Its memory writer used GPT-5-nano, and its protocol differs from APEX-MEM's. This strongly motivates a current-model full-history baseline for v3; it does not rank all memory architectures.

## 4. Would stronger RAG or memory crush it?

They are credible competitors, especially on broad long-history QA. [APEX-MEM](https://aclanthology.org/2026.acl-long.749.pdf) adds source evidence, entity resolution, temporal validity, graph/SQL queries, and hybrid search. On its LongMemEval setup, its Claude 4.5 Sonnet reader scored 86.2% versus 62.2% for a full-context reader. [TiMem](https://aclanthology.org/2026.findings-acl.1091.pdf) provides evidence that adaptive hierarchy helps; its high-level summaries alone were weaker than the full hierarchy with detailed records. [LongMemEval](https://arxiv.org/pdf/2410.10813) also shows that adding extracted facts to a raw-turn search index improves retrieval and answers without a large new architecture.

There is **no matched result** showing that any published system beats `memory_v3` on the local recurring-choice prediction task. Added complexity can hurt: [Mem0's graph ablation](https://arxiv.org/pdf/2504.19413) improves temporal questions but lowers single-hop and multi-hop scores. Published systems also differ in model, ingestion, retrieval budget, and scoring, so cross-paper rankings are unreliable.

## 5. The decisive experiment

Use the same conversations, reader model, answer prompt, judge, and reported token/cost budget for:

1. Full raw history.
2. Tuned raw-turn RAG, including date filtering and fact-augmented keys.
3. A simple SQL timeline of dated choices.
4. `memory_v3` with and without summaries and with one-pass versus agentic reading.
5. A stronger published memory implementation if runnable under the same protocol.

Score historical recall, transition dates, current state, cycle identification, future prediction, and justified abstention separately. Run more than one user and three pattern questions. This will show where structured memory actually helps and which extra components earn their cost.
