# Evidence ledger

This file records source observations separately from interpretation. Line references identify the current checkout; saved result files describe earlier runs where noted.

## Architecture

1. **One tool-calling reader.** memory_v3/agent.py:100–125 constructs one MemoryStore, one IngestionPipeline, one Summarizer, six read tools, and one LangChain create_agent reader. The write pipeline is a fixed sequence at memory_v3/ingest.py:122–138. **Interpretation:** “agentic” refers to the read path, not a multi-agent or agentic-write architecture.
2. **Three LLM instances.** memory_v3/agent.py:112–121 creates extraction, chat, and summary model wrappers with temperatures 0, 0.3, and 0.2. **Qualification:** These may point to the same named model; separate wrappers do not imply separately trained models.
3. **Batch and live paths.** memory_v3/agent.py:129–176 ingests dated sessions; :180–202 asks stateless questions; :206–264 replays process-local pending turns and flushes them into ingestion. **Interpretation:** batch QA does not test the live buffer.

## Memory and retrieval

4. **Three record types.** memory_v3/prompts.py:20–90 defines dated choices, stable attributes, and dated events. memory_v3/ingest.py:164–177 parses them. **Qualification:** extraction quality is model-dependent; the offline tests script responses.
5. **Structured write sequence.** memory_v3/ingest.py:122–138,218–334 normalizes near-matching values, drops exact same-day repeats, then writes rows. :338–364 indexes five-turn raw chunks even when extraction is empty. **Interpretation:** structured memory and raw evidence have different retry/idempotency behavior.
6. **Storage surfaces.** memory_v2/store.py:35–95 defines SQLite memories, conversations, and summary metadata; :231–260 writes raw chunks to SQL and FAISS; :273–306 stores summaries in FAISS. **Qualification:** structured rows contain one effective date, not a separate transcript-observation date or a direct raw-chunk provenance link.
7. **Summary hierarchy.** memory_v2/summarizer.py:206–304 builds per-speaker week → month → year → lifetime summaries and can rebuild an affected chain using source hashes. :308–625 shows multiple LLM calls for domain analyses and narratives. **Interpretation:** the documented one-call cost applies only to extraction per ingestion call.
8. **Reader capabilities.** memory_v3/tools.py:39–232 exposes entity discovery, SQL timeline, semantic raw and summary search, exact-date raw lookup, and direct summary lookup. memory_v3/prompts.py:95–130 guides selection. **Qualification:** memory_v3/agent.py:184–202 records tool names and arguments, not returned evidence, in chat_with_trace.

## Code/document gaps

9. **Lifetime summary injection.** memory_v2/summarizer.py:604–619 says the blueprint will be injected into future conversations; memory_v3/agent.py:118–122 and :287–293 supply only CHAT_SYSTEM and messages, with summaries reachable via tools. **Conclusion from code:** automatic injection is absent.
10. **Raw weekly source use.** memory_v2/summarizer.py:313–386 retrieves raw conversations but uses them only to require a nonempty week; its generation prompts use structured preference, fact, and event rows. **Conclusion from code:** weekly prose is not directly grounded in fetched raw text.
11. **Potential boundary failures.** memory_v3/agent.py:258–264 clears the live buffer before ingest returns; memory_v3/ingest.py:338–345 has no raw-chunk duplicate check. memory_v2/summarizer.py:658–701 caches discovered domains per instance. memory_v2/store.py:199–220 retrieves a preference sequence without a type predicate. **Qualification:** these are code-level risks, not measured failure rates.
12. **Date distinction.** memory_v3/prompts.py:77–83 resolves relative event dates; memory_v3/ingest.py:318–330 stores the extracted effective date, while :338–344 indexes raw transcript chunks under the ingestion date. memory_v3/agent.py:145–148 updates summaries using the ingestion date. **Inference:** an event dated into another period may not dirty that event period's summary.
13. **Speaker attribution.** memory_v3/prompts.py:85–90 does not request a speaker field. memory_v3/ingest.py:164–177 accepts one only if returned and :318–326 otherwise uses the ingest-call speaker. **Inference:** dialogue turn labels alone do not guarantee correct per-person structured records in a multi-speaker conversation.

## Empirical evidence

14. **Synthetic design.** scripts/generate_eval_dataset.py:47–76 and :348–443 generate three deterministic daily choice schedules and 67 questions for one user. The current runner benchmarks after 60 days of ingestion at benchmarks/synthetic/run_v3.py:123–176. **Interpretation:** this is a controlled temporal task, not evidence of cross-user generalization or live adaptation.
15. **Synthetic archived result.** results/synthetic/v3/qa_results.json:1–36 records 0.143 token F1, 0.478 LLM-judge accuracy, and 0/3 pattern-identification success. It names gemini-3.1-flash-lite; current runner defaults gemini-3.5-flash at benchmarks/synthetic/run_v3.py:102, and the archived memory rows lack the current type field. **Qualification:** historical run, not current-revision measurement.
16. **Synthetic baseline mismatch.** evaluation/evaluate_synthetic.py:575–605 defaults to 30 days while its saved combined overview answers 67 questions; current v3 defaults to 60 days. Current F1 at benchmarks/synthetic/run_v3.py:68–76 uses token sets; legacy F1 at evaluation/evaluate_synthetic.py:96–122 uses counts, stemming, and date normalization. **Interpretation:** direct score ranking is confounded by evidence window and metric.
17. **LoCoMo scope and read ablation.** results/locomo/README.md:3–30 reports one conversation, 199 questions. benchmarks/locomo/README.md:19–45 and runners show v3 one-pass and agentic read the same store, but different answer prompts and call counts; v3 one-pass and mem0 use the same answer prompt with unequal context budgets. **Interpretation:** neither comparison isolates one architecture variable completely.
18. **LoCoMo scorer mismatch.** benchmarks/locomo/common.py:62–75 accepts “don't know” on adversarial items; baselines/locomo/task_eval/evaluation.py:216–221 accepts only “no information available” or “not mentioned” and :203–204 truncates category-3 answers at a semicolon. **Conclusion from code:** the local paper-exact label is inaccurate.
19. **LoCoMo artifact drift.** archived results/locomo/v3_agentic/locomo_results.json tool calls include search_preferences; current memory_v3/tools.py:68–76 names search_memories. **Conclusion from artifacts and code:** the saved run predates the current tool API.

## Prior work: primary-source observations

20. [APEX-MEM](https://aclanthology.org/2026.acl-long.749.pdf), abstract and §§3–4: append-only temporal graph facts, source evidence, and multi-tool retrieval including SQL graph traversal. **Interpretation:** broad “temporal structured memory plus agentic read” is prior art; its graph and validity model differ from v3's flat choice ledger.
21. [Chronos](https://arxiv.org/html/2603.16862), §§3.1–3.4: event and raw-turn calendars, datetime ranges, and a tool-calling reader. **Interpretation:** dual raw/structured temporal retrieval is prior art; its event-focused structure differs from v3's typed choice rows and summary hierarchy.
22. [TiMem](https://aclanthology.org/2026.findings-acl.1091.pdf), §3: segment/session/day/week/profile levels with consolidation and complexity-aware recall. **Interpretation:** hierarchical temporal summarization is prior art; reported scores use another protocol and should not rank directly against local runs.
