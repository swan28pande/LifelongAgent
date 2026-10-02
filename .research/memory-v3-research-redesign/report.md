# If strong models solve the current benchmark, what should this research test?

**Assessment, 2026-09-29.** I am taking your assumption as given: a strong model with the whole available history can solve the local 60-day task. No direct-prompting baseline was run. [Evidence ledger](evidence.md) · [Sources](sources.md).

## 1. How the research claim changes

1. **Treat the current synthetic set as a pipeline check.** It shows whether the system can ingest dated conversations and answer questions about them. If full-history prompting solves those questions, accuracy on this set does not establish a reason to use the hierarchy or tool-calling reader.
2. **Choose one harder claim.** My recommendation is: *Can an agent maintain a justified, up-to-date model of a user's repeated choices and use it for future decisions as conversations arrive?* “Justified” means it can point to the observations behind a belief and admit when the evidence is insufficient. This is a working research question, not a novelty claim; [HorizonBench](https://arxiv.org/html/2604.17283), [PERMA](https://arxiv.org/html/2603.23231), and [CAPTURE](https://arxiv.org/html/2609.02265) already study related problems.
3. **Measure the whole-system cost.** A memory architecture may still matter if it matches a full-history model's decisions with lower total write, summary, and read cost or latency as histories and query counts grow. A larger context window does not keep a user's history across independent calls by itself. This would be an efficiency result, and the comparison must include cached full history where available.

The key distinction is between an **observation** (“I drank an oat latte today”), a **belief about the user** (“oat latte is their usual morning order”), and an **action** (“order an oat latte tomorrow”). The current store mainly records dated observations and asks the summarizer to infer patterns. Those three things need separate evaluation.

**Concrete case:** The user normally orders oat lattes. On April 10 they say, “Black coffee is my new usual.” On April 11 they accept a free oat latte. On April 12 they ask, “Order my usual for tomorrow.” A useful agent should treat April 10 as a preference change, April 11 as a possible exception, and justify its April 12 choice. If the evidence is genuinely ambiguous, asking is an acceptable decision. The present dataset mostly tests exact deterministic schedules, so it cannot distinguish these behaviors.

## 2. Is the current dataset good enough?

**For a development check, yes. For a claim about lifelong memory, no.** The [generator](../../scripts/generate_eval_dataset.py#L47) creates one person, 60 daily sessions, and three fixed rules: weekly coffee alternation, three-day clothing alternation, and weekday exercise. Each conversation is prompted to mention all three. The [67 questions](../../scripts/generate_eval_dataset.py#L348) include only **three** direct pattern-description items and **nine** future-choice items.

Four problems are especially consequential:

1. **The test can reward shortcuts.** Each sampled exercise recall date is a Wednesday, so all eight answers are morning yoga. All three clothing prediction answers are fitted t-shirt. The “difficulty” tags are also tied to domains, so a domain effect could look like a difficulty effect. [Sampling code](../../scripts/generate_eval_dataset.py#L383).
2. **The gold rule can conflict with the conversation.** The generated user sometimes speaks in generalities that contradict the strict schedule, and one day contains two different drinks. A system can reasonably interpret the dialogue differently from the generator's hidden rule. [Examples and validation limits](evidence.md).
3. **The timing misses the lifelong part.** The evaluator ingests all 60 days, builds summaries, and only then asks the questions. It does not test what the agent believed on day 12, how quickly it revised a belief, or how it handled a correction. [Runner](../../benchmarks/synthetic/run_v3.py#L126).
4. **The score can hide errors.** The judge accepts an answer that lists several options as long as one contains the gold choice. The token-F1 metric can penalize a concise correct prediction because the gold includes a long explanation. [Scorer](../../benchmarks/synthetic/run_v3.py#L39).

The repository's LoCoMo check adds general historical conversation QA, but its local result covers one dialogue and does not directly test repeated-choice forecasting or decision quality. [LoCoMo report](../../results/locomo/README.md).

## 3. A better experiment, step by step

1. **Define the target.** At each checkpoint, ask for the user's current choice or constraint, a future choice in a specified situation, and what the agent should do. Also ask for the supporting conversation turns. Include cases where the right answer is “uncertain.”
2. **Build varied histories.** Use many independent users and rule families. Include sparse observations, one-off exceptions, explicit changes, quiet changes inferred from behavior, reversions, corrections, different contexts, and users with no real pattern. Hold out entire users and schedule types. Add a human-authored or real-data slice to test whether gains survive outside the generator.
3. **Replay time forward.** After each session, give the system only information known up to that point. Probe it before later events are revealed. This tests the live write/read path and prevents future evidence from answering earlier questions.
4. **Score each capability directly.** Use exact values and dates for historical claims; compare predicted probabilities with outcomes for future choices; score action success, needless actions or clarifications, unsupported claims, and stale-belief errors. Report results by user and scenario, alongside total cost and latency. Probabilities matter because a personal routine may be likely without being certain.
5. **Use strong, simple baselines.** Compare the same answer model with full available history, cached full history, recent history, basic raw-dialogue retrieval, an exact dated ledger, and simple “last choice,” weekday, or recency rules. Then remove v3's summaries and multi-step tool use one at a time to see whether each component earns its cost. Use the same checkpoints and score for every method.

For the system's broader **agent** ambition, add tasks in which the remembered state changes a tool call or recommendation. [Mem2ActBench](https://aclanthology.org/2026.acl-long.370/) and [PersonalAlign](https://aclanthology.org/2026.acl-long.1669/) demonstrate why response and action quality are different targets. The current `memory_v3` reader answers questions; an action study would require a task-acting layer as well as a benchmark.

**Decision rule:** If full history or a simple chronological/statistical baseline matches the proposed system on future decisions and total cost, the added memory architecture has no demonstrated advantage for that workload. If v3 wins only on generated schedules and loses on held-out users or real behavior, treat the gain as a generator-specific result.

## 4. The related literature is much larger

The papers previously discussed were close **architecture** matches, including [APEX-MEM](https://aclanthology.org/2026.acl-long.749/), [Chronos](https://arxiv.org/html/2603.16862), and [TiMem](https://aclanthology.org/2026.findings-acl.1091/). Other lines ask different questions:

| Direction | Examples | What it adds to this project |
|---|---|---|
| Evolving and implicit user preferences | [HorizonBench](https://arxiv.org/html/2604.17283), [PERMA](https://arxiv.org/html/2603.23231), [PersonaMem-v2](https://arxiv.org/html/2512.06688) | Test state changes and cues beyond explicit daily labels. HorizonBench's best tested full-history model scored 52.8% on its filtered five-choice items; that result is specific to its difficult synthetic setup. |
| Memory maintenance and trust | [MemOps](https://arxiv.org/html/2607.12893), [Memora](https://arxiv.org/html/2604.20006), [CAPTURE](https://arxiv.org/html/2609.02265) | Test updates, forgetting, obsolete facts, and whether an apparent preference change should be trusted. |
| Using memory in action | [Mem2ActBench](https://aclanthology.org/2026.acl-long.370/), [PersonalAlign](https://aclanthology.org/2026.acl-long.1669/) | Check tool arguments, vague requests, and routine-based assistance. |
| Real behavior and long histories | [RealTalk](https://arxiv.org/html/2502.13270), [MemoryCD](https://arxiv.org/html/2603.25973), [LongMemEval-V2](https://arxiv.org/html/2605.12493) | Test realism or scale beyond one generated user. These datasets serve different tasks and should not be ranked by headline score. |
| Prediction and feedback | [LATTE](https://arxiv.org/html/2605.26612), [contextual bandits](https://arxiv.org/abs/1003.0146) | Connect memory to forecasting choices and learning from the result, a problem studied beyond LLM memory research. |

The research opportunity is therefore **specific and conditional**: show that a method makes better or cheaper forward-time personalized decisions than strong full-context and simple temporal baselines, and identify which memory operation caused the gain. The current dataset cannot establish that claim yet.
