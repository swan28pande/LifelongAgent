# Benchmark breakdowns (vs. generator_v2)

Reference: the researcher's generator_v2 ideas are
(1) 2 years of daily assistant-user conversations from a code-computed ground-truth world state;
(2) rule-based preferences (cycles, day-of-week, conditional) with regime shifts, explicit/implicit/unstated causes, temporary reverting shifts, one-off exceptions, distractors;
(3) evolving facts (set/add/remove) and event chains (planned -> happened/cancelled);
(4) 5-level difficulty ladder (session frequency, mention rate, paraphrase, lags);
(5) world-state-derived QA (recall, pattern, prediction, change detection, why, exception vs shift, reversion, facts over time, durations, distractor probes, other-person disambiguation, abstention).

All statements below come from the paper text; "not specified in paper" marks gaps.

---

## LoCoMo (ACL, 2024)

**Fundamental ideas**
1. Two LLM agents (gpt-3.5-turbo) converse with each other (human-human style, not user-assistant). Each has a persona expanded from an MSC persona seed and a temporal event graph (text-davinci-003) of up to 25 causally linked events over 6-12 months; each session is conditioned on events that fall between the previous and current session dates. Agents also share and react to images.
2. Human annotators then edit the dialogues for long-range consistency and alignment with the event graph (about 15% of turns edited, about 19% of images removed or replaced).
3. Scale: 10 conversations, about 27.2 sessions, about 588 turns and about 16.6K tokens each on average (up to 32 sessions).
4. Ground truth: the event graph for event summarization; for QA, 1,986 questions annotated with evidence turn IDs. Who wrote the QA pairs is not specified in the paper.
5. QA types: single-hop, multi-hop, temporal, open-domain/commonsense, adversarial (unanswerable). Other tasks: event summarization and multimodal dialogue generation.
6. Metrics: token F1 (partial match) for QA, recall@k for RAG retrieval, FactScore-based F1 for summarization, MMRelevance for dialogue generation.

**Tests changing preferences/facts?** Only indirectly: events unfold over time, but there is no explicit update/change-detection question category.

**Overlap with generator_v2:** Shares an event timeline driving conversations (partial 1, 3) and recall/temporal/abstention QA (part of 5). It lacks a code-level world state, rule-based preferences, regime shifts, reversion and a difficulty ladder. It has multimodality, human editing and a two-speaker setup (useful for other-person disambiguation, since the paper notes misattribution to the wrong speaker).

---

## LongMemEval (ICLR, 2025)

**Fundamental ideas**
1. Question-first construction. An ontology of 164 user attributes (five categories) seeds LLM-written (Llama 3 70B) background paragraphs. The LLM proposes QA pairs, then human experts filter and rewrite every question and break each answer into evidence statements, with optional timestamps.
2. Each evidence statement is embedded indirectly in a task-oriented user-assistant session via LLM self-chat, then human-edited. Evidence sessions are inserted among filler sessions (self-chat on non-conflicting attributes plus ShareGPT/UltraChat) with plausible timestamps.
3. Scale: 500 questions. LongMemEval_S is about 115K tokens per question; LongMemEval_M is 500 sessions and about 1.5M tokens. The history length can be configured freely. The calendar time span is not specified in the paper.
4. Five abilities and seven question types: single-session-user, single-session-assistant, single-session-preference, multi-session, knowledge-update, temporal-reasoning, and abstention (30 false-premise questions). Most questions need evidence from multiple sessions (up to six).
5. Metric: QA accuracy from a prompt-engineered GPT-4o judge (over 97% agreement with humans). Recall@k and NDCG@k use the annotated evidence locations.

**Tests changing preferences/facts?** Yes, for facts: the knowledge-update type tests changes in the user's life states. Preference evolution is not a separate category.

**Overlap with generator_v2:** Shares user-assistant sessions (1), updated facts (3), recall, temporal and abstention QA (5), and configurable length as a partial difficulty axis (4). It lacks a daily 2-year timeline, a code-computed world state, preference rules, shifts, reversion and "why" questions. The histories are stitched haystacks rather than one coherent life.

---

## PrefEval (ICLR, 2025)

**Fundamental ideas**
1. 1,000 unique preference-query pairs across 20 topics (travel, shopping, entertainment, etc.). Each pair comes in three forms (explicit; implicit choice-based dialogue; implicit persona-driven dialogue over 4-8 turns), giving 3,000 pairs. They were manually curated with help from GPT-4 and Claude 3/3.5 Sonnet.
2. Each query is built so that a generic, non-personalized answer would violate the stated preference.
3. Long context comes from interleaving real LMSYS-Chat-1M multi-session turns between the preference and the query, up to 100K tokens. No calendar time span is given.
4. Ground truth is the single stated preference. Tasks are generation and classification (4-option MCQ).
5. Metric: Claude 3 Sonnet as judge, running four binary checks that aggregate into error types (preference-unaware violation, hallucination, inconsistency, unhelpful). Preference-following accuracy means no error. Human check: 5% error on 200 samples. The classification task reports MCQ accuracy.

**Tests changing preferences/facts?** Mostly no. There is only an analysis experiment that inserts a conflicting preference earlier than the original (5 topics) and tests adherence to the later original.

**Overlap with generator_v2:** Shares the preference focus and explicit vs implicit expression (2, partly 4 via paraphrase/implicitness). It lacks temporal structure, rules, shifts, causes, facts/events and diverse QA types. It evaluates proactive preference following in generation, which generator_v2's QA does not.

---

## PersonaMem (COLM, 2025)

**Fundamental ideas**
1. GPT-4o pipeline: a PersonaHub persona is expanded with demographics, then a general personal history is generated (the prompt asks for 10 events within ten years, with timestamps). Topic-specific histories follow, holding events, preferences, preference updates and the reasons for those updates.
2. Each history segment is expanded into a user-chatbot session (15-30 turns). The LLM cites the source event and self-reflects to catch missed events. Sessions are concatenated in topological order that keeps causality within each topic, and short preference-free filler interactions are inserted.
3. Scale: 20 personas, over 180 histories of 10, 20 or 60 sessions (about 32K, 128K or 1M tokens), 15 topics, about 6K queries.
4. Ground truth comes from the structured histories (the user's profile or preference at a given time). Questions answerable without context are removed. A human study of 90 pairs was run.
5. Seven in-situ query types: recall facts, suggest new ideas, acknowledge latest preference, track full preference evolution, revisit reasons for updates, preference-aligned recommendations, generalize to new scenarios.
6. Metric: 4-option MCQ accuracy (distractors are outdated or irrelevant). There is also a generative log-probability setting. No LLM judge is used.

**Tests changing preferences/facts?** Yes. Preferences update with explicit reasons, and questions test the latest value, the full evolution and the reason for the change.

**Overlap with generator_v2:** Shares a structured history driving conversations (partial 1), preference change with causes (2), and change-detection and "why" questions (5). It lacks code-computed rules or cycles, temporary reverting shifts, one-off exceptions, a difficulty ladder, and daily granularity. Its ground truth is an LLM-generated history rather than code.

---

## HorizonBench (arXiv, 2026)

**Fundamental ideas**
1. A state-first generator builds conversations from a structured mental state graph (persona, social graph, life events, preferences with typed dependency edges). Every preference change records its triggering event as provenance.
2. Events are sampled based on the agent state and event history, with category weights from empirical chatbot usage. With p=0.15, a life event changes 2-5 preferences at once, propagating along dependency edges. The event is mentioned without restating the new values, so evolved preferences are implicit.
3. Expression tracking: preferences not expressed for more than 30 days become eligible for temporal-recall test turns. Explicit and implicit variants rewrite only the preference-expressing turns.
4. Scale: 4,245 items from 360 users, with 6-month histories averaging about 4,300 turns and about 163K tokens.
5. Items are 5-option MCQs of counterfactual assistant responses, with the pre-evolution value as a hard negative. A 5-LLM consensus filter keeps only items unanswerable without the history. Controlled dimensions are evolution status, explicitness and context length.
6. Metric: accuracy (chance is 20%) plus the rate of choosing the pre-evolution distractor. No LLM judge is used for scoring (the format is MCQ).

**Tests changing preferences/facts?** Yes, this is its core focus: life-event-driven preference evolution with provenance, and diagnosing retrieval failure versus belief-update failure.

**Overlap with generator_v2:** The closest in spirit. It shares a structured ground-truth state (1), preference shifts with implicit causes (2), and controlled difficulty axes (4). It lacks periodic or conditional preference rules, reverting shifts, exceptions, fact set/add/remove, event chains and open-ended QA types (only "current preference" MCQ). Its span is 6 months rather than 2 years.

---

## BEAM (ICLR, 2026)

**Fundamental ideas**
1. A plan-driven LLM generator creates user-assistant conversations. The seed includes domain, title, theme, subtopics, 15-20 narratives (evolving aspects), an MBTI-based profile, a relationship graph and a timeline. Plans are recursively split into sub-plans of time-anchored bullets, which are then augmented with bullets for contradiction, update and instruction-following.
2. User turns come from bullet batches (LLaMA-3.3-70B). Assistant turns are role-played, with question-detection and follow-up modules. For 10M tokens, ten interlocking plans are built by sequential expansion or hierarchical decomposition.
3. Scale: 100 conversations across 19 domains (coding, math, health, therapy and others), from 128K to 10M tokens (about 144 to 10,435 user messages). The concrete calendar span is not specified in the paper.
4. Ground truth: GPT-4.1-mini proposes probes from plan bullets linked to turn IDs, and humans select valid ones. The final set is 2,000 questions (20 per conversation, 2 per ability).
5. Ten abilities: abstention, contradiction resolution, event ordering, information extraction, instruction following, information update, multi-hop, preference following, summarization, temporal reasoning.
6. Metric: nugget-based LLM-judge scoring (0 / 0.5 / 1 per nugget). Event ordering uses Kendall tau-b.

**Tests changing preferences/facts?** Yes. Information-update, contradiction-resolution and "evolving preferences" (preference following) probes come from plan bullets encoding an initial fact and its revision.

**Overlap with generator_v2:** Shares planned timelines driving conversations (partial 1), fact updates (3), and temporal, ordering and abstention questions (5). It lacks code-computed ground truth, rule-based preferences, reverting shifts and exceptions, and a difficulty ladder beyond length. It adds instruction-following, contradiction and non-personal domains (coding and math).

---

## MemoryAgentBench (ICLR, 2026)

**Fundamental ideas**
1. It mostly reformats existing long-context datasets into incremental chunks fed as user messages ("please memorize..."), plus two new datasets. It is not a single simulated user life.
2. Four competencies: Accurate Retrieval (SH/MH-Doc QA, LongMemEval(S*) with 5 dialogues of about 355K tokens and 300 questions, EventQA on novels), Test-Time Learning (5 intent/question classification datasets, movie recommendation), Long-Range Understanding (InfiniteBench summarization, DetectiveQA), and Selective Forgetting (FactConsolidation).
3. FactConsolidation: MQuAKE counterfactual edit pairs, with the rewritten fact placed after the original, concatenated to 6K-262K tokens. It has single-hop and multi-hop questions. Agents are told that newer facts have larger serial numbers and that the newest wins.
4. Context lengths average 103K to 1.44M tokens per dataset. There is no calendar time span, and "time" means chunk order.
5. Metrics: accuracy / SubEM, Recall@5 (recommendation), F1 (summarization). A GPT-4o judge is used for LongMemEval and summarization.

**Tests changing preferences/facts?** Facts only, via FactConsolidation's synthetic overwrite pairs. Preferences are not tested.

**Overlap with generator_v2:** Shares fact overwriting (3), in a synthetic counterfactual form. It lacks a coherent user timeline, preferences, causes, reversion and a difficulty ladder. It is an agent-competency harness (retrieval, test-time learning, long-range understanding, forgetting) rather than a lifelog benchmark.

---

## LoCoMo-Plus (arXiv, 2026)

**Fundamental ideas**
1. It extends LoCoMo with "Level-2 cognitive memory": latent constraints (causal, state, goal, value) that must shape later responses under cue-trigger semantic disconnect.
2. Construction: LLMs (gpt-5-nano, gpt-4o, gpt-4.1, gemini-2.5-flash/pro) generate short implicit cue dialogues. Humans keep the memory-worthy ones. An LLM writes an underspecified trigger query plus a time-gap indicator. BM25 and MPNet filtering removes pairs with high overlap, and a final human check confirms each pair elicits memory.
3. Validated cue/trigger pairs are inserted into LoCoMo conversations, with the trigger placed after a gap consistent with t (examples say "weeks later" or "months later"). The number of cognitive instances is not stated in the text (it appears only in a figure). The paper describes them as "intentionally limited".
4. Ground truth for cognitive items is a valid response space (responses consistent with the constraint), not a single answer. The original LoCoMo factual categories are kept.
5. Evaluation: no task-type disclosure in prompts. An LLM judge (gemini-2.5-flash in the agreement study, with GPT-4o as an alternative) gives 3-level labels for factual and commonsense items and binary labels for temporal, adversarial and cognitive items. Agreement with 2 human annotators and stability across judges are reported.

**Tests changing preferences/facts?** Partly. Some causal cues describe a habit change triggered by an event, but there is no systematic tracking of updates or reversions.

**Overlap with generator_v2:** Shares implicit or unstated causes and lags (2, 4) and "why"-style causal linking (5). It lacks a code-computed world state, rule-based preferences, fact evolution and reversion. It contributes a response-consistency evaluation for implicit constraints, which generator_v2's QA does not cover.
