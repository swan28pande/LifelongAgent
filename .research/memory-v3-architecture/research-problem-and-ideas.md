# Research problem and ideas behind memory_v3

## The problem

A lifelong assistant must remember **how a person changes**. Imagine these daily choices in March 2026:

~~~text
March 1–7     oat milk latte
March 8–14    black coffee
March 15–21   oat milk latte
~~~

If memory keeps only “current coffee: oat milk latte,” it cannot answer what the person chose on March 10 or when the first switch happened. If it keeps only undated statements, it cannot reliably recover the sequence. Seeing the latte return also changes the interpretation: the black-coffee week may be part of a repeating routine rather than a permanent change. The project's [stated goal](/home/cloaked/projects/LifelongAgent/PROJECT_PLAN.md) is to preserve both the history and any rule supported by it.

The concrete research question is: **Can an assistant turn months of conversation into faithful dated observations, use them to answer past and current questions, and identify genuine recurring patterns without losing the original evidence?**

Here a *preference* means a **choice made for a date** from a recurring category. A deliberate rest day counts as a choice; an unmentioned day supplies no observation. The system also records relatively stable *facts* and one-time *events*. ([Extraction definitions](/home/cloaked/projects/LifelongAgent/memory_v3/prompts.py:20))

## The ideas used to solve it

### 1. Keep a dated history of choices

For each ingested conversation, an LLM extracts short records such as “coffee order → black coffee → March 8.” The system keeps a repeated choice on each new date; it removes only exact same-day repeats. This gives SQL a real timeline to search and count. Existing entity names and values are shown to the extractor, and similar value spellings are normalized so “oat latte” and “oat milk latte” do not create a false transition. ([Write pipeline](/home/cloaked/projects/LifelongAgent/memory_v3/ingest.py:122), [extraction](/home/cloaked/projects/LifelongAgent/memory_v3/ingest.py:142), [deduplication](/home/cloaked/projects/LifelongAgent/memory_v3/ingest.py:275))

### 2. Preserve the original conversation alongside the records

Extraction can omit detail or make a mistake. The system therefore stores the conversation in five-turn chunks as well: in SQLite for lookup by date and in FAISS for semantic search. A reader can inspect what was actually said around March 8 to check the extracted row. ([Chunking](/home/cloaked/projects/LifelongAgent/memory_v3/ingest.py:338), [storage](/home/cloaked/projects/LifelongAgent/memory_v2/store.py:231))

### 3. Build summaries at several time scales

The summarizer creates a separate summary **for each speaker and period**. It first chooses which preference categories look recurring, then asks an LLM about each category's changes. Facts and events are summarized alongside them. These are generated interpretations; the dated SQL rows remain available as evidence. See [domain selection](/home/cloaked/projects/LifelongAgent/memory_v2/summarizer.py:658) and the [summary order](/home/cloaked/projects/LifelongAgent/memory_v2/summarizer.py:206).

#### How categories are created and chosen

A **category** is a kind of choice that can have different answers on different days. The code calls it an **entity**; it calls the answer **content**:

| Entity (category) | Date | Content (choice) |
| --- | --- | --- |
| Coffee order | March 1 | Oat milk latte |
| Coffee order | March 8 | Black coffee |
| Coffee order | March 15 | Oat milk latte |

1. **The extractor creates the name.** While ingesting a conversation, an LLM might name this entity “coffee order.” The code defines the broad record types (preference, fact, event), but no fixed list of preference entities. On later days, the extractor sees existing names and is asked to reuse them. See the [extraction rules](/home/cloaked/projects/LifelongAgent/memory_v3/prompts.py:20) and [extractor input](/home/cloaked/projects/LifelongAgent/memory_v3/ingest.py:142).
2. **The test dataset does not supply that name to the agent.** The synthetic generator defines coffee, clothing, and exercise schedules to make conversations and expected answers. The v3 benchmark passes the generated conversations to the agent, not those schedules. The extractor could call the coffee category “coffee order” even though the generator calls it “coffee.” See the [synthetic schedules](/home/cloaked/projects/LifelongAgent/scripts/generate_eval_dataset.py:70) and [benchmark input](/home/cloaked/projects/LifelongAgent/benchmarks/synthetic/run_v3.py:115).
3. **The summarizer chooses what to analyze.** During a weekly build, it counts stored preference rows by entity. Entities with at least two rows become candidates. It sends each candidate's name, count, and a few example choices to an LLM, which is asked to exclude categories with fewer than three observations or that are not genuine recurring choices. This selection step does **not** show the LLM the dates; later pattern analysis uses dated transitions. See [weekly entry](/home/cloaked/projects/LifelongAgent/memory_v2/summarizer.py:308) and [selection code](/home/cloaked/projects/LifelongAgent/memory_v2/summarizer.py:658).
4. **It keeps the selected list.** Only selected entities get pattern analysis; all dated rows remain searchable in SQLite. The list is cached for each speaker in the summarizer object. If “exercise type” becomes recurring after “coffee order” was selected, the same object keeps its old list. A new agent using the same stored data creates a new summarizer and can select both. See the [cache](/home/cloaked/projects/LifelongAgent/memory_v2/summarizer.py:664) and [agent construction](/home/cloaked/projects/LifelongAgent/memory_v3/agent.py:109).

Here is the path for the coffee example above. Assume there is one choice recorded each day, plus the fact “city: Boston” on March 5 and the event “ran a 10K” on March 12. The example summaries below are **illustrative**, not output from an actual run.

| Step | What the model receives | Example of what gets saved |
| --- | --- | --- |
| **1. Week** | That week's coffee rows, reduced to changes with dates and durations; that week's fact and event rows. | **2026-W10** (March 2–8): “Latte from March 2; changed to black coffee March 8. A weekly switch is possible.” Also includes “city: Boston.” |
| **2. Month** | Weekly summaries touching March **plus March's coffee change sequence** and March's fact/event rows. | **2026-03**: “Latte → black coffee on March 8 → latte on March 15; a seven-day alternation is a candidate.” Keeps Boston and the March 12 event. |
| **3. Year** | Monthly summaries for 2026, plus the year's fact/event rows. | **2026**: compares the March candidate with other months. With March alone, the evidence is too short to establish a yearly rule. |
| **4. Lifetime** | Yearly summaries. | A profile combining longer-term preference claims, facts, and dated events. |

~~~text
Conversation --> dated choices, facts, events in SQLite
             --> raw chunks in SQLite + FAISS

One week's dated rows ------------------> week:2026-W10:user
Week summaries + March's dated rows ----> month:2026-03:user
Month summaries + year's facts/events --> year:2026:user
Year summaries -------------------------> lifetime:user

~~~

Each summary is one FAISS document. The saved text for the example week would have roughly this shape (abridged):

~~~text
[week:2026-W10:user] Weekly Summary for 2026-W10
{'patterns': {'coffee order': {
    'mathematical_sequence': 'oat milk latte March 2 -> black coffee March 8',
    'potential_cycle': 'possible weekly switch',
    'reasoning': '...'}},
 'facts': ['city: Boston'], 'events': [], 'narrative': '...'}
~~~

1. **Week:** The code compresses repeated coffee values into a dated transition sequence, then asks the LLM to log changes and *speculate* about a cycle. A second LLM call writes a short narrative from structured facts and events. Raw conversation is checked to see whether the week has any conversation, but its text is **not passed to these summary prompts**. See [weekly construction](/home/cloaked/projects/LifelongAgent/memory_v2/summarizer.py:308) and [transition formatting](/home/cloaked/projects/LifelongAgent/memory_v2/summarizer.py:55).
2. **Month:** The code supplies weekly summaries and the month's own dated preference transitions to the LLM. It asks for a repeating rule for each category. It separately merges the month's facts and orders its events. Weeks at the edge of March can contain days from February or April. See [monthly construction](/home/cloaked/projects/LifelongAgent/memory_v2/summarizer.py:391).
3. **Year:** The LLM compares monthly summaries and is asked to confirm each proposed rule. The year's facts and events are gathered again from SQL. It does **not** directly re-read the full daily preference timeline here. See [yearly construction](/home/cloaked/projects/LifelongAgent/memory_v2/summarizer.py:487).
4. **Lifetime:** The LLM combines yearly summaries into one profile. It does not directly re-read daily rows here. See [lifetime construction](/home/cloaked/projects/LifelongAgent/memory_v2/summarizer.py:582).

The saved names are **week:2026-W10:user**, **month:2026-03:user**, **year:2026:user**, and **lifetime:user**. Each summary is generated text in FAISS; rebuilding it replaces the earlier version. See [summary storage](/home/cloaked/projects/LifelongAgent/memory_v2/store.py:273).

An incremental update uses a **hash**: a short fingerprint of a summary's inputs, saved in SQLite. When the caller requests a summary update after ingesting a day:

1. Make a fingerprint from this week's conversations and dated rows. If it matches the saved fingerprint, stop.
2. If they differ, rebuild the week. Then compare the month using its weekly summaries. Rebuild the month only if they changed.
3. Compare the year using its monthly summaries. If they changed, rebuild the year and then the lifetime profile. The lifetime profile has no separate hash.

For example, adding a March 8 coffee choice changes that week's inputs. Its summary is rebuilt; an update reaches March's summary only if the weekly summary text changes. A hash detects changed inputs, **not** whether a summary or proposed rule is correct. A “confirmed” rule is still an LLM judgment: there is no separate cycle checker. See [incremental updates](/home/cloaked/projects/LifelongAgent/memory_v2/summarizer.py:228) and [stored hashes](/home/cloaked/projects/LifelongAgent/memory_v2/store.py:329).

### 4. Give the answering model a choice of evidence

**Who controls what?** The caller decides when to ingest a day or build summaries. The write pipeline then follows fixed code steps. For each question, the **reader LLM** chooses from six read-only tools. The system prompt and tool descriptions guide it, but the code does not force a fixed tool order. Tool results return to the reader, which can call another tool or answer. See [agent construction](/home/cloaked/projects/LifelongAgent/memory_v3/agent.py:100), [reader guidance](/home/cloaked/projects/LifelongAgent/memory_v3/prompts.py:95), and [tool definitions](/home/cloaked/projects/LifelongAgent/memory_v3/tools.py:39).

~~~text
WRITE — caller chooses when
ingest(day, conversation)
  -> extraction LLM: preferences, facts, events
  -> normalize similar values -> remove exact same-day repeats
  -> save dated rows in SQLite
  -> save raw conversation chunks in SQLite + FAISS
  -> optional summary update -> save week/month/year/lifetime in FAISS

READ — model chooses what to look up
chat(question) -> reader LLM -> chosen read tool -> store
                    ^                 |
                    └── tool result ──┘  (repeat if needed)
                 -> answer
~~~

The six tools are:

| Tool | What it is asked to return | Example use |
| --- | --- | --- |
| [list_entities](/home/cloaked/projects/LifelongAgent/memory_v3/tools.py:42) | Entity names and speakers actually present in SQLite. Its description asks the model to call it first. | Find that “coffee order” is the stored name before filtering. |
| [search_memories](/home/cloaked/projects/LifelongAgent/memory_v3/tools.py:68) | Dated preference, fact, or event rows, optionally filtered by entity, speaker, type, and date range. Results are chronological. | Compare coffee choices on March 7, 8, and 15. |
| [semantic_search_conversations](/home/cloaked/projects/LifelongAgent/memory_v3/tools.py:135) | Raw conversation chunks whose **meaning** matches a query. | Find discussions about why the user changed coffee. It is not reliable for a named date. |
| [read_conversations_on](/home/cloaked/projects/LifelongAgent/memory_v3/tools.py:152) | Raw conversation chunks from one date, optionally up to seven days on either side. | Check exactly what was said on March 8. |
| [get_summary](/home/cloaked/projects/LifelongAgent/memory_v3/tools.py:188) | One known week's, month's, year's, or lifetime summary by its period ID. | Read the March 2026 summary. |
| [semantic_search_summaries](/home/cloaked/projects/LifelongAgent/memory_v3/tools.py:205) | Summaries across all levels that match a topic by meaning. | Find a broad coffee habit when the period is unknown. |

For **“When did I first switch to black coffee?”**, one possible reader trace is:

1. Call **list_entities()**. The tool might return “coffee order” as an available entity.
2. Call **search_memories(entity="coffee order", type="preference")**. The tool returns dated rows in order.
3. If the wording matters, call **read_conversations_on(date="2026-03-08")** to inspect the original exchange.
4. Compare the rows and answer **March 8, 2026**. This is a possible route; the model may choose different calls in a real run.

An illustrative result from step 2 shows what the reader actually receives:

~~~text
id=42 [2026-03-07] [preference] (coffee order) oat milk latte — user
id=43 [2026-03-08] [preference] (coffee order) black coffee — user
~~~

For **“What is my coffee pattern?”**, the reader may start with **semantic_search_summaries(query="coffee choices")** or **get_summary(level="month", identifier="2026-03")**, then check the dated rows with **search_memories** before stating a precise rule. For **“What did we discuss about coffee?”**, **semantic_search_conversations** is the direct search. The lifetime summary is available through tools; this code does **not** automatically place it in every chat prompt.

There are two ways to reach the same stores:

1. **Batch evaluation:** **ingest_range()** writes days in date order without summary updates by default. The caller can then run **build_summaries()** and ask questions with **chat()**. See the [batch path](/home/cloaked/projects/LifelongAgent/memory_v3/agent.py:150) and [summary build](/home/cloaked/projects/LifelongAgent/memory_v3/agent.py:273).
2. **Live conversation:** **converse()** gives the reader today's still-unwritten turns as message context. **flush()** later ingests those turns and updates summaries by default. See the [live path](/home/cloaked/projects/LifelongAgent/memory_v3/agent.py:206).

### 5. Separate today's conversation from older memory

During live use, today's turns stay in the reader's message context. At the end of the day, flush sends them through ingestion and updates summaries. Thus a choice mentioned at breakfast can affect a dinner answer before it has been written to persistent memory. ([Live path](/home/cloaked/projects/LifelongAgent/memory_v3/agent.py:206))

## What a successful answer would look like

For “When did I first switch from oat milk latte to black coffee?”, the agent could list the actual stored entity names, query that entity's dated preference rows, and answer **March 8** after comparing consecutive values. It could read the March 7–8 raw conversation to verify the wording. This is an *available route*; actual tool selection varies. ([SQL tool](/home/cloaked/projects/LifelongAgent/memory_v3/tools.py:68), [date tool](/home/cloaked/projects/LifelongAgent/memory_v3/tools.py:153))

Success on the broader research problem requires accurate extraction across users and dates, correct handling of breaks and reversions, reliable identification of a rule only when the data supports one, useful answers to questions about future dates, and sensible abstention when evidence is insufficient. Write and answer cost matter too: “one LLM call per day” describes one **extraction call per ingestion invocation**; summary generation adds calls. ([Pipeline](/home/cloaked/projects/LifelongAgent/memory_v3/ingest.py:142), [summarizer](/home/cloaked/projects/LifelongAgent/memory_v2/summarizer.py:206))

## How it is evaluated

Both benchmarks test **offline question answering**: ingest dated conversations, build memory and summaries, ask questions, then compare answers with expected answers. The expected answers are used for scoring, not given to the memory agent.

### Synthetic: changing preferences

1. **Task:** One generated person, Jordan, has 60 daily conversations. Hidden schedules make coffee switch every seven days, clothing every three days, and exercise follow a day-of-week rule. See the [generator](/home/cloaked/projects/LifelongAgent/scripts/generate_eval_dataset.py:47).
2. **Run:** The benchmark ingests all 60 days in date order, builds summaries once, and asks 67 questions through **chat()**. For example, a future question asks what Jordan will drink on April 30, after the ingested period. See the [runner](/home/cloaked/projects/LifelongAgent/benchmarks/synthetic/run_v3.py:115) and [questions](/home/cloaked/projects/LifelongAgent/datasets/eval/qa_pairs.json).

| Question type | Number | What it tests |
| --- | ---: | --- |
| Facts and changing facts | 22 | What is known about Jordan? |
| Recall | 24 | What was chosen on a past date? |
| Pattern identification | 3 | What repeating rule explains the choices? |
| Future prediction | 9 | What choice would the rule predict after day 60? |
| Transition | 9 | When did a choice first change? |

**Scoring:** Each reply gets word-overlap **F1** from 0 to 1 and a separate LLM judge verdict of **right or wrong**. The judge sees the question, expected answer, and reply, but not the source conversation. It accepts paraphrases and extra details when the core answer is right. Scores are averaged over questions and reported by type and difficulty. See the [judge prompt and scoring code](/home/cloaked/projects/LifelongAgent/benchmarks/synthetic/run_v3.py:39).

The saved run scored **0.143 F1** and **32/67 judge-marked correct**; it got **0/3 pattern-identification questions**. This is an archived run using gemini-3.1-flash-lite; the current runner defaults to gemini-3.5-flash, so these numbers are not a fresh measurement of the current default. See the [saved results](/home/cloaked/projects/LifelongAgent/results/synthetic/v3/qa_results.json) and [current default](/home/cloaked/projects/LifelongAgent/benchmarks/synthetic/run_v3.py:102).

### LoCoMo: long conversations

LoCoMo contains **10 long conversations between pairs of people**, each split into dated sessions with questions about what happened. The v3 runner makes a separate store for each conversation, ingests its sessions, builds summaries, and lets the tool-using reader answer. The saved v3 run covers **one conversation: 19 sessions and 199 questions**. See the [runner](/home/cloaked/projects/LifelongAgent/benchmarks/locomo/run_v3_agentic.py:72) and [saved run](/home/cloaked/projects/LifelongAgent/results/locomo/v3_agentic/README.md).

The [LoCoMo scorer](/home/cloaked/projects/LifelongAgent/benchmarks/locomo/common.py:43) uses word-overlap F1 with stemming for single-hop, temporal, and commonsense questions. Multi-hop questions use F1 for each comma-separated answer part. Adversarial questions score **1** only when the reply contains an abstention phrase such as “I don't know.” The saved overall score is **0.460** across 199 questions. That is an average F1 score, **not** 46% of answers fully correct. See the [saved scores](/home/cloaked/projects/LifelongAgent/results/locomo/v3_agentic/locomo_results.json).

### What these scores establish

The scripts score **final answers**. They do not separately check whether every choice was extracted or whether a summary's proposed rule is correct. Synthetic questions directly test routine discovery; LoCoMo mainly tests retrieval and answering from long conversations. LoCoMo's adversarial score checks for an abstention phrase, so correcting a false premise without that phrase still scores zero. Offline tests with scripted model responses check data flow, not extraction or reasoning quality. See the [LoCoMo rule](/home/cloaked/projects/LifelongAgent/benchmarks/locomo/common.py:72) and [v3 tests](/home/cloaked/projects/LifelongAgent/tests/test_memory_v3.py:1).

For the implementation details, see [Agent structure](/home/cloaked/projects/LifelongAgent/.research/memory-v3-architecture/agent-structure.md) and the [full walkthrough](/home/cloaked/projects/LifelongAgent/.research/memory-v3-architecture/report.md).
