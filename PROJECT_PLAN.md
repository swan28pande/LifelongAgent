# Lifelong Assistant

**Author:** Swanand Pande
**Drafted:** 28 August 2026
**Revised:** 10 September 2026

An assistant that knows a person's facts and preferences — and, more importantly, knows
that preferences change. Someone who likes espresso today may prefer a latte a month
from now.

---

## Introduction

A person holds preferences about the activities and things around them, and those
preferences are not static. They drift, cycle, and occasionally break and reform.

As people lean on assistants for everyday tasks, capturing that evolution becomes a
product requirement rather than a nicety. An assistant that remembers only the latest
observation is wrong about a cycle; one that remembers only the first is wrong about a
change. The goal is a system that holds both the history and the rule that explains it.

---

## Scope

1. **Dataset 1** — assistant/user conversations containing facts and evolving preference patterns.
2. **Dataset 2** — a learning-trace dataset, in which the assistant evolves its understanding one day at a time.
3. Generate a response to a new user message.
4. Initiate a new conversation.

---

## Current state

### Ground truth

Three preferences are embedded in the synthetic data, each with a different *kind* of
rule, so that a system cannot succeed by learning one shape of pattern:

| Entity | Rule | Why it's different |
|---|---|---|
| `coffee` | Fixed 7-day cycle | Periodic — requires counting across weeks |
| `clothing` | Alternates every 3 days | Periodic — period does not align to the week |
| `exercise` | Day-of-week: Mon/Wed/Fri → morning yoga, Tue/Thu → evening run | Calendar-based — no fixed period at all |

The exercise rule also carries weekend states — Saturday climbing, Sunday rest — which
matter more than they look. A rest day is a *choice*, not an absence of data, and a
system that silently drops it will misread the weekly rhythm. This is a known failure
mode worth testing for explicitly.

### Main evaluation dataset

| | |
|---|---|
| Generator | gpt-4o-mini |
| Sessions | 60 (1 session = 1 day) |
| QA pairs | 67 |
| Question types | factual, factual_evolving, recall, pattern_identification, prediction, transition |

---

## Things to do

### (a) Agentic approach 

Do not hardcode the context. Expose tools and let the agent decide what to call.

The prior design ran fixed steps. The agentic version hands the model tools instead, so it can inspect
the store before writing, correct its own earlier entries, and
decide when a summary level is worth building.

#### Tools

| Tool | Mode | Purpose |
|---|---|---|
| `list_entities` | both | Read the live taxonomy before writing or querying |
| `list_preferences` | write | Inspect what is already recorded |
| `add_preference` | write | Record one dated choice |
| `remove_preference` | write | Correct a duplicate or mis-filed entry |
| `index_conversation` | write | Add a chunk to the RAG store |
| `summarize_week` | write | Per-entity transitions plus facts |
| `summarize_month` | write | Consolidate weeks into candidate rules |
| `summarize_year` | write | Confirm which rules held |
| `summarize_lifetime` | write | Standing profile |
| `build_context` | read | One-call assembly for a query |
| `search_preferences` | read | Exact dated timeline |
| `search_conversations` | read | Verbatim retrieval |
| `get_summary` | read | Fetch one summary by level |

Summarization is deliberately split by level rather than exposed as one call, so the
agent chooses how far up the hierarchy to roll and the evaluation can attribute cost to
a specific level.

Context assembly holds no domain knowledge: it reads the entity taxonomy out of the
store at call time and matches the query against whatever is actually there, then
discards any entity name the model returns that does not literally exist. The earlier
builder carried a hand-written table of expected domains with per-domain keyword lists —
which works only on a dataset whose categories are known in advance, and silently
returns nothing for anything else.

---

### (b) Continual learning 

The agent should build its understanding on the go — one day at a time, using only what
it has seen so far.





