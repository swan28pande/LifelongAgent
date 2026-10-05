# Recall audit sources

Canonical inputs are `datasets/v2/u1..u5/probing_questions.json`, `qa_pool.json`, `world_state.json`, `conversations.json`, and the individual session JSON files referenced by recall evidence or the focused transcript reviews. Persona YAML files provide exception pools and explicit exceptions. [sources.json](sources.json) lists every read input with its SHA-256 hash.

Generator interpretation was checked against:

- [qa.py](/home/cloaked/projects/LifelongAgent/generator_v2/qa.py:160): recall truth, direct/inferred tags, same-phase nearest evidence, fallback, and bounded change dates.
- [rules.py](/home/cloaked/projects/LifelongAgent/generator_v2/rules.py:1): calendar weekday rules, regime anchors, cycles, nested and conditional rules.
- [simulator.py](/home/cloaked/projects/LifelongAgent/generator_v2/simulator.py:224): sampling, forced exceptions, and per-day truth.
- [conversation.py](/home/cloaked/projects/LifelongAgent/generator_v2/conversation.py:32): daily-only generation, paraphrase policies, absent topic restrictions, and closed-option saved extraction.
- [probing.py](/home/cloaked/projects/LifelongAgent/generator_v2/probing.py:88): monthly assignment considers target dates only for `fact_at_time`.
- [DECISIONS.md](/home/cloaked/projects/LifelongAgent/generator_v2/DECISIONS.md:1): resumed base anchors, intended inference, and observation limits.

No root `CONTEXT.md` or relevant ADR files exist in this checkout. The investigation follows the parent [audit plan](../plan.md). It uses only local evidence and no new model validation, paid APIs, or network access.
