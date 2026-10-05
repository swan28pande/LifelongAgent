# Monthly cumulative memory_v4 evaluation

Implement an isolated loader and runner under `setup_2/`, using the existing
memory_v4 agent with summaries and distillation and the existing judge.

The accepted protocol is recorded in `../DECISIONS.md`: chronological monthly
ingestion into a growing per-user database; cumulative question repeats; original
question dates/references retained. Future months cannot be ingested before the
current checkpoint's answers and grades are durable. Record month-specific results,
latency, tool calls, and usage. Resume safely after interruption.

Verify with offline fixtures and actual-data scheduling checks. No live evaluation,
dataset correction, or changes outside `setup_2/` are part of this implementation.
