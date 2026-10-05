# DirectPrompting: full-context monthly benchmark

Run `python -m setup_2.run --method full_context` from the project root. The
[baseline guide](../README.md) covers installation, parallel users and resume.

`method.py` adapts the existing `experiments/methods/full_context.py` class
`FullContext`. Raw observed sessions persist in `store/baseline.sqlite3`,
are restored chronologically on restart, and appear directly in the answer
prompt with their conversation dates. There is no embedding model or retrieval
index. The shared answer model, prompt and original question probe date are used.

Finalization includes **all** history observed through the current checkpoint.
Earlier sessions are never silently dropped. If history exceeds
`experiments.config.FULL_CONTEXT_MAX_TOKENS` (900,000), using the existing
four-characters-per-token estimate, the run fails explicitly before evaluation.
That estimate does not replace the model's actual token limit; a provider
context-limit error also leaves the checkpoint resumable. Reports retain
`sessions_dropped: 0` and record session and approximate context-token counts.

Every current user's complete history fits this estimate: approximately 225,690
to 324,100 tokens. Actual prompt token usage is recorded by the model callback
during a live run. No live model calls were made during this setup.

Duplicate identical sessions are ignored; changed content for an existing ID
is rejected. Evaluations and judging never enter the stored history. Interrupted
memory builds roll back to the previous checkpoint; missing answers/grades
resume against the fixed built history.

Install this folder's `requirements.txt`; no embedding model or database server
is required. Offline tests verify persistence, original dates, cumulative
questions, complete-context behavior, interrupted builds/evaluation and separate
user processes.
