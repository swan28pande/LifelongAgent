# Make the three baseline stores grow and resume safely

Type: task
Status: resolved

Persist raw observations for all baselines, save/update NaiveRAG's FAISS index,
restore full history for DirectPrompting, and persist/update TiMem nodes and cached
vectors with correct calendar ranges. Bind native TiMem generation to the shared
LLM and usage callback. All recoverable state belongs under the user's store.

## Answer

Created a shared SQLite observation/metadata store; NaiveRAG saves and increments
its FAISS index, and full context restores all dated sessions without silently
truncating them. TiMem persists generated nodes/vectors, reconstructs an isolated
in-process Qdrant index from cached vectors, and updates only changed summaries.
Partial weeks are separated at month edges and period bounds end at observed data.

Native TiMem generation uses the shared model/callback, async streaming and one
event loop per user invocation; clients are closed on that loop. Generation failures
and timeouts propagate rather than invoking native infinite retries. Prompt logs
remain inside the store. Four small changes are confined to the copied native source;
original source code and native generation prompts remain untouched outside setup_2.

Actual vector-library tests cover restarts, failed builds, partial finalization,
missing answers/grades, idempotence, empty months and independent users. Native
generator checks verify five levels, usage counts, prompt log paths, loop-bound
transport behavior, async client cleanup and timeout cancellation.
