# Required behavior: planned and bounded memory retrieval

Status: required behavior recorded; implementation investigation complete; implementation not performed or authorized in this task.

Requested by the user on 2026-10-01 for `memory_v3_update/`.

## User requirements

1. Before executing retrieval tools, the agent must plan from the current task/query and available evidence and choose the most useful next tool.
2. Prefer an adaptive loop: decide whether an answer is already possible; otherwise select one tool, execute it, inspect the result, and decide again.
3. Execute at most **five retrieval tool calls per query/live turn** before returning an answer. Five is a ceiling, not a target; answering with zero or fewer calls is allowed.
4. Enforce the selected tool through framework/application control flow. A different tool must be blocked before execution, and the agent must retry the correct planned call.
5. Inspect the existing implementation, document the relevant implementation and reliability gaps, and download the actual framework repositories to inspect their source/documentation.
6. Produce useful repository maps and investigation documentation as needed. **Do not edit application code or implement the change.**

## Operational interpretation for investigation

These details make the request reviewable; they are interpretations or proposed defaults, not additional verbatim user instructions.

- Scope: the retrieval/answer path shared by `chat()`, `chat_with_trace()`, and `converse()`. Ingestion and summary generation are outside the retrieval-call budget.
- Each actual invocation of a registered retrieval tool costs one slot, including invocations that return no results or fail after execution begins. Rejecting a proposed call before invocation does not spend a retrieval slot; repair attempts need a separate finite limit.
- One selected tool runs at a time. A model response containing multiple calls cannot bypass the required reassessment between tools.
- A committed selection includes the tool name and validated arguments. Execution must use that selection; changing arguments or selecting another tool requires a new decision, rather than silently changing the committed call.
- After the fifth execution, control must enter an answer-only path with retrieval disabled. If evidence is insufficient, return an uncertainty response consistent with the existing answer interface.
- A blocked deviation must preserve the committed selection while the agent retries it. Invalid planning output can instead be repaired before any selection is committed.
- Retries must terminate. On repeated invalid output or execution errors, return an explicit safe fallback without silently authorizing another tool or exceeding the five-call ceiling.
- Record decisions, blocked proposals, actual executions, and the stopping reason separately, so a trace can demonstrate compliance.

## Research deliverables

Investigation, source evidence, source versions, implementation proposal, repository navigation, and future acceptance checks live under [`.research/memory-v3-retrieval-planning/`](../../.research/memory-v3-retrieval-planning/).

The investigation must distinguish existing behavior, framework capabilities verified in downloaded source, recommended design, and compatibility questions requiring later implementation-time validation.
