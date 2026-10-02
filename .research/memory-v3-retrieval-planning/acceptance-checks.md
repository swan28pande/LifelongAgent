# Future acceptance checks

These are proposed checks for a later implementation. No test code was written or run in this investigation.

Use the actual compiled reader graph with controlled decision-model responses and spies at the final retrieval invocation point. The test target must be `memory_v3_update`, not its sibling `memory_v3`. Spy entries, controller events and terminal state must agree; counting model-emitted call proposals alone is insufficient.

## Enforcement behavior

| Scenario | Required observation |
|---|---|
| Answer exists in the current conversation | A structured answer decision returns a final answer; zero retrievals |
| One lookup is sufficient | Decision precedes the selected invocation; next decision answers; exactly one call |
| First result leaves a specific gap | Next decision receives that result and remaining slots and selects one further useful lookup |
| Five attempts needed | Exactly five invocations; the next decision is answer-only and returns a terminal answer |
| Model requests a sixth lookup | Zero additional invocations; rejection/answer-only repair/finalization is recorded |
| Evidence still insufficient after five | Unknown/qualified answer in the agreed format; stop reason identifies exhausted budget rather than asserting global memory absence |
| Tool returns no matches | One slot spent; subsequent decision sees that source-specific empty result |
| Tool returns name/date correction text | One slot spent because the tool ran; the next decision may correct deliberately |
| Retrieval raises an expected error | Attempt counted once; error observation returned to decision/finalization; no automatic uncounted re-execution |
| Unexpected executor failure/cancellation before state update | No executor restart or further retrieval from the old count; terminal failure/fallback only |
| Malformed decision or unknown tool name | No tool invoked; bounded planning repair; malformed output never becomes trusted state |
| Invalid semantic arguments | Negative/zero limits, unreasonable `k`, invalid dates/type/level and extra keys are rejected before commitment/invocation |
| Argument normalization | Equivalent omitted/defaulted arguments produce one canonical commitment; meaningful changes are detected |
| Proposed name differs from committed name | Wrong tool never begins execution; commitment preserved; a distinct correct-call retry occurs or finite retry exhaustion returns fallback |
| Wrong but registered/hidden tool proposed | Same blocking guarantee as an unknown tool; membership in the global registry is not enough |
| Arguments differ from commitment | No invocation with changed arguments; same committed call retried |
| Multiple calls proposed together | Entire invalid batch blocked before any dispatch, even if all calls name the selected tool |
| Raw response contains parsed first action plus extra/invalid calls | Raw-response validation detects the extras; a first-only parser cannot bypass the guard |
| Repeated deviation or malformed output | Repair/model-call limit terminates; never expands the allowed set or loops forever |
| Duplicate/stale decision reused | A completed/consumed commitment cannot invoke again without a new decision and another accounted slot |
| Unexpected low recursion setting | Controlled terminal failure/fallback; never presented as a valid answer extracted from a tool message |
| `chat`, `chat_with_trace`, and `converse` | All use the same guard, cap, correct retry, and finalization behavior |
| Two live turns | Each receives a fresh five-slot budget while seeing the day's conversation |
| Concurrent independent chat invocations on one agent | Each has isolated commitment/count/repair state; neither overwrites the other |
| Trace after rejection and failure | Proposed/blocked calls, actual attempts and outcomes remain distinguishable; execution count is at most five |
| Trace and public answer | Final answer field/string is consistent; internal decisions are not mixed into user-visible answer text |

The direct-dispatch implementation should make model-emitted post-commit deviations structurally unavailable. Still exercise the invocation guard with a mismatched proposed call to prove it cannot execute. If a native emission phase is retained, drive wrong/multiple/changed-argument responses through that real phase and verify repair preserves the original commitment. Do not simulate rejection by silently rewriting a proposal and calling it a successful retry.

## Retry and resume checks

For the preferred short stateless loop, verify the executor has no automatic tool/node retry configured. A consciously repeated lookup is a new accounted decision/invocation. Bounded model transport retries do not retrieve memory, but must not create an unbounded wait.

If resumability is later added, add fault injection at dispatch, after tool-body entry, before counter/result persistence, and during resume. The five-call limit must then be checked against a durable attempt ledger for the same logical query. Test replay/time-travel behavior explicitly. This is additional functionality, not part of the minimum proposal.

## Retrieval adequacy and provider checks

- Use representative exact-date, timeline/count, broad-summary, topic-search and current-conversation questions. Check useful tool choice, stop-on-sufficient-evidence behavior and answer correctness separately from execution-budget compliance.
- Include more than 200 structured rows, missing summaries, degraded/stale summaries, unavailable semantic indexes, raw-only evidence, multiple speakers, tied dates and large date windows. Verify partial evidence is not represented as exhaustive coverage.
- Validate the real structured envelope against the installed Google/OpenAI integration and selected model/backend. Check malformed output, provider refusal/empty content, raw extra calls, named-choice support and callback/token accounting.
- Compare answer accuracy, uncertainty rate, actual executions, model calls and latency with the current reader using identical data/tasks. A five-call limit can reduce unnecessary retrieval while making some multi-hop tasks harder; do not infer net quality improvement from the cap alone.

Pass enforcement first with deterministic spies. Then use a small provider integration check and representative evaluation to resolve compatibility and quality questions. No additional planner service or infrastructure is needed to test the proposed short loop.
