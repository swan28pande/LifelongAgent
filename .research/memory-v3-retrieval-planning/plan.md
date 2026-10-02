# Investigation plan

Date: 2026-10-01.

## Objective

Determine the smallest reliable framework-enforced plan → one retrieval → reassess loop for `memory_v3_update/`, with at most five executed retrieval calls per invocation and blocked/retried deviations from the committed choice. Record a concrete implementation proposal without editing application code.

Requirement source: [specification](../../.scratch/memory-v3-retrieval-planning/spec.md).

## Initial discovery

- `memory_v3_update/agent.py` imports `langchain.agents.create_agent`, supplies all read tools, and sets a graph recursion limit of 60.
- `CHAT_SYSTEM` contains retrieval and stopping advice but no explicit framework-controlled planning state.
- The active Python interpreter has neither LangChain nor LangGraph installed. Package versions used by a deployed reader must therefore not be inferred from this environment.
- Framework source investigation should cover both LangChain's agent/middleware implementation and LangGraph's graph/tool execution machinery.

## Definitions and assumptions

- A decision is a concise structured record describing the evidence gap and either one next retrieval or an answer. It is not a request to expose private chain of thought.
- A retrieval execution is an invocation of one of `build_read_tools()`'s registered tools; a planner's structured-output transport is not memory retrieval.
- A committed selection is a validated tool name and argument set saved before execution.
- Per-query state resets for each public invocation, including each `converse()` turn; live conversational history is input evidence rather than a persistent budget.
- Downloaded repositories are reference artifacts, not runtime dependency changes. No packages will be installed for this task.

## Questions and evidence required

| Question | Required evidence | Sources |
|---|---|---|
| What actually runs today? | Entry points, model setup, registered tools, prompts, store operations, tracing, tests and harness imports with line references | Local source |
| What does the framework currently enforce? | Agent graph routing, tool concurrency, error handling, recursion semantics, middleware count/exit behavior | Official cloned framework repositories and docs |
| What is the smallest dependable design? | Explicit state/control-flow contract, pre-execution validation, budget and retry rules, comparison of middleware vs a small graph | Synthesis from source evidence |
| What can make the plan fail despite routing? | Missing data, result truncation, approximate retrieval, stale summaries, provider capabilities and error paths | Local source and official integration/framework docs |
| How would future implementation prove compliance? | Behavior-level acceptance matrix with adversarial proposals, parallel calls, errors, exhausted budgets and per-turn reset | Proposed checks; no test implementation now |

## Counter-evidence checks

- Does `create_agent` already implement an explicit pre-tool planner or a precise five-call limit?
- Does an existing limit count executions or model proposals, and does it produce the requested task answer when exhausted?
- Does exposing one tool truly prevent a model from emitting another name or multiple calls?
- Can a tool batch, automatic tool retry, graph replay, or separately invoked public path bypass counting or validation?
- Can a planning step be combined with the next-action decision to avoid a second LLM call solely to reproduce a selected call?
- Are current evaluation imports actually using `memory_v3_update/`?

## Workflow

1. Preserve source fingerprints and record requirements.
2. Map the local reader, tools, supporting storage, and evaluation interfaces.
3. Download the official LangChain and LangGraph repositories; record immutable revisions. Download their official docs repository if the code repositories do not include the relevant current documentation.
4. Inspect relevant source and tests, cross-check official online documentation, and distinguish evidence from interpretation.
5. Compare minimal implementation paths and document a preferred proposal, open compatibility questions, and future acceptance checks.
6. Independently review the synthesis against the requirement and verify application source fingerprints are unchanged.

## Stopping conditions

Stop when the requirements, local flow, authoritative framework behavior, proposed enforced loop, reliability gaps, navigation map, and acceptance checks are documented and source immutability is verified. Do not implement or edit code, upgrade dependencies, run model-backed retrieval, ingest data, or change project architecture documentation as though the proposal had been adopted.

## Completion

The requirement specification, local/upstream navigation map, evidence ledger, immutable source manifest, implementation proposal, and future acceptance matrix are complete. Independent read-only audits checked local retrieval behavior and framework enforcement, followed by fresh-context synthesis and final document review. [Verification](verification.md) records source immutability and the limits of this investigation. Provider compatibility, test execution and empirical quality evaluation remain future implementation work.
