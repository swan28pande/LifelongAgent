# Sources and downloaded revisions

Investigation date: 2026-10-01, America/New_York. Repository metadata captured at `2026-10-02T01:28:12Z` (the same local evening).

All downloads are shallow Git clones of official repositories, with a full working tree for the downloaded revision. They are reference artifacts under `upstream/`; no packages were installed or project dependencies changed. Machine-readable provenance is in [upstream-revisions.json](upstream-revisions.json).

## Downloaded repositories

| Repository | Local directory | Immutable revision | Package metadata at that revision |
|---|---|---|---|
| [LangChain](https://github.com/langchain-ai/langchain) | [upstream/langchain](upstream/langchain/) | `ff46bb478bfbef98dd35ea705fa4e92a33881142` | `langchain` 1.4.3; `langchain-core` 1.6.6 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | [upstream/langgraph](upstream/langgraph/) | `157a06dda988d85afeb8751ff27b35ab3f4f8bf4` | `langgraph` 1.2.12; `langgraph-prebuilt` 1.1.0 |
| [Official documentation](https://github.com/langchain-ai/docs) | [upstream/docs](upstream/docs/) | `99dd9a9e38d59b9bff54354f6c3ef790c1be946d` | Documentation source, not a runtime dependency |
| [Google integration](https://github.com/langchain-ai/langchain-google) | [upstream/langchain-google](upstream/langchain-google/) | `b476e4b0a0bffbeb5c296bcd9e83aa3f07b68a40` | `langchain-google-genai` 4.4.0 |

These are downloaded branch revisions, not a claim that the application has those versions installed. The active interpreter reports LangChain, LangGraph, LangChain Core, OpenAI integration, and Google integration as not installed; Pydantic is 2.11.9. No reader-specific dependency lock or manifest was found outside unrelated evaluation/baseline dependencies. Compatibility must be checked against the environment that will actually run the agent.

## Authoritative implementation sources

| Source | Relevant locations |
|---|---|
| [LangChain agent factory, pinned](https://github.com/langchain-ai/langchain/blob/ff46bb478bfbef98dd35ea705fa4e92a33881142/libs/langchain_v1/langchain/agents/factory.py) | ToolNode construction; dynamic model tool binding; model/tool routing and middleware hooks |
| [Tool limiter, pinned](https://github.com/langchain-ai/langchain/blob/ff46bb478bfbef98dd35ea705fa4e92a33881142/libs/langchain_v1/langchain/agents/middleware/tool_call_limit.py) | `ToolCallLimitState`, `after_model`, batch counting and terminal behavior |
| [Limiter tests, pinned](https://github.com/langchain-ai/langchain/blob/ff46bb478bfbef98dd35ea705fa4e92a33881142/libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_tool_call_limit.py) | Proposal counts and mixed/parallel batch behavior |
| [Tool selector, pinned](https://github.com/langchain-ai/langchain/blob/ff46bb478bfbef98dd35ea705fa4e92a33881142/libs/langchain_v1/langchain/agents/middleware/tool_selection.py) | Name-only selection schema, last-user-message input, filtered exposure, malformed-output handling |
| [ToolNode, pinned](https://github.com/langchain-ai/langgraph/blob/157a06dda988d85afeb8751ff27b35ab3f4f8bf4/libs/prebuilt/langgraph/prebuilt/tool_node.py) | Registered-tool map, invocation, concurrency, tool-name validation and errors |
| [StateGraph, pinned](https://github.com/langchain-ai/langgraph/blob/157a06dda988d85afeb8751ff27b35ab3f4f8bf4/libs/langgraph/langgraph/graph/state.py) | Nodes, conditional edges, state schemas and retry defaults |
| [Graph execution, pinned](https://github.com/langchain-ai/langgraph/blob/157a06dda988d85afeb8751ff27b35ab3f4f8bf4/libs/langgraph/langgraph/pregel/main.py) | Recursion exhaustion and graph invocation |
| [Graph retry implementation, pinned](https://github.com/langchain-ai/langgraph/blob/157a06dda988d85afeb8751ff27b35ab3f4f8bf4/libs/langgraph/langgraph/pregel/_retry.py) | Task invocation and repeated execution after exceptions |
| [Google chat model, pinned](https://github.com/langchain-ai/langchain-google/blob/b476e4b0a0bffbeb5c296bcd9e83aa3f07b68a40/libs/genai/langchain_google_genai/chat_models.py) | Structured JSON, raw/parsed results, function-calling parser and named tool choice |
| [OpenAI chat model, pinned](https://github.com/langchain-ai/langchain/blob/ff46bb478bfbef98dd35ea705fa4e92a33881142/libs/partners/openai/langchain_openai/chat_models/base.py) | Structured output, named tool choice, parallel tool flag and raw/parsed results |

## Official online documentation cross-checks

- [Agents](https://docs.langchain.com/oss/python/langchain/agents): high-level agent loop and customization.
- [Prebuilt middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in): tool limits, tool selector, and to-do middleware.
- [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom): model/tool wrappers and state/control hooks.
- [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents): structured decisions and graph routing.
- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api): state, conditional edges, super-steps and recursion.
- [Models](https://docs.langchain.com/oss/python/langchain/models): model-level structured output.
- [Google integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai): structured output and provider/backend options.
- [SQLite SELECT](https://www.sqlite.org/lang_select.html#limitoffset): negative `LIMIT` values and tied sort order.

The downloaded middleware guide disagrees with the downloaded limiter implementation on `exit_behavior="end"` with pending calls. [Evidence F4](evidence.md) records this discrepancy; use the pinned implementation and its tests for that revision's behavior.

## Local evidence and verification

The eight files in `memory_v3_update/`, relevant tests, experiment adapters, and benchmark runners were read statically. [application-source-baseline.json](application-source-baseline.json) records 213 pre-existing application/source files and this package's README, excluding dependency caches, downloaded investigation repositories, and research/scratch documents.

There were no application executions, dependency installations, model-backed experiments, or tests in this investigation. Downloaded tests were inspected as evidence, not executed. Conclusions about framework behavior are source-verified; provider compatibility and empirical answer quality remain untested.
