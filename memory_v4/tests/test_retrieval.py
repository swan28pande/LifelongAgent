"""Behavior checks against the real compiled reader, with no model/network calls."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from memory_v4.agent import AgenticMemoryAgent
from memory_v4.retrieval import (
    MAX_OBSERVATION_CHARS,
    MAX_RETRIEVAL_CALLS,
    FinalAnswer,
    PlannedMemoryReader,
    RetrievalDecision,
    _Run,
    _State,
)


def retrieve(topic="first", *, name="lookup", arguments=None):
    return {
        "action": "retrieve",
        "evidence_gap": f"Need evidence about {topic}",
        "tool_name": name,
        "arguments_json": json.dumps(
            arguments if arguments is not None else {"topic": topic}
        ),
        "answer": "",
    }


def answer(text="done"):
    return {
        "action": "answer",
        "evidence_gap": "",
        "tool_name": "",
        "arguments_json": "{}",
        "answer": text,
    }


def packet(parsed, *, calls=None, invalid_calls=None, parsing_error=None):
    return {
        "raw": AIMessage(
            content="", tool_calls=calls or [], invalid_tool_calls=invalid_calls or []
        ),
        "parsed": parsed,
        "parsing_error": parsing_error,
    }


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses) if not callable(responses) else responses
        self.requests = []
        self.schemas = []

    def with_structured_output(self, schema, *, method, include_raw):
        assert method == "json_schema" and include_raw
        self.schemas.append(schema)

        def respond(messages, config):
            self.requests.append((schema, messages))
            response = (
                self.responses(messages, schema)
                if callable(self.responses)
                else next(self.responses)
            )
            if isinstance(response, Exception):
                raise response
            return response if "raw" in response else packet(response)

        return RunnableLambda(respond)


def lookup_tools(calls, *, fail=False, output=None):
    @tool
    def lookup(topic: str, attempt: int = 0) -> str:
        """Retrieve evidence for one topic."""
        calls.append(("lookup", topic, attempt))
        if fail:
            raise RuntimeError("index unavailable")
        return output if output is not None else f"Evidence: {topic}"

    @tool
    def other(topic: str) -> str:
        """Another registered retrieval tool, never implicitly authorized."""
        calls.append(("other", topic))
        return "unexpected"

    return [lookup, other]


def invoke(reader, query="question", **config):
    return reader.invoke(
        {"messages": [HumanMessage(content=query)]}, config=config or None
    )


def test_zero_retrieval_answer_is_terminal():
    calls = []
    result = invoke(
        PlannedMemoryReader(
            ScriptedModel([answer("already known")]), lookup_tools(calls)
        )
    )
    assert result["answer"] == "already known"
    assert result["num_tool_calls"] == 0 and calls == []
    assert result["messages"][-1].content == "already known"
    assert result["decisions"] == [{"action": "answer", "answer": "already known"}]


def test_generated_hierarchy_is_retrieved_through_public_planned_reader(
    agent_factory,
    conversations,
):
    model = ScriptedModel(
        [
            retrieve(
                "hobbies",
                name="semantic_retrieve_memory",
                arguments={"query": "camping"},
            ),
            answer("Camping is a hobby."),
        ]
    )
    agent = agent_factory(model)
    agent.ingest("2026-03-02", conversations)
    agent.build_summaries(distill=True)
    result = agent.chat_with_trace("What are the recorded hobbies?")
    assert result["answer"] == "Camping is a hobby."
    assert result["num_tool_calls"] == 1 and result["num_model_calls"] == 2
    assert result["tool_calls"] == [
        {"tool": "semantic_retrieve_memory", "args": {"query": "camping"}}
    ]
    assert result["rejections"] == []
    output = result["executions"][0]["output"]
    headers = [
        f"═══ {level} ═══"
        for level in ("DISTILLED", "LIFETIME", "YEAR", "MONTH", "WEEK")
    ]
    assert [output.index(header) for header in headers] == sorted(
        output.index(header) for header in headers
    )
    assert "REMAINING RETRIEVAL CALLS: 4" in model.requests[1][1][0].content
    assert "semantic_retrieve_memory" in model.requests[0][1][0].content
    assert "semantic_search_summaries" not in agent.tool_names()
    assert not any("distill_knowledge" == name for name in agent.tool_names())


def test_missing_summaries_are_reassessed_before_exact_fact_lookup(
    agent_factory,
    conversations,
):
    model = ScriptedModel(
        [
            retrieve(
                name="semantic_retrieve_memory", arguments={"query": "Alice's hobbies"}
            ),
            retrieve(
                name="search_memories",
                arguments={"entity": "hobby", "speaker": "alice", "type": "fact"},
            ),
            answer("Alice likes camping."),
        ]
    )
    agent = agent_factory(model)
    agent.ingest("2026-03-02", conversations)
    result = agent.chat_with_trace("What is Alice's hobby?")
    assert result["num_tool_calls"] == 2
    assert "no summaries available" in result["executions"][0]["output"]
    assert "camping" in result["executions"][1]["output"]
    assert "no summaries available" in model.requests[1][1][-1].content
    assert result["answer"] == "Alice likes camping."


def test_summary_and_detail_tools_share_five_call_budget(agent_factory, conversations):
    summary = retrieve(name="semantic_retrieve_memory", arguments={"query": "camping"})
    decisions = (
        [summary]
        + [
            retrieve(
                name="get_summary",
                arguments={"level": level, "identifier": period, "speaker": "alice"},
            )
            for level, period in (
                ("week", "2026-W10"),
                ("month", "2026-03"),
                ("year", "2026"),
            )
        ]
        + [
            retrieve(
                name="search_memories",
                arguments={"entity": "hobby", "speaker": "alice"},
            )
        ]
    )
    model = ScriptedModel(decisions + [summary, {"answer": "Camping is recorded."}])
    agent = agent_factory(model)
    agent.ingest("2026-03-02", conversations, update_summaries=True)
    result = agent.chat_with_trace("Describe the hobby history.")
    assert result["num_tool_calls"] == 5
    assert [entry["tool"] for entry in result["executions"]] == [
        "semantic_retrieve_memory",
        "get_summary",
        "get_summary",
        "get_summary",
        "search_memories",
    ]
    assert all(entry["status"] == "success" for entry in result["executions"])
    assert result["rejections"][0]["phase"] == "answer"
    assert model.requests[-1][0] is FinalAnswer
    assert result["stop_reason"] == "tool_budget_exhausted"


def test_changed_summary_call_is_blocked_and_original_selection_retried(memory_store):
    from memory_v4.tools import build_read_tools

    class ChangedCallReader(PlannedMemoryReader):
        corrupt_next = True

        def _proposal(self, selected):
            call = super()._proposal(selected)
            if self.corrupt_next:
                self.corrupt_next = False
                return {**call, "name": "search_memories"}
            return call

    memory_store.save_summary("lifetime:alice", "Profile", "camping")
    model = ScriptedModel(
        [
            retrieve(name="semantic_retrieve_memory", arguments={"query": "camping"}),
            answer("Camping."),
        ]
    )
    reader = ChangedCallReader(model, build_read_tools(memory_store))
    result = invoke(reader)
    assert result["num_tool_calls"] == 1 and result["num_model_calls"] == 2
    assert len(result["rejections"]) == 1
    assert result["rejections"][0]["phase"] == "execution"
    assert result["executions"][0]["tool"] == "semantic_retrieve_memory"
    assert memory_store.embeddings.query_calls == ["camping"]


def test_extra_summary_arguments_are_rejected_before_retrieval(memory_store):
    from memory_v4.tools import build_read_tools

    memory_store.save_summary("lifetime:alice", "Profile", "camping")
    model = ScriptedModel(
        [
            retrieve(
                name="semantic_retrieve_memory",
                arguments={"query": "camping", "k": 100},
            ),
            retrieve(name="semantic_retrieve_memory", arguments={"query": "camping"}),
            answer("Camping."),
        ]
    )
    result = invoke(PlannedMemoryReader(model, build_read_tools(memory_store)))
    assert result["num_tool_calls"] == 1
    assert "Unknown arguments" in result["rejections"][0]["reason"]
    assert memory_store.embeddings.query_calls == ["camping"]


def test_reassesses_result_and_stops_after_one_call():
    calls = []
    model = ScriptedModel([retrieve(), answer()])
    result = invoke(PlannedMemoryReader(model, lookup_tools(calls)))
    assert calls == [("lookup", "first", 0)]
    assert result["num_tool_calls"] == 1 and result["num_model_calls"] == 2
    assert model.requests[0][0] is RetrievalDecision
    assert [d["action"] for d in result["decisions"]] == ["retrieve", "answer"]
    assert "Evidence: first" in model.requests[1][1][-1].content
    assert "REMAINING RETRIEVAL CALLS: 4" in model.requests[1][1][0].content
    assert result["tool_calls"] == [
        {"tool": "lookup", "args": {"attempt": 0, "topic": "first"}}
    ]


def test_fifth_call_switches_to_answer_only_schema():
    calls = []
    model = ScriptedModel(
        [retrieve(str(n)) for n in range(5)] + [{"answer": "complete"}]
    )
    result = invoke(PlannedMemoryReader(model, lookup_tools(calls)))
    assert len(calls) == result["num_tool_calls"] == MAX_RETRIEVAL_CALLS
    assert (
        result["answer"] == "complete"
        and result["stop_reason"] == "tool_budget_exhausted"
    )
    assert result["num_model_calls"] == 6
    assert model.requests[-1][0] is FinalAnswer
    assert "AVAILABLE TOOLS" not in model.requests[-1][1][0].content
    assert "REMAINING RETRIEVAL CALLS: 0" in model.requests[-1][1][0].content


def test_sixth_retrieval_proposal_is_blocked_then_answer_repaired():
    calls = []
    model = ScriptedModel(
        [retrieve(str(n)) for n in range(6)] + [{"answer": "I don't know."}]
    )
    result = invoke(PlannedMemoryReader(model, lookup_tools(calls)))
    assert len(calls) == result["num_tool_calls"] == 5
    assert result["rejections"][0]["phase"] == "answer"
    assert result["answer"] == "I don't know."
    assert result["num_model_calls"] == 7


@pytest.mark.parametrize(
    "output", ["(no matching memories)", "No entity named first; use known."]
)
def test_empty_and_corrective_results_spend_a_slot(output):
    calls = []
    result = invoke(
        PlannedMemoryReader(
            ScriptedModel([retrieve(), answer()]), lookup_tools(calls, output=output)
        )
    )
    assert len(calls) == result["num_tool_calls"] == 1
    assert result["executions"][0]["output"] == output


def test_tool_errors_count_once_and_are_reassessed():
    calls = []
    model = ScriptedModel(
        [retrieve(str(n)) for n in range(5)] + [{"answer": "I don't know."}]
    )
    result = invoke(PlannedMemoryReader(model, lookup_tools(calls, fail=True)))
    assert len(calls) == result["num_tool_calls"] == 5
    assert all(entry["status"] == "error" for entry in result["executions"])
    assert "index unavailable" in model.requests[1][1][-1].content


@pytest.mark.parametrize(
    "bad",
    [
        retrieve(name="missing"),
        retrieve(arguments={"topic": "first", "unknown": 1}),
        retrieve(arguments={"topic": 12}),
        retrieve(arguments={"topic": "first", "attempt": True}),
        {**retrieve(), "arguments_json": "[]"},
        {**retrieve(), "arguments_json": "not JSON"},
        {**retrieve(), "arguments_json": '{"topic":"first","topic":"second"}'},
        {**retrieve(), "answer": "also an answer"},
        {**answer(), "tool_name": "lookup"},
        {**answer(), "evidence_gap": "Still missing evidence"},
        {**retrieve(), "execution_count": -100},
    ],
)
def test_bad_decisions_are_repaired_before_commitment(bad):
    calls = []
    result = invoke(
        PlannedMemoryReader(
            ScriptedModel([bad, retrieve(), answer()]), lookup_tools(calls)
        )
    )
    assert calls == [("lookup", "first", 0)]
    assert result["num_tool_calls"] == 1 and len(result["rejections"]) == 1
    assert [d["action"] for d in result["decisions"]] == ["retrieve", "answer"]


@pytest.mark.parametrize(
    "kind", ["wrong", "multiple", "first_only", "invalid", "changed_transport"]
)
def test_raw_tool_calls_cannot_bypass_structured_validation(kind):
    decision = retrieve()
    calls = [{"name": "lookup", "args": {"topic": "first"}, "id": "bad"}]
    invalid = []
    if kind in ("multiple", "first_only"):
        calls.insert(
            0, {"name": "RetrievalDecision", "args": decision, "id": "transport"}
        )
    elif kind == "invalid":
        calls = []
        invalid = [
            {"name": "lookup", "args": "not JSON", "id": "bad", "error": "invalid"}
        ]
    elif kind == "changed_transport":
        calls = [
            {
                "name": "RetrievalDecision",
                "args": retrieve("other evidence"),
                "id": "transport",
            }
        ]
    actual = []
    model = ScriptedModel(
        [packet(decision, calls=calls, invalid_calls=invalid), retrieve(), answer()]
    )
    result = invoke(PlannedMemoryReader(model, lookup_tools(actual)))
    assert actual == [("lookup", "first", 0)]
    assert result["num_tool_calls"] == 1 and len(result["rejections"]) == 1
    assert result["rejections"][0]["proposals"]


def test_structured_output_transport_is_not_a_retrieval():
    decision = retrieve()
    calls = []
    model = ScriptedModel(
        [
            packet(
                decision,
                calls=[{"name": "RetrievalDecision", "args": decision, "id": "schema"}],
            ),
            answer(),
        ]
    )
    result = invoke(PlannedMemoryReader(model, lookup_tools(calls)))
    assert result["num_tool_calls"] == len(calls) == 1
    assert result["rejections"] == []


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda call: {**call, "name": "other"},
        lambda call: {**call, "name": "missing"},
        lambda call: {**call, "args": {"topic": "changed"}},
        lambda call: {**call, "id": "stale"},
        lambda call: [call, call],
        lambda call: {**call, "extra": "unplanned"},
    ],
)
def test_dispatch_deviation_is_blocked_and_same_commitment_retried(corrupt):
    class TamperedReader(PlannedMemoryReader):
        proposals = 0

        def _proposal(self, selected):
            self.proposals += 1
            call = super()._proposal(selected)
            return corrupt(call) if self.proposals == 1 else call

    calls = []
    result = invoke(
        TamperedReader(ScriptedModel([retrieve(), answer()]), lookup_tools(calls))
    )
    assert calls == [("lookup", "first", 0)]
    assert result["num_model_calls"] == 2  # No second planner after the rejected call.
    assert len(result["rejections"]) == 1
    assert result["rejections"][0]["phase"] == "execution"
    assert result["rejections"][0]["decision_id"] == result["executions"][0]["id"]
    assert [d["tool"] for d in result["decisions"] if d["action"] == "retrieve"] == [
        "lookup"
    ]


def test_equivalent_omitted_defaults_match_committed_arguments():
    class DefaultsReader(PlannedMemoryReader):
        def _proposal(self, selected):
            call = super()._proposal(selected)
            call["args"].pop("attempt")
            return call

    calls = []
    result = invoke(
        DefaultsReader(ScriptedModel([retrieve(), answer()]), lookup_tools(calls))
    )
    assert calls == [("lookup", "first", 0)] and result["rejections"] == []


def test_persistent_wrong_call_terminates_without_replanning_or_executing():
    class WrongReader(PlannedMemoryReader):
        def _proposal(self, selected):
            return {**super()._proposal(selected), "name": "other"}

    calls = []
    result = invoke(WrongReader(ScriptedModel([retrieve()]), lookup_tools(calls)))
    assert calls == [] and result["num_tool_calls"] == 0
    assert result["num_model_calls"] == 1 and len(result["rejections"]) == 3
    assert result["stop_reason"] == "tool_call_repair_exhausted"
    assert result["answer"] == "I don't know."


@pytest.mark.parametrize(
    "bad", [RuntimeError("model offline"), {"answer": "   "}, {"unknown": 1}]
)
def test_persistent_bad_output_has_finite_repair_budget(bad):
    calls = []
    result = invoke(PlannedMemoryReader(ScriptedModel([bad] * 3), lookup_tools(calls)))
    assert calls == [] and result["num_model_calls"] == 3
    assert len(result["rejections"]) == 3 and result["answer"] == "I don't know."


def test_changed_registered_tool_identity_cannot_execute():
    class ChangedRegistryReader(PlannedMemoryReader):
        def _proposal(self, selected):
            call = super()._proposal(selected)
            self._tools[selected.tool_name].name = "other"
            return call

    calls = []
    result = invoke(
        ChangedRegistryReader(ScriptedModel([retrieve()]), lookup_tools(calls))
    )
    assert calls == [] and result["num_tool_calls"] == 0
    assert result["num_model_calls"] == 1 and len(result["rejections"]) == 3
    assert result["stop_reason"] == "tool_call_repair_exhausted"
    assert all(
        "identity changed" in rejection["reason"] for rejection in result["rejections"]
    )


def test_tool_start_callback_failures_consume_slots_without_body_retries():
    attempts = []

    class FailingCallback(BaseCallbackHandler):
        raise_error = True

        def on_tool_start(self, serialized, input_str, **kwargs):
            attempts.append(serialized["name"])
            raise RuntimeError("callback unavailable")

    calls = []
    model = ScriptedModel(
        [retrieve(str(n)) for n in range(5)] + [{"answer": "I don't know."}]
    )
    result = invoke(
        PlannedMemoryReader(model, lookup_tools(calls)), callbacks=[FailingCallback()]
    )
    assert calls == [] and attempts == ["lookup"] * 5
    assert result["num_tool_calls"] == 5 and result["num_model_calls"] == 6
    assert all(entry["status"] == "error" for entry in result["executions"])
    assert "callback unavailable" in model.requests[1][1][-1].content
    assert result["stop_reason"] == "tool_budget_exhausted"


def test_maximum_repairs_still_fit_the_public_graph_limit():
    class RepairingReader(PlannedMemoryReader):
        def __init__(self, *args):
            self.proposals = {}
            super().__init__(*args)

        def _proposal(self, selected):
            attempts = self.proposals.get(selected.decision_id, 0) + 1
            self.proposals[selected.decision_id] = attempts
            call = super()._proposal(selected)
            return {**call, "name": "other"} if attempts < 3 else call

    calls = []
    responses = []
    for n in range(5):
        responses.extend([{"unknown": 1}, {"unknown": 1}, retrieve(str(n))])
    responses.extend([{"unknown": 1}, {"unknown": 1}, {"answer": "done"}])
    result = invoke(
        RepairingReader(ScriptedModel(responses), lookup_tools(calls)),
        recursion_limit=60,
    )
    assert (
        result["answer"] == "done" and result["stop_reason"] == "tool_budget_exhausted"
    )
    assert result["num_model_calls"] == 18 and len(result["rejections"]) == 22
    assert len(calls) == result["num_tool_calls"] == 5


def test_persistent_sixth_call_requests_end_with_a_terminal_fallback():
    calls = []
    model = ScriptedModel([retrieve(str(n)) for n in range(8)])
    result = invoke(PlannedMemoryReader(model, lookup_tools(calls)))
    assert len(calls) == result["num_tool_calls"] == 5
    assert result["num_model_calls"] == 8 and len(result["rejections"]) == 3
    assert result["stop_reason"] == "decision_repair_exhausted"
    assert result["messages"][-1].content == result["answer"] == "I don't know."


def test_controller_failure_after_commitment_does_not_replan():
    class BrokenReader(PlannedMemoryReader):
        def _proposal(self, selected):
            raise RuntimeError("controller failed after commitment")

    calls = []
    result = invoke(BrokenReader(ScriptedModel([retrieve()]), lookup_tools(calls)))
    assert calls == [] and result["num_model_calls"] == 1
    assert len(result["decisions"]) == 1 and result["num_tool_calls"] == 0
    assert (
        result["stop_reason"] == "graph_error" and result["answer"] == "I don't know."
    )


def test_bounded_observations_mark_incomplete_evidence():
    calls = []
    model = ScriptedModel([retrieve(), answer()])
    result = invoke(
        PlannedMemoryReader(
            model, lookup_tools(calls, output="x" * (MAX_OBSERVATION_CHARS + 1))
        )
    )
    entry = result["executions"][0]
    assert entry["truncated"] and "evidence is incomplete" in entry["output"]
    assert len(entry["output"]) < MAX_OBSERVATION_CHARS + 200
    assert "Output truncated" in model.requests[1][1][-1].content


def test_unexpected_node_failure_keeps_attempt_ledger_and_never_restarts():
    class FailingReader(PlannedMemoryReader):
        def _execute(
            self, state: _State, config: RunnableConfig, runtime: Runtime[_Run]
        ) -> dict:
            super()._execute(state, config, runtime)
            raise RuntimeError("failure before returned state update")

    calls = []
    result = invoke(FailingReader(ScriptedModel([retrieve()]), lookup_tools(calls)))
    assert len(calls) == result["num_tool_calls"] == 1
    assert result["executions"][0]["status"] == "success"
    assert (
        result["stop_reason"] == "graph_error" and result["answer"] == "I don't know."
    )


def test_consumed_commitment_cannot_execute_twice():
    class ReplayReader(PlannedMemoryReader):
        def _execute(
            self, state: _State, config: RunnableConfig, runtime: Runtime[_Run]
        ) -> dict:
            update = super()._execute(state, config, runtime)
            if update["next"] == "decide":
                update.update(call=state["call"], next="execute")
            return update

    calls = []
    result = invoke(ReplayReader(ScriptedModel([retrieve()]), lookup_tools(calls)))
    assert len(calls) == result["num_tool_calls"] == 1
    assert result["stop_reason"] == "execution_not_authorized"


@pytest.mark.parametrize("limit, expected", [(1, 0), (2, 1)])
def test_low_recursion_limit_finalizes_without_retrieval_recovery(limit, expected):
    calls = []
    result = invoke(
        PlannedMemoryReader(ScriptedModel([retrieve()]), lookup_tools(calls)),
        recursion_limit=limit,
    )
    assert len(calls) == result["num_tool_calls"] == expected
    assert (
        result["stop_reason"] == "recursion_limit"
        and result["answer"] == "I don't know."
    )


def test_input_cannot_supply_budget_or_commitment():
    calls = []
    model = ScriptedModel([retrieve(str(n)) for n in range(5)] + [{"answer": "done"}])
    reader = PlannedMemoryReader(model, lookup_tools(calls))
    result = reader.invoke(
        {
            "messages": [HumanMessage(content="query")],
            "executions": [],
            "remaining": 1000,
            "selection": "other",
            "next": "execute",
        }
    )
    assert len(calls) == result["num_tool_calls"] == 5


def test_concurrent_queries_have_isolated_accounting_and_decisions():
    barrier = Barrier(2)
    calls = []

    def respond(messages, schema):
        query = messages[1].content
        observations = [
            m for m in messages if m.content.startswith("RETRIEVAL OBSERVATION")
        ]
        if not observations:
            barrier.wait(timeout=10)
        if schema is FinalAnswer:
            return {"answer": query}
        return retrieve(f"{query}:{len(observations)}")

    reader = PlannedMemoryReader(ScriptedModel(respond), lookup_tools(calls))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda query: invoke(reader, query), ["a", "b"]))
    assert [result["num_tool_calls"] for result in results] == [5, 5]
    assert [result["answer"] for result in results] == ["a", "b"]
    assert len(calls) == 10
    for query, result in zip(["a", "b"], results):
        assert all(
            call["args"]["topic"].startswith(query + ":")
            for call in result["tool_calls"]
        )


def agent_for(reader):
    agent = AgenticMemoryAgent.__new__(AgenticMemoryAgent)
    agent.chat_agent = reader
    agent.recursion_limit = 60
    agent._pending = []
    return agent


def test_all_public_entry_points_use_same_enforced_reader():
    calls = []
    responses = []
    for _ in range(3):
        responses.extend([retrieve(str(n)) for n in range(5)] + [{"answer": "done"}])
    agent = agent_for(
        PlannedMemoryReader(ScriptedModel(responses), lookup_tools(calls))
    )
    assert agent.chat("query") == "done"
    assert agent.chat_with_trace("query")["num_tool_calls"] == 5
    assert agent.converse("query") == "done"
    assert len(calls) == 15 and agent.pending_turns == 2


def test_live_history_is_preserved_while_budget_resets_each_turn():
    calls = []
    responses = []
    for text in ["first answer", "second answer"]:
        responses.extend([retrieve(str(n)) for n in range(5)] + [{"answer": text}])
    model = ScriptedModel(responses)
    agent = agent_for(PlannedMemoryReader(model, lookup_tools(calls)))
    assert agent.converse("earlier detail") == "first answer"
    assert agent.converse("follow-up") == "second answer"
    second_start = model.requests[6][1]
    assert [m.content for m in second_start[1:]] == [
        "earlier detail",
        "first answer",
        "follow-up",
    ]
    assert "REMAINING RETRIEVAL CALLS: 5" in second_start[0].content
    assert len(calls) == 10 and agent.pending_turns == 4
