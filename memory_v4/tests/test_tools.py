"""Read-tool schema and coverage checks without embeddings or a persistent store."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from memory_v4.tools import build_read_tools


class StoreSpy:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.calls = []

    def get_all_entities(self):
        return ["coffee"]

    def get_all_speakers(self):
        return ["user"]

    def query_memories(self, **arguments):
        self.calls.append(arguments)
        return self.rows[: arguments["limit"]]

    def get_summary(self, identifier):
        self.calls.append(identifier)
        return "summary"

    def get_lifetime_summary(self, speaker):
        self.calls.append(f"lifetime:{speaker}")
        return "lifetime summary"


@pytest.fixture
def registry():
    return {tool.name: tool for tool in build_read_tools(StoreSpy())}


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("search_memories", {"limit": -1}),
        ("search_memories", {"limit": 0}),
        ("search_memories", {"limit": 201}),
        ("search_memories", {"limit": True}),
        ("search_memories", {"type": "unknown"}),
        ("search_memories", {"start_date": "2026-2-01"}),
        ("search_memories", {"start_date": "2026-02-30"}),
        ("search_memories", {"start_date": "2026-03-01", "end_date": "2026-02-01"}),
        ("semantic_search_conversations", {"query": "topic", "k": 21}),
        ("semantic_search_conversations", {"query": "topic", "k": 0}),
        ("semantic_retrieve_memory", {"query": True}),
        ("semantic_retrieve_memory", {"query": "  "}),
        ("read_conversations_on", {"date": "2026-01-01", "days_around": -1}),
        ("read_conversations_on", {"date": "2026-01-01", "days_around": 8}),
        ("read_conversations_on", {"date": "tomorrow"}),
        ("get_summary", {"level": "day", "identifier": "2026-01-01"}),
        ("get_summary", {"level": "month", "identifier": "2026-13"}),
        ("get_summary", {"level": "week", "identifier": "2021-W53"}),
        ("get_summary", {"level": "year", "identifier": ""}),
        ("get_summary", {"level": "lifetime", "identifier": "2026"}),
    ],
)
def test_semantic_argument_errors_are_rejected_by_schema(registry, name, arguments):
    with pytest.raises(ValidationError):
        registry[name].get_input_schema().model_validate(arguments, strict=True)


def test_name_and_type_normalization_reaches_the_actual_query():
    store = StoreSpy()
    lookup = {tool.name: tool for tool in build_read_tools(store)}["search_memories"]
    lookup.invoke({"entity": " Coffee ", "speaker": " User ", "type": " FACT "})
    assert store.calls == [
        {
            "entity": "coffee",
            "speaker": "user",
            "type": "fact",
            "start_date": None,
            "end_date": None,
            "limit": 101,
        }
    ]


def test_sql_truncation_is_detected_with_one_extra_row():
    rows = [
        {
            "id": i,
            "date": "2026-01-01",
            "entity": "coffee",
            "content": str(i),
            "speaker": "user",
        }
        for i in range(3)
    ]
    store = StoreSpy(rows)
    lookup = {tool.name: tool for tool in build_read_tools(store)}["search_memories"]
    text = lookup.invoke({"limit": 2})
    assert "id=0" in text and "id=1" in text and "id=2" not in text
    assert "Results truncated" in text and store.calls[0]["limit"] == 3


def test_complete_sql_result_is_not_marked_truncated():
    row = {
        "id": 1,
        "date": "2026-01-01",
        "entity": "coffee",
        "content": "tea",
        "speaker": "user",
    }
    store = StoreSpy([row])
    lookup = {tool.name: tool for tool in build_read_tools(store)}["search_memories"]
    assert "truncated" not in lookup.invoke({"limit": 2})


def test_summary_period_and_speaker_are_canonicalized():
    store = StoreSpy()
    summary = {tool.name: tool for tool in build_read_tools(store)}["get_summary"]
    assert (
        summary.invoke(
            {"level": " MONTH ", "identifier": " 2026-03 ", "speaker": " User "}
        )
        == "summary"
    )
    assert store.calls == ["month:2026-03:user"]


def test_scalar_summary_input_keeps_normalization():
    store = StoreSpy()
    lookup = next(
        tool for tool in build_read_tools(store) if tool.name == "get_summary"
    )
    assert lookup.invoke(" LIFETIME ") == "lifetime summary"
    assert store.calls == ["lifetime:user"]


class ConversationStoreSpy:
    def __init__(self):
        self.calls = []

    def get_conversations_by_date_range(self, start, end):
        self.calls.append((start, end))
        return [SimpleNamespace(page_content="stored conversation")]


@pytest.mark.parametrize(
    "date, window, expected",
    [
        ("2026-03-01", 0, ("2026-03-01", "2026-03-01")),
        ("2026-03-01", 2, ("2026-02-27", "2026-03-03")),
        ("0001-01-01", 0, ("0001-01-01", "0001-01-01")),
        ("0999-12-31", 1, ("0999-12-30", "1000-01-01")),
    ],
)
def test_date_lookup_preserves_iso_dates_and_window(date, window, expected):
    store = ConversationStoreSpy()
    lookup = next(
        tool for tool in build_read_tools(store) if tool.name == "read_conversations_on"
    )
    assert lookup.invoke({"date": date, "days_around": window}) == "stored conversation"
    assert store.calls == [expected]


@pytest.mark.parametrize(
    "arguments",
    [
        {"date": "tomorrow"},
        {"date": "2026-03-01", "days_around": -1},
        {"date": "2026-03-01", "days_around": 8},
        {"date": "2026-03-01", "days_around": True},
    ],
)
def test_invalid_date_lookup_arguments_never_reach_store(arguments):
    store = ConversationStoreSpy()
    lookup = next(
        tool for tool in build_read_tools(store) if tool.name == "read_conversations_on"
    )
    with pytest.raises(ValidationError):
        lookup.invoke(arguments)
    assert store.calls == []


def test_scalar_date_input_keeps_normalization():
    store = ConversationStoreSpy()
    lookup = next(
        tool for tool in build_read_tools(store) if tool.name == "read_conversations_on"
    )
    assert lookup.invoke(" 0001-01-01 ") == "stored conversation"
    assert store.calls == [("0001-01-01", "0001-01-01")]
