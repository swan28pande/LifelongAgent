"""Hierarchical generation, persistence, and retrieval using real offline storage."""

import pytest

from memory_v4.summarizer import Summarizer
from memory_v4.tools import build_read_tools


def test_layer_search_preserves_each_level_without_reembedding(memory_store):
    levels = {"week": 2, "month": 2, "year": 1, "lifetime": 1, "distilled": 2}
    for level in levels:
        for i, topic in enumerate(("camping", "cooking", "career")):
            memory_store.save_summary(f"{level}:{i}:alice", "Example", topic)
    # Many close weekly matches must not displace the less common layers.
    for i in range(10):
        memory_store.save_summary(f"week:extra-{i}:alice", "Example", "camping")
    embeddings = memory_store.embeddings
    document_calls = len(embeddings.document_calls)
    results = memory_store.search_summaries_by_layer("camping")
    assert {level: len(docs) for level, docs in results.items()} == levels
    assert all("camping" in docs[0].page_content for docs in results.values())
    assert len(embeddings.document_calls) == document_calls
    assert embeddings.query_calls == ["camping"]


def test_layer_search_sees_updates_and_reloaded_summaries(memory_store):
    memory_store.save_summary("week:2026-W10:alice", "Week", "cooking")
    memory_store.search_summaries_by_layer("camping")
    memory_store.save_summary("week:2026-W10:alice", "Week", "camping")
    from memory_v4.store import MemoryStore

    reloaded = MemoryStore(memory_store.base_dir)
    docs = reloaded.search_summaries_by_layer("camping")["week"]
    assert len(docs) == 1 and "camping" in docs[0].page_content
    assert "cooking" not in docs[0].page_content


def test_empty_and_disabled_summary_layers(memory_store):
    assert memory_store.search_summaries_by_layer("camping") == {}
    memory_store.save_summary("week:2026-W10:alice", "Week", "camping")
    assert memory_store.search_summaries_by_layer("camping", k_week=0) == {}
    lookup = next(
        t
        for t in build_read_tools(memory_store)
        if t.name == "semantic_retrieve_memory"
    )
    memory_store._summary_store.delete(list(memory_store._summary_store.docstore._dict))
    assert "no summaries available" in lookup.invoke({"query": "camping"})


def test_ingestion_and_summary_chain_are_per_speaker_and_incremental(
    agent_factory,
    conversations,
    summary_model,
):
    agent = agent_factory()
    for date in ("2026-03-02", "2026-03-03"):
        report = agent.ingest(date, conversations, update_summaries=True)
        assert report.added == 3 and report.errors == []
    docs = list(agent.store._summary_store.docstore._dict.values())
    assert len(docs) == 8
    assert {doc.metadata["speaker"] for doc in docs} == {"alice", "bob"}
    for speaker in ("alice", "bob"):
        week = agent.store.get_summary(f"week:2026-W10:{speaker}")
        month = agent.store.get_summary(f"month:2026-03:{speaker}")
        year = agent.store.get_summary(f"year:2026:{speaker}")
        assert "Weekly narrative." in week and week.count("hobby: camping") == 1
        assert "Monthly narrative." in month and "Key facts: hobby: camping" in month
        assert "Facts: hobby: camping" in year
        assert "Camping" in agent.store.get_lifetime_summary(speaker)
    assert "camp trip" in agent.store.get_summary("week:2026-W10:alice")
    assert "camp trip" not in agent.store.get_summary("week:2026-W10:bob")
    requests = len(summary_model.requests)
    agent.summarizer.update_after_ingest("2026-03-03")
    assert len(summary_model.requests) == requests
    assert len(agent.store.query_memories(type="fact")) == 4
    chunk = agent.store.get_conversations_by_date_range("2026-03-02", "2026-03-02")[0]
    assert chunk.metadata["speaker"] == "alice,bob"


def test_batch_defaults_and_optional_distillation(agent_factory, conversations):
    agent = agent_factory()
    agent.ingest("2026-03-02", conversations)
    assert agent.store._summary_store is None
    agent.build_summaries()
    assert not any(
        doc.metadata["identifier"].startswith("distilled:")
        for doc in agent.store._summary_store.docstore._dict.values()
    )
    agent.build_summaries(distill=True)
    themes = {"relationships", "identity", "patterns", "timeline"}
    distilled = [
        doc
        for doc in agent.store._summary_store.docstore._dict.values()
        if doc.metadata["identifier"].startswith("distilled:")
    ]
    assert len(distilled) == 8
    assert {doc.metadata["theme"] for doc in distilled} == themes
    updated = agent.distill_knowledge()
    assert set(updated) == {"alice", "bob"}
    assert all(set(documents) == themes for documents in updated.values())
    assert all(
        text.startswith("Updated")
        for documents in updated.values()
        for text in documents.values()
    )
    assert len(agent.store._summary_store.docstore._dict) == 16


def test_flush_updates_hierarchy_by_default(agent_factory):
    agent = agent_factory()
    agent._pending = [{"speaker": "Alice", "text": "Camping is my hobby."}]
    assert agent.flush("2026-03-02").errors == []
    assert agent.pending_turns == 0
    assert agent.store.get_lifetime_summary("alice")
    assert agent.flush("2026-03-02") is None


@pytest.mark.parametrize(
    "date, last_week", [("2026-03", "2026-W14"), ("2016-02", "2016-W09")]
)
def test_partial_month_includes_its_actual_last_week(
    memory_store, summary_model, monkeypatch, date, last_week
):
    memory_store.add_memories(
        [
            {
                "entity": "hobby",
                "content": "camping",
                "speaker": "alice",
                "date": date,
            }
        ],
        source_date=date,
    )
    summarizer = Summarizer(memory_store, llm=summary_model.llm)
    weeks = []
    monkeypatch.setattr(summarizer, "weekly", lambda week, **kwargs: weeks.append(week))
    for name in ("monthly", "yearly", "lifetime"):
        monkeypatch.setattr(summarizer, name, lambda *args, **kwargs: None)
    summarizer.run()
    assert weeks[-1] == last_week


@pytest.mark.parametrize("date", ["2024", "2024-02"])
def test_lifetime_generation_accepts_partial_dates(memory_store, summary_model, date):
    memory_store.add_memories(
        [{"entity": "hobby", "content": "camping", "speaker": "alice", "date": date}],
        source_date=date,
    )
    memory_store.save_summary("year:2024:alice", "Year", "Camping in 2024.")
    summarizer = Summarizer(memory_store, llm=summary_model.llm)
    assert "Camping" in summarizer.lifetime("alice")
