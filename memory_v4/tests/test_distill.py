"""Continual knowledge updates through real prompts, SQLite, and FAISS storage."""

import json

import pytest
from langchain_core.runnables import RunnableLambda

from memory_v4.distill import THEMES, KnowledgeDistiller
from memory_v4.store import MemoryStore


def save_hierarchy(store):
    for identifier, content in (
        ("week:2026-W10:alice", "Current week: camping changed."),
        ("month:2026-03:alice", "Current month: a new camping routine."),
        ("week:2026-W09:alice", "Old week: cooking."),
        ("month:2026-02:alice", "Old month: cooking."),
        ("year:2026:alice", "Year history: career and cooking."),
        ("lifetime:alice", "Lifetime history: career and cooking."),
        ("week:2026-W10:bob", "Bob's private career history."),
        ("month:2026-03:bob", "Bob's private monthly history."),
    ):
        store.save_summary(identifier, "Summary", content)


def save_documents(store, themes=THEMES):
    for theme in themes:
        store.save_summary(
            f"distilled:{theme}:alice",
            f"Distilled {theme.title()} — alice",
            f"Historical {theme} (2026-01 – 2026-02). {{reason: cooking}}",
            {"speaker": "alice", "theme": theme},
        )


def test_initial_update_creates_v4_themes_from_full_hierarchy(
    memory_store, summary_model
):
    save_hierarchy(memory_store)
    distiller = KnowledgeDistiller(memory_store, summary_model.llm)
    result = distiller.update_after_ingest("alice", "2026-03-02")
    assert set(result) == {"relationships", "identity", "patterns", "timeline"}
    assert len(summary_model.requests) == 4
    for request in summary_model.requests:
        human = request[-1].content
        assert "CURRENT DOCUMENT:" not in human
        assert "Old month: cooking." in human
        assert "Year history:" in human and "Lifetime history:" in human
        assert "Bob's private" not in human
    assert all("Created" in document for document in result.values())


def test_existing_documents_merge_only_affected_summaries_and_persist(memory_store):
    save_hierarchy(memory_store)
    save_documents(memory_store)
    original_count = memory_store._summary_store.index.ntotal
    requests = []

    def merge(prompt):
        human = prompt.to_messages()[-1].content
        requests.append(human)
        current = human.split("CURRENT DOCUMENT:\n", 1)[1].split(
            "\n\nALL SUMMARIES:\n", 1
        )[0]
        return json.dumps(
            {"document": current + "\nNew camping routine (since 2026-03-02)."}
        )

    distiller = KnowledgeDistiller(memory_store, RunnableLambda(merge))
    result = distiller.update_after_ingest("alice", "2026-03-02")
    assert len(requests) == 4
    for human in requests:
        assert "=== WEEK 2026-W10 ===" in human
        assert "=== MONTH 2026-03 ===" in human
        assert "Current week: camping changed." in human
        assert "Current month: a new camping routine." in human
        assert all(
            marker not in human
            for marker in (
                "Old week:",
                "Old month:",
                "Year history:",
                "Lifetime history:",
                "Bob's private",
            )
        )
    reloaded = MemoryStore(memory_store.base_dir)
    assert reloaded._summary_store.index.ntotal == original_count
    for theme, document in result.items():
        assert (
            f"Historical {theme} (2026-01 – 2026-02). {{reason: cooking}}" in document
        )
        assert "New camping routine (since 2026-03-02)." in document
        assert document in reloaded.get_summary(f"distilled:{theme}:alice")


def test_missing_theme_uses_full_history_without_rebuilding_existing_themes(
    memory_store, summary_model
):
    save_hierarchy(memory_store)
    save_documents(memory_store, themes=("relationships", "identity", "patterns"))
    distiller = KnowledgeDistiller(memory_store, summary_model.llm)
    result = distiller.update_after_ingest("alice", "2026-03-02")
    assert len(result) == len(summary_model.requests) == 4
    assert result["timeline"].startswith("Created")
    for theme in ("relationships", "identity", "patterns"):
        assert result[theme].startswith("Updated")
    for request in summary_model.requests:
        human = request[-1].content
        if "Theme: timeline\n" in human:
            assert "Old month: cooking." in human
            assert "Year history:" in human and "Lifetime history:" in human
        else:
            assert "Current week: camping changed." in human
            assert "Old month:" not in human and "Lifetime history:" not in human


def test_no_affected_summaries_leaves_existing_documents_untouched(
    memory_store, summary_model
):
    save_documents(memory_store)
    originals = {
        theme: memory_store.get_summary(f"distilled:{theme}:alice") for theme in THEMES
    }
    distiller = KnowledgeDistiller(memory_store, summary_model.llm)
    assert distiller.update_after_ingest("alice", "2026-03-02") == {}
    assert summary_model.requests == []
    assert all(
        memory_store.get_summary(f"distilled:{theme}:alice") == text
        for theme, text in originals.items()
    )


def test_initial_update_without_summaries_does_not_call_model(
    memory_store, summary_model
):
    distiller = KnowledgeDistiller(memory_store, summary_model.llm)
    assert distiller.update_after_ingest("alice", "2026-03-02") == {}
    assert summary_model.requests == []
    assert memory_store._summary_store is None


@pytest.mark.parametrize(
    "date, week", [("2021-01-01", "2020-W53"), ("2018-12-31", "2019-W01")]
)
def test_incremental_update_uses_iso_week_year(memory_store, summary_model, date, week):
    save_documents(memory_store)
    memory_store.save_summary(f"week:{week}:alice", "Week", "Year boundary: camping.")
    distiller = KnowledgeDistiller(memory_store, summary_model.llm)
    assert len(distiller.update_after_ingest("alice", date)) == 4
    assert all(
        f"=== WEEK {week} ===" in request[-1].content
        and "Year boundary: camping." in request[-1].content
        for request in summary_model.requests
    )


def test_failed_incremental_merge_preserves_saved_knowledge(memory_store):
    save_hierarchy(memory_store)
    save_documents(memory_store)
    originals = {
        theme: memory_store.get_summary(f"distilled:{theme}:alice") for theme in THEMES
    }

    def fail(prompt):
        raise RuntimeError("Provider unavailable")

    distiller = KnowledgeDistiller(memory_store, RunnableLambda(fail))
    assert len(distiller.update_after_ingest("alice", "2026-03-02")) == 4
    assert all(
        memory_store.get_summary(f"distilled:{theme}:alice") == text
        for theme, text in originals.items()
    )
