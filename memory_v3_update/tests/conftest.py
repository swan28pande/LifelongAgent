"""Keep controlled reader tests offline, even when .env enables tracing."""

import json
import os
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

for variable in (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
):
    os.environ[variable] = "false"


@pytest.fixture
def memory_store(tmp_path, monkeypatch):
    """Real SQLite/FAISS storage, with small deterministic offline embeddings."""
    from langchain_core.embeddings import Embeddings
    from memory_v3_update import store

    class OfflineEmbeddings(Embeddings):
        def __init__(self):
            self.document_calls = []
            self.query_calls = []

        @staticmethod
        def vector(text):
            return [
                float(text.lower().count(word))
                for word in ("camping", "cooking", "career")
            ]

        def embed_documents(self, texts):
            self.document_calls.append(list(texts))
            return [self.vector(text) for text in texts]

        def embed_query(self, text):
            self.query_calls.append(text)
            return self.vector(text)

    embeddings = OfflineEmbeddings()
    monkeypatch.setattr(store, "PrefixedEmbeddings", lambda **kwargs: embeddings)
    return store.MemoryStore(str(tmp_path))


@pytest.fixture
def summary_model():
    """Exercise the real prompt/parser pipelines with predictable JSON responses."""
    from langchain_core.runnables import RunnableLambda

    requests = []

    def respond(prompt):
        messages = prompt.to_messages()
        human = messages[-1].content
        requests.append(messages)
        if "TRANSCRIPT:" in human:
            date = re.search(r"CONVERSATION DATE: ([^\n]+)", human).group(1)
            result = {
                "facts": [
                    {
                        "entity": "hobby",
                        "content": "camping",
                        "speaker": speaker,
                        "date": date,
                    }
                    for speaker in ("alice", "bob")
                ],
                "events": [
                    {
                        "entity": "trip",
                        "content": "camp trip",
                        "speaker": "alice",
                        "date": date,
                    }
                ],
                "preferences": [],
            }
        elif human.startswith("Week:"):
            result = {"narrative": "Weekly narrative."}
        elif human.startswith("Month:"):
            result = {
                "narrative": "Monthly narrative.",
                "facts": ["hobby: camping"],
                "events": [],
            }
        elif human.startswith("Year:"):
            result = {"facts": ["hobby: camping"], "events": []}
        elif "YEARLY SUMMARIES:" in human:
            result = {
                "title": "Lifetime profile",
                "summary": "Camping is an enduring hobby.",
            }
        elif "ALL SUMMARIES:" in human:
            theme = re.search(r"Theme: ([^\n]+)", human).group(1)
            prefix = "Updated" if "CURRENT DOCUMENT:" in human else "Created"
            result = {"document": f"{prefix} {theme}: camping (since 2026-03)."}
        else:
            raise AssertionError(f"Unexpected model request: {human}")
        return json.dumps(result)

    return SimpleNamespace(llm=RunnableLambda(respond), requests=requests)


@pytest.fixture
def agent_factory(memory_store, summary_model, monkeypatch):
    from memory_v3_update import agent

    def create(chat_model=None):
        reader_model = chat_model if chat_model is not None else Mock()
        monkeypatch.setattr(
            agent,
            "_make_llm",
            lambda model, temperature=0.0, callbacks=None, **options: (
                reader_model if temperature == 0.3 else summary_model.llm
            ),
        )
        return agent.AgenticMemoryAgent(memory_store.base_dir)

    return create


@pytest.fixture
def conversations():
    return [
        {
            "time_of_day": "Morning",
            "turns": [
                {
                    "speaker": "Alice",
                    "text": "Camping helps me relax. I went on a camp trip.",
                },
                {"speaker": "Bob", "text": "Camping is my hobby too."},
            ],
        }
    ]
