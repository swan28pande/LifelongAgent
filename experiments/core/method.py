"""The interface every memory method implements."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from .types import Answer, Instance, Question, Session


class UsageCounter(BaseCallbackHandler):
    """Counts tokens of every LangChain LLM call a method makes through its own models."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls = self.input_tokens = self.output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        with self.lock:
            for gens in response.generations:
                for g in gens:
                    meta = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                    self.calls += 1
                    self.input_tokens += meta.get("input_tokens", 0)
                    self.output_tokens += meta.get("output_tokens", 0)

    def snapshot(self) -> dict:
        with self.lock:
            return {"calls": self.calls, "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens}


class MemoryMethod(ABC):
    """Build a memory from sessions in date order, then answer questions from it.

    persistent: the store survives a restart, so ingestion can resume where it stopped.
    Otherwise the runner re-ingests every session on each run.
    """

    name: str = ""
    persistent: bool = False

    def __init__(self, store_dir: Path, instance: Instance, model: str, usage: UsageCounter):
        self.store_dir = store_dir
        self.instance = instance
        self.model = model
        self.usage = usage

    @abstractmethod
    def ingest(self, session: Session) -> dict:
        """Add one session. Raise on failure; return any stats worth logging."""

    def finalize(self) -> None:
        """Called once after the last session (e.g. to build summaries)."""

    @abstractmethod
    def answer(self, question: Question) -> Answer:
        """Answer one question. Must be safe to call from several threads."""
