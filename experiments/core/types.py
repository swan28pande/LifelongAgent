"""The common format every benchmark is loaded into and every method consumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    speaker: str      # "user" / "assistant", or a person's name for two-person benchmarks
    text: str


@dataclass
class Session:
    id: str
    date: str         # ISO date (YYYY-MM-DD) the session took place
    turns: list[Turn]
    time: Optional[str] = None   # time of day, when the benchmark has it


@dataclass
class Question:
    id: str
    question: str
    answer: str
    accept: list[str]
    answer_type: str             # value | date | free_text | yes_no | abstain
    grading: str                 # exact | date_exact | llm_judge | locomo_f1
    category: str                # question type used for the per-category breakdown
    asked_on: Optional[str] = None   # ISO date the question is asked, if the benchmark defines one
    meta: dict = field(default_factory=dict)


@dataclass
class Instance:
    """One memory to build: a conversation history plus the questions asked of it."""
    id: str
    sessions: list[Session]
    questions: list[Question]
    user: Optional[str] = None


@dataclass
class Answer:
    text: str
    meta: dict = field(default_factory=dict)


def prompt_for(q: Question) -> str:
    """The question text every method receives, identical across methods."""
    return f"(Asked on {q.asked_on}) {q.question}" if q.asked_on else q.question
