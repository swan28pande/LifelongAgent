"""Control: the whole history in the prompt (oldest sessions dropped past the token budget)."""

from .. import config
from ..core.method import MemoryMethod
from ..core.types import Answer, Question, Session
from ._common import answer_from_context, format_session, make_llm


class FullContext(MemoryMethod):
    name = "full_context"

    def __init__(self, store_dir, instance, model, usage):
        super().__init__(store_dir, instance, model, usage)
        self.sessions: list[str] = []
        self.context = ""
        self.dropped = 0
        self.llm = make_llm(model, usage)

    def ingest(self, session: Session) -> dict:
        self.sessions.append(format_session(session))
        return {}

    def finalize(self) -> None:
        budget = config.FULL_CONTEXT_MAX_TOKENS * 4
        kept, size = [], 0
        for text in reversed(self.sessions):
            if size + len(text) > budget:
                break
            kept.append(text)
            size += len(text)
        self.dropped = len(self.sessions) - len(kept)
        self.context = "\n\n".join(reversed(kept))

    def answer(self, q: Question) -> Answer:
        return answer_from_context(self.llm, q, self.context, {"sessions_dropped": self.dropped})
