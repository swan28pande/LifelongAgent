"""Control: all observed history in the prompt, restored from the user's store."""

from experiments import config
from experiments.core.method import MemoryMethod
from experiments.core.types import Answer, Question, Session
from .._common import answer_from_context, format_session, make_llm
from ..store import Sessions


class FullContext(MemoryMethod):
    name = "full_context"
    persistent = True

    def __init__(self, store_dir, instance, model, usage):
        super().__init__(store_dir, instance, model, usage)
        self.data = Sessions(store_dir)
        self.sessions = [format_session(s) for s in self.data.all()]
        self.context = ""
        self.dropped = 0
        self.pending = False
        self.llm = make_llm(model, usage)
        self.finalize()

    def ingest(self, session: Session) -> dict:
        added = self.data.add(session)
        if added:
            self.sessions.append(format_session(session))
            self.pending = True
        return {'added': added}

    def finalize(self) -> None:
        budget = config.FULL_CONTEXT_MAX_TOKENS * 4
        context = "\n\n".join(self.sessions)
        if len(context) > budget:
            raise ValueError('Full observed history exceeds FULL_CONTEXT_MAX_TOKENS '
                             '(four-character estimate); choose a suitable context budget/model')
        self.context = context
        self.pending = False

    def answer(self, q: Question) -> Answer:
        if self.pending:
            raise RuntimeError('Finalize observed sessions before answering')
        return answer_from_context(self.llm, q, self.context,
                                   {'sessions_dropped': 0, 'sessions': len(self.sessions),
                                    'estimated_context_tokens': (len(self.context) + 3) // 4})
