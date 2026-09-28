"""The researcher's method: memory_v3 (dated SQL records + hierarchical summaries + agentic retrieval)."""

from ..core.method import MemoryMethod
from ..core.types import Answer, Question, Session, prompt_for


class OursV3(MemoryMethod):
    name = "ours_v3"
    persistent = True

    def __init__(self, store_dir, instance, model, usage):
        super().__init__(store_dir, instance, model, usage)
        from memory_v3.agent import AgenticMemoryAgent
        self.agent = AgenticMemoryAgent(base_dir=str(store_dir), model=model, callbacks=[usage])

    def ingest(self, session: Session) -> dict:
        report = self.agent.ingest(session.date, [{
            "time_of_day": session.time or "All Day",
            "turns": [{"speaker": t.speaker, "text": t.text} for t in session.turns],
        }])
        if report.errors:
            raise RuntimeError(report.errors[0])
        return {"added": report.added, "type_counts": report.type_counts}

    def finalize(self) -> None:
        self.agent.build_summaries(force=True)

    def answer(self, q: Question) -> Answer:
        trace = self.agent.chat_with_trace(prompt_for(q))
        return Answer(trace["answer"], {"tool_calls": trace["tool_calls"]})
