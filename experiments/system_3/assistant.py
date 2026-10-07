"""Live assistants over a memory method: reply turn by turn, write the day to memory at its end."""

from __future__ import annotations

import datetime as dt

from langchain_core.messages import HumanMessage, SystemMessage

from experiments.core.types import Session, Turn
from experiments.methods._common import make_llm, text_of

ASSISTANT_SYSTEM = """\
You are a warm, concise personal AI assistant with long-term memory of the user. Today is {today}.
Reply to the user's latest message in 1-3 sentences. Use the memory context when it is relevant and
mention remembered details naturally; if memory does not cover something, do not make it up.
Memory items carry the date of the conversation they came from."""


# The same reply style the retrieval assistants get from ASSISTANT_SYSTEM; memory_v4's chat agent
# otherwise answers like a QA system ("Pescatarian.") instead of holding a conversation.
LIVE_STYLE = ("You are chatting live with the user as their warm, concise personal assistant: reply to "
              "their latest message in 1-3 natural sentences, using what you remember when it is relevant. "
              "When they ask about their past, answer the question directly.")


def today(date: str) -> str:
    return f"{dt.date.fromisoformat(date):%A}, {date}"


class RetrievalAssistant:
    """Baselines: retrieve memory for every user message; one shared prompt writes the reply."""

    def __init__(self, method, model, usage):
        self.method, self.llm = method, make_llm(model, usage)
        self.date, self.turns = "", []

    def start_day(self, date: str) -> None:
        self.date, self.turns = date, []

    def reply(self, message: str) -> str:
        self.turns.append({"speaker": "user", "text": message})
        context = self.method.retrieve(message) or "(no memories yet)"
        convo = "\n".join(f"{t['speaker']}: {t['text']}" for t in self.turns[-12:])
        out = text_of(self.llm.invoke([
            SystemMessage(content=ASSISTANT_SYSTEM.format(today=today(self.date))),
            HumanMessage(content=f"MEMORY CONTEXT:\n{context}\n\nTODAY'S CONVERSATION:\n{convo}\n\n"
                                 "Write the assistant's reply.")]).content)
        self.turns.append({"speaker": "assistant", "text": out})
        return out

    def end_day(self, turns: list[dict]) -> dict:
        stats = self.method.ingest(Session(self.date, self.date, [Turn(t["speaker"], t["text"]) for t in turns]))
        self.method.finalize()
        return stats or {}

    def before_probes(self) -> None:
        pass


class AgentAssistant:
    """memory_v4: its own live API, converse() during the day and flush() at the day boundary."""

    def __init__(self, method):
        self.method, self.agent, self.date = method, method.agent, ""

    def start_day(self, date: str) -> None:
        self.date = date

    def reply(self, message: str) -> str:
        # converse() with the date given the way the retrieval assistants get it: memory_v4's
        # live API has no date input, and the note stays out of the turns written to memory.
        from langchain_core.messages import AIMessage

        history = [AIMessage(content=t["text"]) if t["speaker"] == "assistant" else HumanMessage(content=t["text"])
                   for t in self.agent._pending]
        note = HumanMessage(content=f"[Context: today is {today(self.date)}. {LIVE_STYLE}]")
        answer = self.agent._run([note] + history + [HumanMessage(content=message)], False)
        self.agent._pending += [{"speaker": "user", "text": message}, {"speaker": "assistant", "text": answer}]
        return answer

    def end_day(self, turns: list[dict]) -> dict:
        report = self.agent.flush(self.date)
        return {"added": getattr(report, "added", 0)} if report else {}

    def before_probes(self) -> None:
        # Same monthly finalization as setup_2 (summaries; distillation for ours_v4d).
        self.method.finalize()
