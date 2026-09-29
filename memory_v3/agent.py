"""
AgenticMemoryAgent — the top-level interface to the v3 memory system.

Writing and reading are built differently, because they are different problems:

    ingest() → a fixed pipeline (extract → dedupe → write → index), one LLM
               call per day, same output for the same input
    chat()   → a tool-calling loop, because how many retrievals a question needs
               is not knowable before it is asked

A live day runs in two phases:
    converse() → answer now, with today's conversation carried in context
    flush()    → at day's end, hand the whole conversation to the write path

Memory covers previous days; the in-context history covers today. That is why nothing
is lost by deferring the write — a preference mentioned at breakfast is still visible
at dinner, and it reaches the store before tomorrow begins.

Usage:
    agent = AgenticMemoryAgent(base_dir="results/synthetic/v3/store",
                               model="gemini-3.5-flash")

    # Batch: replay a recorded day
    agent.ingest("2026-03-01", [{"time_of_day": "Morning", "turns": [...]}])

    # Live: answer through the day, write once at the end of it
    agent.converse("I switched to black coffee this morning.")
    agent.converse("Remind me what I'm drinking?")   # sees the earlier turn
    agent.flush("2026-03-01")
"""

import os
from typing import Dict, List, Optional

import dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .store import MemoryStore
from .summarizer import Summarizer

from .ingest import IngestionPipeline, IngestReport
from .prompts import CHAT_SYSTEM
from .tools import build_read_tools

dotenv.load_dotenv()

# Each ReAct step is one LLM call; this caps a runaway loop on a single day/query.
DEFAULT_RECURSION_LIMIT = 60


def _use_vertex() -> bool:
    """
    Vertex unless told otherwise.

    Vertex authenticates with application-default credentials, so there is no API key
    to carry. Set GOOGLE_GENAI_USE_VERTEXAI=false to fall back to the Developer API,
    which does need GOOGLE_API_KEY.
    """
    return os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() not in (
        "false", "0", "no"
    )


def _vertex_project() -> Optional[str]:
    """Project from the environment, else whatever ADC was set up against."""
    explicit = os.getenv("GOOGLE_CLOUD_PROJECT")
    if explicit:
        return explicit
    try:
        import google.auth

        return google.auth.default()[1]
    except Exception:
        return None


def _make_llm(model: str, temperature: float = 0.0, callbacks=None):
    if "gemini" not in model.lower():
        return ChatOpenAI(model=model, temperature=temperature, callbacks=callbacks)

    from langchain_google_genai import ChatGoogleGenerativeAI

    if not _use_vertex():
        return ChatGoogleGenerativeAI(
            model=model, temperature=temperature, callbacks=callbacks
        )

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        callbacks=callbacks,
        vertexai=True,
        project=_vertex_project(),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    )


class AgenticMemoryAgent:
    def __init__(
        self,
        base_dir: str,
        model: str = "gemini-3.5-flash",
        chat_model: Optional[str] = None,
        callbacks=None,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    ):
        self.store = MemoryStore(base_dir)
        self.recursion_limit = recursion_limit

        ingest_llm = _make_llm(model, 0.0, callbacks)
        chat_llm = _make_llm(chat_model or model, 0.3, callbacks)

        self.pipeline = IngestionPipeline(self.store, llm=ingest_llm)
        self.summarizer = Summarizer(self.store, llm=_make_llm(model, 0.2, callbacks))

        self.read_tools = build_read_tools(self.store)
        self.chat_agent = create_agent(
            chat_llm, self.read_tools,
            system_prompt=CHAT_SYSTEM, name="memory_reader",
        )

        # Turns spoken since the last flush, awaiting ingestion.
        self._pending: List[Dict] = []

    # ── Ingestion ───────────────────────────────────────────────────

    def ingest(
        self,
        date: str,
        conversations: List[Dict],
        speaker: str = "user",
        update_summaries: bool = False,
    ) -> IngestReport:
        """
        Run one day through the ingestion pipeline.

        conversations: [{time_of_day, turns: [{speaker, text}]}]
        update_summaries: if True, incrementally rebuild only the summaries
            affected by this date (the containing week, month, year, lifetime)
            rather than requiring a separate build_summaries() call.
        Returns a report of what was extracted, written, and indexed.
        """
        report = self.pipeline.run(date, conversations, speaker=speaker)
        if update_summaries:
            self.summarizer.update_after_ingest(date)
        return report

    def ingest_range(
        self,
        sessions: Dict[str, Dict],
        dates: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> List[IngestReport]:
        """
        Ingest many days from an eval-dataset `sessions` dict:
        {"YYYY-MM-DD": {"turns": [{speaker, text}, ...]}, ...}

        Days are processed in date order, so the store only ever contains what was
        known up to the day being read.
        """
        reports = []
        for date in dates or sorted(sessions.keys()):
            session = sessions[date]
            turns = [
                {"speaker": t["speaker"], "text": t["text"]} for t in session["turns"]
            ]
            report = self.ingest(
                date,
                [{"time_of_day": session.get("time_of_day", "All Day"), "turns": turns}],
            )
            if verbose:
                print(report)
            reports.append(report)
        return reports

    # ── Chat ────────────────────────────────────────────────────────

    def chat(self, query: str, verbose: bool = False) -> str:
        """Answer a query, letting the agent retrieve whatever it needs first."""
        return self._run(query, verbose)

    def chat_with_trace(self, query: str) -> dict:
        """Answer a query and return the full tool-call trace alongside the answer."""
        inbox = [HumanMessage(content=query)]
        result = self.chat_agent.invoke(
            {"messages": inbox},
            config={"recursion_limit": self.recursion_limit},
        )
        messages = result["messages"]
        tool_calls = []
        for m in messages:
            calls = getattr(m, "tool_calls", None)
            if calls:
                for c in calls:
                    tool_calls.append({"tool": c["name"], "args": c["args"]})
        return {
            "answer": self._text_of(messages[-1]),
            "tool_calls": tool_calls,
            "num_tool_calls": len(tool_calls),
        }

    # ── Live conversation (fast path / slow path) ───────────────────

    def converse(
        self, message: str, speaker: str = "user", verbose: bool = False
    ) -> str:
        """
        Answer a live turn with the day's conversation still in context.

        The running conversation is replayed to the chat agent on every turn, so
        something said at breakfast is still available at dinner even though nothing
        has been written to the store yet. Memory covers previous days; the in-context
        history covers today.

        Only the read path runs here, so the caller waits on retrieval and generation
        and nothing else. Recording inline would add several tool round-trips before
        the reply, and the agent could decide to summarize mid-conversation.

        Call `flush` at the end of the day to hand the whole conversation to the
        write path.
        """
        history = [
            HumanMessage(content=t["text"])
            if t["speaker"] != "assistant"
            else AIMessage(content=t["text"])
            for t in self._pending
        ]
        answer = self._run(history + [HumanMessage(content=message)], verbose)

        self._pending.append({"speaker": speaker, "text": message})
        self._pending.append({"speaker": "assistant", "text": answer})
        return answer

    def flush(
        self,
        date: str,
        time_of_day: str = "All Day",
        speaker: str = "user",
        update_summaries: bool = True,
    ) -> Optional[IngestReport]:
        """
        Ingest everything buffered since the last flush.

        Ingesting the whole session at once beats ingesting turn by turn: extraction
        sees the full day when deciding what was actually chosen, and it costs two LLM
        calls instead of two per turn.

        update_summaries defaults to True for live use — each flush is a day boundary,
        so the incremental update runs automatically.

        Returns the ingest report, or None if nothing was pending.
        """
        if not self._pending:
            return None

        turns, self._pending = self._pending, []
        return self.ingest(
            date,
            [{"time_of_day": time_of_day, "turns": turns}],
            speaker=speaker,
            update_summaries=update_summaries,
        )

    @property
    def pending_turns(self) -> int:
        """How many turns are buffered and not yet written to memory."""
        return len(self._pending)

    # ── Summaries ───────────────────────────────────────────────────

    def build_summaries(self, force: bool = False) -> None:
        """
        Build the weekly → monthly → yearly → lifetime hierarchy over everything
        currently stored.

        Driven by the caller rather than decided per-day: a summary of a period still
        in progress gets rebuilt on every subsequent day and can contradict the raw
        timeline while it is stale. Run this at period boundaries, or once at the end
        of a batch ingest.
        """
        self.summarizer.run(force=force)

    # ── Internals ───────────────────────────────────────────────────

    def _run(self, task, verbose: bool) -> str:
        """`task` is either a single prompt string or a prepared message list."""
        inbox = [HumanMessage(content=task)] if isinstance(task, str) else task
        result = self.chat_agent.invoke(
            {"messages": inbox},
            config={"recursion_limit": self.recursion_limit},
        )
        messages = result["messages"]

        if verbose:
            for m in messages:
                calls = getattr(m, "tool_calls", None)
                if calls:
                    for c in calls:
                        print(f"  → {c['name']}({c['args']})")

        return self._text_of(messages[-1])

    @staticmethod
    def _text_of(message) -> str:
        content = message.content
        if isinstance(content, list):
            return " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in content
            ).strip()
        return str(content).strip()

    # ── Inspection ──────────────────────────────────────────────────

    def show_memories(self, entity: Optional[str] = None, limit: int = 20):
        for r in self.store.query_memories(entity=entity, limit=limit):
            print(f"[{r['date']}] ({r['entity']}) {r['content']}")

    def tool_names(self) -> List[str]:
        """Tools the chat agent can reach. Ingestion has none — it is a pipeline."""
        return [t.name for t in self.read_tools]
