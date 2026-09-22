"""
Capability tests for memory_v3.

Run with no arguments and no API keys:

    python3 tests/test_memory_v3.py

Every test here runs offline. The embedding model is swapped for a deterministic
fake, so FAISS still exercises its real code path without pulling a 1.2 GB model,
and the LLM is replaced by a scripted stub so each step is driven through known
inputs instead of whatever a live model happens to return.

What that does and does not prove: the plumbing is verified — the pipeline writes
what it is told, the buffer drains, hallucinated entity names are rejected, the
schemas the model sees are well formed. Retrieval *quality* is not, because fake
embeddings carry no semantics, and neither is the quality of extraction itself.
For that, run the live check:

    LIVE=1 GOOGLE_API_KEY=... python3 tests/test_memory_v3.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import warnings

# faiss and torch each ship their own OpenMP runtime; on macOS loading both aborts
# the process. Must be set before faiss is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# memory_v2's store leaves its sqlite connections to be garbage-collected, which is
# harmless but drowns the test output.
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

import memory_v2.store as store_mod

# Must be patched before any MemoryStore is constructed.
store_mod.HuggingFaceEmbeddings = lambda **kwargs: DeterministicFakeEmbedding(size=64)

from memory_v2.store import MemoryStore                      # noqa: E402
from memory_v3.agent import AgenticMemoryAgent               # noqa: E402
from memory_v3.ingest import IngestionPipeline               # noqa: E402
from memory_v3.tools import build_read_tools                 # noqa: E402


# ── Test doubles ────────────────────────────────────────────────────

class ScriptedLLM(Runnable):
    """
    Returns queued JSON payloads, one per invocation.

    A real Runnable rather than a duck-typed shim: callers compose
    `prompt | llm | JsonOutputParser()`, and anything that is not a Runnable makes
    that composition raise — which the callers catch and treat as a step failure, so
    a sloppier fake silently tests the error path. Emitting a JSON AIMessage also
    exercises the real parse step.
    """

    def __init__(self, *payloads):
        self.queue = list(payloads)
        self.calls = []

    def invoke(self, input, config=None, **kwargs):
        self.calls.append(input)
        payload = self.queue.pop(0) if self.queue else {}
        return AIMessage(content=json.dumps(payload))


class ExplodingLLM(Runnable):
    """Fails on every call, to exercise the pipeline's fallback behaviour."""

    def invoke(self, input, config=None, **kwargs):
        raise RuntimeError("model unavailable")


class ScriptedGraph:
    """Stands in for the compiled chat graph, returning a fixed reply."""

    def __init__(self, reply="done"):
        self.reply = reply
        self.received = []          # every message list this graph was invoked with

    def invoke(self, state, config=None):
        self.received.append(state["messages"])
        return {"messages": state["messages"] + [AIMessage(content=self.reply)]}


def extraction(*items):
    """Build an extract-step payload from (entity, content, date) triples."""
    return {"preferences": [
        {"entity": e, "content": c, "date": d} for e, c, d in items
    ]}


DAY = [{"time_of_day": "Morning",
        "turns": [{"speaker": "User", "text": "Took the night bus in."}]}]


class StoreTestCase(unittest.TestCase):
    """Gives each test a clean store on disk."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="memv3_test_")
        self.store = MemoryStore(self.tmp)
        self.read = {t.name: t for t in build_read_tools(self.store)}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def pipeline(self, *payloads):
        return IngestionPipeline(self.store, ScriptedLLM(*payloads))


# ── 1. Ingestion pipeline: extract → dedupe → write ─────────────────

class TestIngestionPipeline(StoreTestCase):
    """'Update (Add/Remove) Preferences structured database' — as a fixed pipeline."""

    def test_extracted_preference_is_written(self):
        pipe = self.pipeline(extraction(("transport", "night bus", "2026-03-01")))
        report = pipe.run("2026-03-01", DAY)

        rows = self.store.query_memories()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entity"], "transport")
        self.assertEqual(rows[0]["content"], "night bus")
        self.assertEqual(report.added, 1)

    def test_entity_is_normalised(self):
        """Mixed case and stray whitespace must not fragment the taxonomy."""
        pipe = self.pipeline(extraction(("  Transport  ", "tram", "2026-03-01")))
        pipe.run("2026-03-01", DAY)
        self.assertEqual(self.store.get_all_entities(), ["transport"])

    def test_empty_extraction_writes_nothing(self):
        report = self.pipeline({"preferences": []}).run("2026-03-01", DAY)
        self.assertEqual(self.store.query_memories(), [])
        self.assertEqual(report.added, 0)

    def test_one_llm_call_per_day(self):
        """Extraction is the only model call; deduplication is a SQL lookup."""
        self.store.add_memory(content="tram", entity="transport",
                              source_date="2026-02-28")
        llm = ScriptedLLM(extraction(("transport", "night bus", "2026-03-01")))
        IngestionPipeline(self.store, llm).run("2026-03-01", DAY)
        self.assertEqual(len(llm.calls), 1)

    def test_duplicate_within_one_extraction_is_dropped(self):
        """A choice mentioned twice in one day must not be counted twice."""
        pipe = self.pipeline(extraction(
            ("transport", "tram", "2026-03-01"),
            ("transport", "tram", "2026-03-01"),
        ))
        report = pipe.run("2026-03-01", DAY)

        self.assertEqual(len(self.store.query_memories()), 1)
        self.assertEqual(report.added, 1)
        self.assertEqual(report.duplicates, 1)

    def test_duplicate_against_the_store_is_dropped(self):
        """Re-ingesting a day, or a second flush restating it, adds nothing."""
        self.store.add_memory(content="tram", entity="transport",
                              source_date="2026-03-01")
        report = self.pipeline(
            extraction(("transport", "tram", "2026-03-01"))
        ).run("2026-03-01", DAY)

        self.assertEqual(len(self.store.query_memories()), 1, "day double-counted")
        self.assertEqual(report.duplicates, 1)
        self.assertEqual(report.added, 0)

    def test_duplicate_check_ignores_case(self):
        self.store.add_memory(content="Tram", entity="transport",
                              source_date="2026-03-01")
        report = self.pipeline(
            extraction(("transport", "tram", "2026-03-01"))
        ).run("2026-03-01", DAY)
        self.assertEqual(report.duplicates, 1)

    def test_recurrence_on_a_new_date_is_kept(self):
        """The same choice on another day is evidence, not a duplicate."""
        self.store.add_memory(content="tram", entity="transport",
                              source_date="2026-03-01")
        self.pipeline(
            extraction(("transport", "tram", "2026-03-02"))
        ).run("2026-03-02", DAY)
        self.assertEqual(len(self.store.query_memories()), 2)

    def test_same_day_different_entities_both_kept(self):
        pipe = self.pipeline(extraction(
            ("transport", "tram", "2026-03-01"),
            ("meal", "tram", "2026-03-01"),      # same value, different entity
        ))
        pipe.run("2026-03-01", DAY)
        self.assertEqual(len(self.store.query_memories()), 2)

    def test_extraction_failure_is_reported_not_raised(self):
        report = IngestionPipeline(self.store, ExplodingLLM()).run("2026-03-01", DAY)
        self.assertEqual(report.added, 0)
        self.assertTrue(report.errors)
        self.assertIn("extract failed", report.errors[0])

    def test_known_entities_are_offered_to_the_extractor(self):
        """Reusing an existing name is only possible if the model is shown the list."""
        self.store.add_memory(content="tram", entity="transport",
                              source_date="2026-03-01")
        llm = ScriptedLLM(extraction(("transport", "night bus", "2026-03-02")))
        IngestionPipeline(self.store, llm).run("2026-03-02", DAY)
        self.assertIn("transport", str(llm.calls[0]))


# ── 2. RAG database ─────────────────────────────────────────────────

class TestRagDatabase(StoreTestCase):
    """'Update RAG database' — the pipeline's indexing step."""

    def test_conversation_is_indexed_and_retrievable(self):
        report = self.pipeline({"preferences": []}).run("2026-03-01", DAY)
        self.assertEqual(report.chunks_indexed, 1)
        self.assertIn("night bus",
                      self.read["search_conversations"].invoke({"query": "bus", "k": 3}))

    def test_indexing_writes_to_both_sql_and_vector_store(self):
        """Chunks must be reachable by exact date as well as by similarity."""
        self.pipeline({"preferences": []}).run("2026-03-01", DAY)
        by_date = self.store.get_conversations_by_date_range("2026-03-01", "2026-03-01")
        self.assertEqual(len(by_date), 1)

    def test_long_day_is_split_into_chunks(self):
        from memory_v3.ingest import CHUNK_SIZE
        turns = [{"speaker": "User", "text": f"line {i}"}
                 for i in range(CHUNK_SIZE * 2 + 1)]
        report = self.pipeline({"preferences": []}).run(
            "2026-03-01", [{"time_of_day": "All Day", "turns": turns}]
        )
        self.assertEqual(report.chunks_indexed, 3)

    def test_indexing_happens_even_when_extraction_fails(self):
        """The raw record survives so a later run can recover what was said."""
        report = IngestionPipeline(self.store, ExplodingLLM()).run("2026-03-01", DAY)
        self.assertEqual(report.chunks_indexed, 1)

    def test_search_on_empty_store_is_handled(self):
        self.assertIn("no matching", self.read["search_conversations"].invoke(
            {"query": "anything", "k": 3}
        ))


# ── 2b. Retrieval tools ─────────────────────────────────────────────

class TestRetrievalTools(StoreTestCase):
    """The agent assembles its own context, so each source must be reachable."""

    def _seed(self):
        for day, value in [("2026-03-01", "tram"), ("2026-03-02", "night bus")]:
            self.store.add_memory(content=value, entity="transport", source_date=day)

    def test_preferences_come_back_in_date_order(self):
        """Counting and interval questions depend on the ordering being real."""
        self._seed()
        out = self.read["search_preferences"].invoke({"entity": "transport"})
        self.assertLess(out.index("tram"), out.index("night bus"))

    def test_unknown_entity_returns_the_real_names(self):
        """
        An empty result is indistinguishable from 'no such data', so a wrong guess
        has to say so — otherwise the agent concludes the memory is empty and stops.
        """
        self._seed()
        out = self.read["search_preferences"].invoke({"entity": "helicopter"})
        self.assertIn("No entity named", out)
        self.assertIn("transport", out)

    def test_date_window_filters(self):
        self._seed()
        out = self.read["search_preferences"].invoke(
            {"entity": "transport", "start_date": "2026-03-02", "end_date": "2026-03-02"}
        )
        self.assertIn("night bus", out)
        self.assertNotIn("tram", out)

    def test_entities_are_listed_for_the_agent(self):
        self._seed()
        self.assertIn("transport", self.read["list_entities"].invoke({}))

    def test_missing_summary_says_so(self):
        out = self.read["get_summary"].invoke({"level": "lifetime"})
        self.assertIn("no lifetime summary", out.lower())


# ── 3. Live conversation: converse / flush ──────────────────────────

class TestConverseAndFlush(StoreTestCase):
    """Fast read path during the day, pipeline at the end of it."""

    def _agent(self, *payloads):
        agent = AgenticMemoryAgent.__new__(AgenticMemoryAgent)   # skip model setup
        agent.store = self.store
        agent.recursion_limit = 10
        agent._pending = []
        agent.chat_agent = ScriptedGraph("noted")
        agent.pipeline = self.pipeline(*payloads)
        return agent

    def test_turn_is_buffered_not_written(self):
        agent = self._agent()
        agent.converse("I switched to the night bus.")
        self.assertEqual(agent.pending_turns, 2)              # user + assistant
        self.assertEqual(self.store.query_memories(), [])     # nothing written yet

    def test_later_turn_sees_the_earlier_one(self):
        """The day's history is carried in context while the store is still stale."""
        agent = self._agent()
        agent.converse("I switched to the night bus.")
        agent.converse("Remind me what I'm taking?")

        second_turn = agent.chat_agent.received[1]
        self.assertEqual(len(second_turn), 3)                 # Human, AI, Human
        self.assertIn("night bus", second_turn[0].content)
        self.assertIsInstance(second_turn[1], AIMessage)

    def test_flush_ingests_the_whole_day_at_once(self):
        agent = self._agent(extraction(("transport", "night bus", "2026-03-01")))
        agent.converse("I switched to the night bus.")
        agent.converse("Remind me what I'm taking?")

        report = agent.flush("2026-03-01")

        self.assertEqual(report.added, 1)
        self.assertEqual(agent.pending_turns, 0)
        self.assertEqual(self.store.query_memories()[0]["content"], "night bus")

    def test_flush_with_nothing_pending_is_a_no_op(self):
        self.assertIsNone(self._agent().flush("2026-03-01"))

    def test_chat_is_stateless_unlike_converse(self):
        agent = self._agent()
        agent.chat("one")
        agent.chat("two")
        self.assertEqual(len(agent.chat_agent.received[1]), 1)
        self.assertEqual(agent.pending_turns, 0)


# ── 4. Domain blindness ─────────────────────────────────────────────

class TestNoHardcodedDomains(unittest.TestCase):
    """
    The three ground-truth patterns are known in advance, so any dataset vocabulary
    reaching the model is a leak that inflates every score. Tool descriptions are the
    worst offender: they are injected on every single call.
    """

    LEAKED = ("coffee", "latte", "espresso", "hoodie", "t-shirt", "yoga",
              "climbing", "rest day", "clothing", "beverage", "workout")

    def test_no_dataset_vocabulary_reaches_the_model(self):
        from memory_v3.prompts import CHAT_SYSTEM, EXTRACT_SYSTEM

        text = [EXTRACT_SYSTEM, CHAT_SYSTEM]
        for tool in build_read_tools(None):
            text.append(f"{tool.name} {tool.description}")

        blob = "\n".join(text).lower()
        found = sorted({w for w in self.LEAKED if w in blob})
        self.assertEqual(found, [], f"dataset vocabulary leaked into prompts: {found}")


# ── 5. Tool surface ─────────────────────────────────────────────────

class TestToolSurface(unittest.TestCase):

    def test_chat_agent_cannot_write(self):
        names = {t.name for t in build_read_tools(None)}
        for writer in ("add_preference", "remove_preference", "index_conversation"):
            self.assertNotIn(writer, names)
        self.assertIn("search_preferences", names)

    def test_no_tool_hides_an_llm_call(self):
        """
        Retrieval tools are SQL and FAISS only. A tool that quietly makes its own
        model call would put a second, less-informed decision inside what reads as a
        lookup — and would not show up in the agent's own turn count.
        """
        from memory_v3 import tools as tools_mod
        source = open(tools_mod.__file__).read()
        for marker in ("ChatPromptTemplate", "JsonOutputParser", "self.llm", "invoke("):
            self.assertNotIn(marker, source, f"a retrieval tool now calls the model ({marker})")

    def test_every_tool_is_described_for_the_model(self):
        for tool in build_read_tools(None):
            with self.subTest(tool=tool.name):
                self.assertTrue(tool.description.strip(),
                                f"{tool.name} has no description")
                for arg in tool.args:
                    self.assertIn(arg, tool.description,
                                  f"{tool.name}: '{arg}' is undocumented")


# ── 6. Live check (opt-in) ──────────────────────────────────────────

@unittest.skipUnless(os.getenv("LIVE"), "set LIVE=1 to run against a real model")
class TestLiveIngestion(unittest.TestCase):
    """One real day through the pipeline — costs API calls, so it is opt-in."""

    def test_pipeline_records_a_choice(self):
        tmp = tempfile.mkdtemp(prefix="memv3_live_")
        try:
            agent = AgenticMemoryAgent(
                base_dir=tmp, model=os.getenv("MODEL", "gemini-3.5-flash")
            )
            report = agent.ingest("2026-03-01", [{
                "time_of_day": "Morning",
                "turns": [{"speaker": "User",
                           "text": "Took the night bus in this morning, "
                                   "same as the last few days."}],
            }])
            print(f"\n  {report}")

            rows = agent.store.query_memories()
            self.assertTrue(rows, "pipeline recorded nothing at all")
            self.assertTrue(
                any("bus" in r["content"].lower() for r in rows),
                f"expected the choice to be recorded, got {rows}",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
