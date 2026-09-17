"""
Side-by-side comparison of v2 and v3 ingestion.

    python3 tests/test_ingestion_comparison.py                # offline, deterministic
    LIVE=1 GOOGLE_API_KEY=... python3 tests/test_ingestion_comparison.py

The offline tests hold extraction constant and compare what each system *does* with
the same extracted items, which isolates the structural differences from the noise of
two different prompts. Both systems read a different JSON key for their extraction
step, so one stub payload carrying both keys feeds each of them identically.

The live comparison runs real days from the eval dataset through both and prints what
actually diverged. That is the number worth quoting: the offline tests prove the
mechanisms differ, not that the difference matters on real data.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

import memory_v2.store as store_mod

store_mod.HuggingFaceEmbeddings = lambda **kwargs: DeterministicFakeEmbedding(size=64)

from memory_v2.extractor import MemoryExtractor      # noqa: E402
from memory_v2.store import MemoryStore              # noqa: E402
from memory_v3.ingest import IngestionPipeline       # noqa: E402


# ── Shared stub ─────────────────────────────────────────────────────

class SharedLLM(Runnable):
    """
    Feeds both systems the same extraction.

    v2 reads "memories" and v3 reads "preferences", so one payload carrying both keys
    lets each pick up the one it looks for. Holding extraction constant is what makes
    the comparison fair: any divergence downstream is the systems differing, not two
    prompts producing different items.
    """

    def __init__(self, items):
        self.extraction = {"memories": items, "preferences": items}
        self.calls = 0

    def invoke(self, input, config=None, **kwargs):
        self.calls += 1
        return AIMessage(content=json.dumps(self.extraction))


def pref(entity, content, date):
    return {"entity": entity, "content": content, "date": date}


DAY = [{"time_of_day": "Morning",
        "turns": [{"speaker": "User", "text": "Took the night bus in."}]}]


class ComparisonTestCase(unittest.TestCase):
    """Two clean stores, one per system."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cmp_")
        self.v2_store = MemoryStore(os.path.join(self.tmp, "v2"))
        self.v3_store = MemoryStore(os.path.join(self.tmp, "v3"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_v2(self, date, items, conversations=DAY):
        llm = SharedLLM(items)
        MemoryExtractor(self.v2_store, llm=llm).extract_and_store(date, conversations)
        return llm

    def run_v3(self, date, items, conversations=DAY):
        llm = SharedLLM(items)
        report = IngestionPipeline(self.v3_store, llm=llm).run(date, conversations)
        return llm, report

    def rows(self, store):
        return [(r["entity"], r["content"], r["date"])
                for r in store.query_memories(limit=1000)]


# ── 1. Identical inputs, identical writes ───────────────────────────

class TestAgreement(ComparisonTestCase):
    """Where the systems should not differ, they must not."""

    def test_same_extraction_produces_the_same_rows(self):
        items = [pref("transport", "night bus", "2026-03-01"),
                 pref("meal", "soup", "2026-03-01")]
        self.run_v2("2026-03-01", items)
        self.run_v3("2026-03-01", items)

        self.assertEqual(sorted(self.rows(self.v2_store)),
                         sorted(self.rows(self.v3_store)))

    def test_entity_normalisation_matches(self):
        items = [pref("  Transport  ", "tram", "2026-03-01")]
        self.run_v2("2026-03-01", items)
        self.run_v3("2026-03-01", items)

        self.assertEqual(self.v2_store.get_all_entities(),
                         self.v3_store.get_all_entities())

    def test_both_index_the_same_chunks(self):
        turns = [{"speaker": "User", "text": f"line {i}"} for i in range(11)]
        day = [{"time_of_day": "All Day", "turns": turns}]
        item = [pref("transport", "tram", "2026-03-01")]

        self.run_v2("2026-03-01", item, conversations=day)
        _, report = self.run_v3("2026-03-01", item, conversations=day)

        v2_chunks = self.v2_store.get_conversations_by_date_range(
            "2026-03-01", "2026-03-01")
        self.assertEqual(len(v2_chunks), report.chunks_indexed)

    def test_recurrence_is_kept_by_both(self):
        """A repeat on a later date is evidence, and neither may drop it."""
        first = [pref("transport", "tram", "2026-03-01")]
        second = [pref("transport", "tram", "2026-03-02")]

        self.run_v2("2026-03-01", first)
        self.run_v2("2026-03-02", second)
        self.run_v3("2026-03-01", first)
        self.run_v3("2026-03-02", second)

        self.assertEqual(len(self.rows(self.v2_store)), 2)
        self.assertEqual(len(self.rows(self.v3_store)), 2)

    def test_both_cost_one_llm_call_per_day(self):
        item = [pref("transport", "tram", "2026-03-01")]
        v2_llm = self.run_v2("2026-03-01", item)
        v3_llm, _ = self.run_v3("2026-03-01", item)

        self.assertEqual(v2_llm.calls, 1)
        self.assertEqual(v3_llm.calls, 1)


# ── 2. Where they diverge ───────────────────────────────────────────

class TestDivergence(ComparisonTestCase):
    """Differences that come from the pipeline, not from the extraction."""

    def test_v2_stores_same_day_duplicates_and_v3_does_not(self):
        """
        v2's dedup is commented out at extractor.py:108, so an item already recorded
        for the same date is written again. Over 60 days that inflates the frequency
        of whatever gets mentioned twice in one conversation.
        """
        item = [pref("transport", "tram", "2026-03-01")]

        self.run_v2("2026-03-01", item)
        self.run_v2("2026-03-01", item)          # same day, ingested twice

        self.run_v3("2026-03-01", item)
        _, report = self.run_v3("2026-03-01", item)

        self.assertEqual(len(self.rows(self.v2_store)), 2, "v2 behaviour changed")
        self.assertEqual(len(self.rows(self.v3_store)), 1)
        self.assertEqual(report.duplicates, 1)

    def test_v2_keeps_repeats_inside_one_extraction_and_v3_does_not(self):
        """
        A choice mentioned twice in one conversation can be extracted twice. v2 writes
        both, so that day counts double against every other day in the pattern.
        """
        twice = [pref("transport", "tram", "2026-03-01"),
                 pref("transport", "tram", "2026-03-01")]

        self.run_v2("2026-03-01", twice)
        _, report = self.run_v3("2026-03-01", twice)

        self.assertEqual(len(self.rows(self.v2_store)), 2)
        self.assertEqual(len(self.rows(self.v3_store)), 1)
        self.assertEqual(report.duplicates, 1)

    def test_v2_drops_the_conversation_when_nothing_is_extracted(self):
        """
        memory_v2/extractor.py:104 returns before the indexing loop at line 116, so a
        day that yields no preference is never written to the RAG store at all — the
        raw conversation is lost, not just the extraction.

        That matters for the benchmark: `recall` questions ask what happened on a
        given day, and a day of pure chat with no dated choice becomes unanswerable
        for v2 while remaining retrievable for v3.
        """
        turns = [{"speaker": "User", "text": "Just catching up, nothing new today."}]
        day = [{"time_of_day": "All Day", "turns": turns}]

        self.run_v2("2026-03-01", [], conversations=day)
        _, report = self.run_v3("2026-03-01", [], conversations=day)

        v2_chunks = self.v2_store.get_conversations_by_date_range(
            "2026-03-01", "2026-03-01")

        self.assertEqual(len(v2_chunks), 0, "v2 gained indexing on empty extraction")
        self.assertEqual(report.chunks_indexed, 1)

    def test_only_v3_reports_what_it_did(self):
        _, report = self.run_v3("2026-03-01", [pref("transport", "tram", "2026-03-01")])
        self.assertEqual(report.added, 1)
        self.assertEqual(report.chunks_indexed, 1)
        self.assertEqual(report.entities, ["transport"])


# ── 3. Capability parity check ──────────────────────────────────────

class TestParity(ComparisonTestCase):
    """Things v2 does that v3 should not have quietly dropped."""

    def test_v3_lacks_v2s_post_hoc_entity_consolidation(self):
        """
        v2's LifelongAgent.consolidate_memories merges fragmented entities across the
        whole store before summarizing. v3 relies on showing known entities at extract
        time instead, and has no cleanup pass. Prevention misses things that a merge
        would catch, so this is a real gap — asserted here so it stays visible rather
        than being discovered during a benchmark run.
        """
        from memory_v2.agent import LifelongAgent
        from memory_v3.agent import AgenticMemoryAgent

        self.assertTrue(hasattr(LifelongAgent, "consolidate_memories"))
        self.assertFalse(
            hasattr(AgenticMemoryAgent, "consolidate_memories"),
            "v3 gained consolidation — delete this test and celebrate",
        )


# ── 4. Live comparison (opt-in) ─────────────────────────────────────

@unittest.skipUnless(os.getenv("LIVE"), "set LIVE=1 to run against a real model")
class TestLiveComparison(unittest.TestCase):
    """
    Run real dataset days through both systems and print what diverged.

    Control the day count with DAYS (default 5) — each day costs 1 call for v2 and
    2 for v3.
    """

    def test_compare_on_real_days(self):
        from memory_v3.agent import _make_llm

        model = os.getenv("MODEL", "gemini-3.1-flash-lite")
        n_days = int(os.getenv("DAYS", "5"))

        path = os.path.join(PROJECT_ROOT, "datasets", "eval", "conversations.json")
        with open(path) as f:
            sessions = json.load(f)["user_1"]["sessions"]
        dates = sorted(sessions)[:n_days]

        tmp = tempfile.mkdtemp(prefix="cmp_live_")
        try:
            v2_store = MemoryStore(os.path.join(tmp, "v2"))
            v3_store = MemoryStore(os.path.join(tmp, "v3"))
            extractor = MemoryExtractor(v2_store, llm=_make_llm(model, 0.0))
            pipeline = IngestionPipeline(v3_store, llm=_make_llm(model, 0.0))

            for date in dates:
                turns = [{"speaker": t["speaker"], "text": t["text"]}
                         for t in sessions[date]["turns"]]
                day = [{"time_of_day": "All Day", "turns": turns}]
                extractor.extract_and_store(date, day)
                print(f"  v3 {pipeline.run(date, day)}")

            v2_rows = v2_store.query_memories(limit=1000)
            v3_rows = v3_store.query_memories(limit=1000)

            def keyed(rows):
                return {(r["entity"], r["content"].lower(), r["date"]) for r in rows}

            v2_set, v3_set = keyed(v2_rows), keyed(v3_rows)

            print(f"\n{'':22} {'v2':>8} {'v3':>8}")
            print(f"{'rows written':22} {len(v2_rows):>8} {len(v3_rows):>8}")
            print(f"{'distinct rows':22} {len(v2_set):>8} {len(v3_set):>8}")
            print(f"{'exact duplicates':22} "
                  f"{len(v2_rows) - len(v2_set):>8} {len(v3_rows) - len(v3_set):>8}")
            print(f"{'entities':22} {len(v2_store.get_all_entities()):>8} "
                  f"{len(v3_store.get_all_entities()):>8}")
            print(f"\n  v2 entities: {v2_store.get_all_entities()}")
            print(f"  v3 entities: {v3_store.get_all_entities()}")

            only_v2 = sorted(v2_set - v3_set)
            only_v3 = sorted(v3_set - v2_set)
            print(f"\n  captured only by v2 ({len(only_v2)}):")
            for r in only_v2[:10]:
                print(f"    {r}")
            print(f"  captured only by v3 ({len(only_v3)}):")
            for r in only_v3[:10]:
                print(f"    {r}")

            # Not an accuracy claim — both ran, on the same days, and wrote something.
            self.assertTrue(v2_rows, "v2 extracted nothing")
            self.assertTrue(v3_rows, "v3 extracted nothing")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
