"""TiMem baseline: 5-level temporal memory tree (L1 fragment → L2 session → L3 daily → L4 weekly → L5 monthly).

Uses TiMem's MemoryGenerator for LLM-based layer generation and Qdrant for
vector retrieval.  Ingest and consolidation happen through TiMem's own code;
answer generation uses the experiment harness's LLM (Gemini via Vertex AI).

Requires:
  - Qdrant running on localhost:16333  (docker-compose in baselines/TiMem/migration/)
  - PYTHONPATH or sys.path including baselines/TiMem
"""

import asyncio
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .. import config
from ..core.method import MemoryMethod
from ..core.types import Answer, Question, Session, prompt_for
from ._common import answer_from_context, make_llm

TIMEM_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "TiMem"
if str(TIMEM_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEM_ROOT))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:16333")
COLLECTION = "timem_experiment"
EMBED_DIM = 768


def _run(coro):
    """Run an async coroutine from sync code, handling nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


class TiMem(MemoryMethod):
    name = "timem"
    persistent = True

    def __init__(self, store_dir, instance, model, usage):
        super().__init__(store_dir, instance, model, usage)
        self.llm = make_llm(model, usage)
        self.instance_id = instance.id

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self._qdrant = QdrantClient(url=QDRANT_URL, check_compatibility=False)
        self._collection = f"{COLLECTION}_{self.instance_id}"

        if not self._qdrant.collection_exists(self._collection):
            self._qdrant.create_collection(
                self._collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

        from memory_v3.store import PrefixedEmbeddings
        self._embedder = PrefixedEmbeddings(
            model_name=config.EMBED_MODEL,
            model_kwargs={"trust_remote_code": True},
        )

        self._generator = None
        self._l1s = []
        self._sessions_l1 = defaultdict(list)
        self._dates_sessions = defaultdict(list)

    def _gen(self):
        if self._generator is None:
            from timem.memory.memory_generator import MemoryGenerator
            self._generator = MemoryGenerator(llm_provider="gemini")
        return self._generator

    def _embed(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    def _store(self, layer: str, content: str, meta: dict):
        from qdrant_client.models import PointStruct
        point_id = str(uuid.uuid4())
        vector = self._embed(content)
        self._qdrant.upsert(
            self._collection,
            points=[PointStruct(
                id=point_id,
                vector=vector,
                payload={"layer": layer, "content": content, **meta},
            )],
        )
        return point_id

    def ingest(self, session: Session) -> dict:
        dialogue = "\n".join(f"{t.speaker}: {t.text}" for t in session.turns)
        prev = self._sessions_l1.get(session.id, [])
        prev_content = "\n".join(prev) if prev else None

        l1 = _run(self._gen().generate_l1_content(dialogue, prev_content))
        self._l1s.append({"session_id": session.id, "date": session.date, "content": l1})
        self._sessions_l1[session.id].append(l1)
        self._dates_sessions[session.date].append(session.id)
        self._store("L1", l1, {"date": session.date, "session_id": session.id})
        return {"l1_chars": len(l1)}

    def finalize(self) -> None:
        gen = self._gen()

        l2s = {}
        for sid, fragments in self._sessions_l1.items():
            l2 = _run(gen.generate_l2_content(fragments))
            date = next((l["date"] for l in self._l1s if l["session_id"] == sid), "")
            l2s[sid] = {"content": l2, "date": date}
            self._store("L2", l2, {"date": date, "session_id": sid})

        l3s = {}
        for date, sids in sorted(self._dates_sessions.items()):
            child_contents = [l2s[sid]["content"] for sid in sids if sid in l2s]
            if child_contents:
                l3 = _run(gen.generate_l3_content(child_contents, date=date))
                l3s[date] = l3
                self._store("L3", l3, {"date": date})

        from datetime import timedelta
        dates = sorted(l3s.keys())
        if not dates:
            return

        weeks = defaultdict(list)
        for d in dates:
            dt = datetime.strptime(d, "%Y-%m-%d")
            y, w, _ = dt.isocalendar()
            weeks[f"{y}-W{w:02d}"].append(d)

        l4s = {}
        for week_id, week_dates in sorted(weeks.items()):
            child_contents = [l3s[d] for d in week_dates if d in l3s]
            if child_contents:
                parts = week_id.split("-W")
                year = int(parts[0])
                week_num = int(parts[1])
                l4 = _run(gen.generate_l4_content(
                    child_contents, year=year, week_number=week_num,
                    week_start=week_dates[0], week_end=week_dates[-1],
                ))
                l4s[week_id] = l4
                self._store("L4", l4, {"week": week_id})

        months = defaultdict(list)
        for week_id in l4s:
            parts = week_id.split("-W")
            year = int(parts[0])
            week_num = int(parts[1])
            dt = datetime.strptime(f"{year}-W{week_num}-1", "%G-W%V-%u")
            month_id = dt.strftime("%Y-%m")
            months[month_id].append(week_id)

        for month_id, month_weeks in sorted(months.items()):
            child_contents = [l4s[w] for w in month_weeks if w in l4s]
            if child_contents:
                parts = month_id.split("-")
                year = int(parts[0])
                month = int(parts[1])
                l5 = _run(gen.generate_l5_content(
                    child_contents, year=year, month=month,
                    month_start=f"{month_id}-01",
                    month_end=f"{month_id}-28",
                ))
                self._store("L5", l5, {"month": month_id})

        print(f"  TiMem consolidation: {len(self._l1s)} L1, {len(l2s)} L2, "
              f"{len(l3s)} L3, {len(l4s)} L4, {len(months)} L5")

    def answer(self, q: Question) -> Answer:
        query_vec = self._embed(prompt_for(q))
        results = self._qdrant.query_points(
            self._collection,
            query=query_vec,
            limit=config.TOP_K,
            with_payload=True,
        )
        lines = []
        for pt in results.points:
            layer = pt.payload.get("layer", "?")
            content = pt.payload.get("content", "")
            date = pt.payload.get("date", "")
            lines.append(f"[{layer}] ({date}) {content}")
        context = "\n\n".join(lines) or "(no memories)"
        return answer_from_context(self.llm, q, context, {"hits": len(lines)})
