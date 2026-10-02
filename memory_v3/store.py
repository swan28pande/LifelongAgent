"""
Unified memory store: SQLite for structured memories, FAISS for raw conversations
and hierarchical summaries.

Single `memories` table — no hardcoded types. The LLM assigns entities freely.
"""

import os
import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"


class PrefixedEmbeddings(HuggingFaceEmbeddings):
    """nomic-embed-text-v1 requires task-specific prefixes for best retrieval."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(
            [f"search_document: {t}" for t in texts]
        )

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"search_query: {text}")


class MemoryStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self.db_path        = os.path.join(base_dir, "memories.db")
        self.conv_index     = os.path.join(base_dir, "faiss_conversations")
        self.summary_index  = os.path.join(base_dir, "faiss_summaries")

        self.embeddings = PrefixedEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"trust_remote_code": True},
        )

        self._init_db()
        self._conv_store    = self._load_faiss(self.conv_index)
        self._summary_store = self._load_faiss(self.summary_index)

    # ── DB setup ────────────────────────────────────────────────────

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    content      TEXT    NOT NULL,
                    entity       TEXT,
                    type         TEXT    DEFAULT 'preference',
                    speaker      TEXT    DEFAULT 'user',
                    source_date  TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity ON memories(entity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_speaker ON memories(speaker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date    ON memories(source_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type    ON memories(type)")

            # Migrate existing DBs that lack the type column
            try:
                conn.execute("SELECT type FROM memories LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE memories ADD COLUMN type TEXT DEFAULT 'preference'")

            # Conversations table for exact chronological retrieval
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_date  TEXT,
                    speaker      TEXT    DEFAULT 'user',
                    content      TEXT    NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_date ON conversations(source_date)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS summary_meta (
                    identifier   TEXT PRIMARY KEY,
                    built_at     TEXT NOT NULL,
                    source_hash  TEXT NOT NULL
                )
            """)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ── Memories (SQL) ──────────────────────────────────────────────

    def add_memory(self, content: str, entity: str, source_date: str,
                   speaker: str = "user", type: str = "preference"):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO memories (content, entity, type, speaker, source_date) VALUES (?,?,?,?,?)",
                (content, entity.lower().strip(), type,
                 speaker.lower().strip(), source_date),
            )

    def add_memories(self, memories: List[Dict], source_date: str):
        """Bulk insert. Each dict: {content, entity, speaker, date, type}."""
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO memories (content, entity, type, speaker, source_date) VALUES (?,?,?,?,?)",
                [
                    (
                        m["content"],
                        m.get("entity", "").lower().strip(),
                        m.get("type", "preference"),
                        m.get("speaker", "user").lower().strip(),
                        m.get("date") or source_date,
                    )
                    for m in memories
                ],
            )

    def query_memories(
        self,
        entity: Optional[str] = None,
        speaker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Flexible SQL query — any combination of filters."""
        clauses, params = [], []
        if entity:
            clauses.append("entity = ?"); params.append(entity.lower())
        if speaker:
            clauses.append("speaker = ?"); params.append(speaker.lower())
        if start_date:
            clauses.append("source_date >= ?"); params.append(start_date)
        if end_date:
            clauses.append("source_date <= ?"); params.append(end_date)
        if type:
            clauses.append("type = ?"); params.append(type)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (f"SELECT id, content, entity, speaker, source_date, type FROM memories "
               f"{where} ORDER BY source_date LIMIT ?")
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            {"id": r[0], "content": r[1],
             "entity": r[2], "speaker": r[3], "date": r[4], "type": r[5]}
            for r in rows
        ]

    def delete_memories(self, memory_ids: List[int]) -> int:
        """Delete memories by id. Returns the number of rows removed."""
        if not memory_ids:
            return 0
        placeholders = ",".join("?" * len(memory_ids))
        with self._conn() as conn:
            cur = conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", memory_ids
            )
            return cur.rowcount

    def get_unique_preferences(self, type: Optional[str] = None) -> List[Dict]:
        """Returns all unique [entity, content] pairs to maintain taxonomic consistency."""
        if type:
            sql = "SELECT DISTINCT entity, content FROM memories WHERE type = ? ORDER BY entity, content"
            params = (type,)
        else:
            sql = "SELECT DISTINCT entity, content FROM memories ORDER BY entity, content"
            params = ()
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [{"entity": r[0], "content": r[1]} for r in rows]

    def get_all_entities(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT entity FROM memories ORDER BY entity").fetchall()
        return [r[0] for r in rows if r[0]]

    def get_all_speakers(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT speaker FROM memories ORDER BY speaker"
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    def get_preference_sequence(
        self, entity: str, start_date: str, end_date: str,
        speaker: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Returns [(date, content), ...] for preference memories of an entity,
        ordered chronologically — used for temporal injection in summaries.
        """
        clauses = [
            "entity = ?",
            "source_date >= ?", "source_date <= ?",
        ]
        params = [entity.lower(), start_date, end_date]
        if speaker:
            clauses.append("speaker = ?")
            params.append(speaker.lower())

        sql = (f"SELECT source_date, content FROM memories "
               f"WHERE {' AND '.join(clauses)} ORDER BY source_date")
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_date_range(self) -> Tuple[Optional[str], Optional[str]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MIN(source_date), MAX(source_date) FROM memories"
            ).fetchone()
        return row[0], row[1]

    # ── Raw conversations (FAISS) ───────────────────────────────────

    def add_conversation(self, text: str, metadata: Dict):
        """Store a raw conversation chunk in the conversation FAISS index AND SQL store."""
        # 1. SQL Store (for fast exact date retrieval)
        source_date = metadata.get("source_date", "")
        speaker = metadata.get("speaker", "user")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (source_date, speaker, content) VALUES (?,?,?)",
                (source_date, speaker, text)
            )

        # 2. FAISS Store (for semantic retrieval)
        doc = Document(page_content=text, metadata=metadata)
        if self._conv_store is None:
            self._conv_store = FAISS.from_documents([doc], self.embeddings)
        else:
            self._conv_store.add_documents([doc])
        self._conv_store.save_local(self.conv_index)

    def search_conversations(self, query: str, k: int = 5) -> List[Document]:
        if self._conv_store is None:
            return []
        return self._conv_store.similarity_search(query, k=k)

    def get_conversations_by_date_range(self, start_date: str, end_date: str) -> List[Document]:
        """Return all conversation chunks whose source_date falls in [start_date, end_date] using SQL."""
        sql = "SELECT content, source_date, speaker FROM conversations WHERE source_date >= ? AND source_date <= ? ORDER BY source_date ASC, id ASC"
        with self._conn() as conn:
            rows = conn.execute(sql, (start_date, end_date)).fetchall()
        return [Document(page_content=r[0], metadata={"source_date": r[1], "speaker": r[2]}) for r in rows]

    def get_recent_conversations(self, k: int = 10) -> List[Document]:
        """Return the k most recent conversation chunks using SQL, ordered chronologically."""
        sql = "SELECT content, source_date, speaker FROM conversations ORDER BY source_date DESC, id DESC LIMIT ?"
        with self._conn() as conn:
            rows = conn.execute(sql, (k,)).fetchall()
        # The query gets the most recent ones, but we want to return them in chronological order
        rows.reverse()
        return [Document(page_content=r[0], metadata={"source_date": r[1], "speaker": r[2]}) for r in rows]

    # ── Summaries (FAISS) ───────────────────────────────────────────

    def save_summary(self, identifier: str, title: str, content: str, metadata: Dict = {}):
        """Upsert a summary document into the summary FAISS store."""
        # Remove old version
        if self._summary_store:
            old_ids = [
                doc_id
                for doc_id, doc in self._summary_store.docstore._dict.items()
                if doc.metadata.get("identifier") == identifier
            ]
            if old_ids:
                self._summary_store.delete(old_ids)

        doc = Document(
            page_content=f"[{identifier}] {title}\n\n{content}",
            metadata={"identifier": identifier, "title": title, **metadata},
        )
        if self._summary_store is None:
            self._summary_store = FAISS.from_documents([doc], self.embeddings)
        else:
            self._summary_store.add_documents([doc])
        self._summary_store.save_local(self.summary_index)

    def get_summary(self, identifier: str) -> Optional[str]:
        if self._summary_store is None:
            return None
        for doc in self._summary_store.docstore._dict.values():
            if doc.metadata.get("identifier") == identifier:
                return doc.page_content
        return None

    def search_summaries(self, query: str, k: int = 3) -> List[Document]:
        if self._summary_store is None:
            return []
        return self._summary_store.similarity_search(query, k=k)

    def search_summaries_by_layer(self, query: str,
                                  k_week: int = 2, k_month: int = 2,
                                  k_year: int = 1, k_lifetime: int = 1,
                                  k_distilled: int = 2) -> Dict[str, List[Document]]:
        """Search each summary layer independently and return results grouped by level."""
        if self._summary_store is None:
            return {}
        by_layer: Dict[str, List[Document]] = {}
        for doc in self._summary_store.docstore._dict.values():
            level = doc.metadata.get("identifier", "").split(":")[0]
            by_layer.setdefault(level, []).append(doc)

        limits = {"week": k_week, "month": k_month, "year": k_year,
                  "lifetime": k_lifetime, "distilled": k_distilled}
        results: Dict[str, List[Document]] = {}
        for level, docs in by_layer.items():
            k = limits.get(level, 1)
            if not docs or k == 0:
                continue
            if len(docs) <= k:
                results[level] = docs
            else:
                mini = FAISS.from_documents(docs, self.embeddings)
                results[level] = mini.similarity_search(query, k=k)
        return results

    def get_lifetime_summary(self, speaker: Optional[str] = None) -> Optional[str]:
        """Return lifetime summary for a specific speaker, or the combined one."""
        if speaker:
            return self.get_summary(f"lifetime:{speaker.lower()}")
        # Try combined first, fall back to first available speaker's lifetime
        combined = self.get_summary("lifetime")
        if combined:
            return combined
        # Collect all per-speaker lifetime summaries
        if self._summary_store:
            parts = []
            for doc in self._summary_store.docstore._dict.values():
                if doc.metadata.get("identifier", "").startswith("lifetime:"):
                    parts.append(doc.page_content)
            if parts:
                return "\n\n".join(parts)
        return None


    # ── Summary metadata ─────────────────────────────────────────────

    def get_summary_hash(self, identifier: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT source_hash FROM summary_meta WHERE identifier = ?",
                (identifier,),
            ).fetchone()
        return row[0] if row else None

    def set_summary_hash(self, identifier: str, source_hash: str):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO summary_meta (identifier, built_at, source_hash) "
                "VALUES (?, ?, ?)",
                (identifier, now, source_hash),
            )

    # ── Helpers ─────────────────────────────────────────────────────

    def _load_faiss(self, path: str) -> Optional[FAISS]:
        if os.path.exists(os.path.join(path, "index.faiss")):
            try:
                return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Warning: could not load FAISS at {path}: {e}")
        return None
