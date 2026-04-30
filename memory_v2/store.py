"""
Unified memory store: SQLite for structured memories, FAISS for raw conversations
and hierarchical summaries.

Single `memories` table — no hardcoded types. The LLM assigns type/subject freely.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"


class MemoryStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self.db_path        = os.path.join(base_dir, "memories.db")
        self.conv_index     = os.path.join(base_dir, "faiss_conversations")
        self.summary_index  = os.path.join(base_dir, "faiss_summaries")

        self.embeddings = HuggingFaceEmbeddings(
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
                    type         TEXT,
                    subject      TEXT,
                    speaker      TEXT    DEFAULT 'user',
                    source_date  TEXT,
                    extracted_at TEXT    DEFAULT (datetime('now'))
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type    ON memories(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_subject ON memories(subject)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_speaker ON memories(speaker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date    ON memories(source_date)")
            # Add speaker column to existing DBs that predate this change
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN speaker TEXT DEFAULT 'user'")
            except Exception:
                pass

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ── Memories (SQL) ──────────────────────────────────────────────

    def add_memory(self, content: str, type: str, subject: str, source_date: str,
                   speaker: str = "user"):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO memories (content, type, subject, speaker, source_date) VALUES (?,?,?,?,?)",
                (content, type.lower().strip(), subject.lower().strip(),
                 speaker.lower().strip(), source_date),
            )

    def add_memories(self, memories: List[Dict], source_date: str):
        """Bulk insert. Each dict: {content, type, subject, speaker}."""
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO memories (content, type, subject, speaker, source_date) VALUES (?,?,?,?,?)",
                [
                    (
                        m["content"],
                        m.get("type", "general").lower().strip(),
                        m.get("subject", "").lower().strip(),
                        m.get("speaker", "user").lower().strip(),
                        source_date,
                    )
                    for m in memories
                ],
            )

    def query_memories(
        self,
        type: Optional[str] = None,
        subject: Optional[str] = None,
        speaker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Flexible SQL query — any combination of filters."""
        clauses, params = [], []
        if type:
            clauses.append("type = ?"); params.append(type.lower())
        if subject:
            clauses.append("subject = ?"); params.append(subject.lower())
        if speaker:
            clauses.append("speaker = ?"); params.append(speaker.lower())
        if start_date:
            clauses.append("source_date >= ?"); params.append(start_date)
        if end_date:
            clauses.append("source_date <= ?"); params.append(end_date)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (f"SELECT id, content, type, subject, speaker, source_date FROM memories "
               f"{where} ORDER BY source_date LIMIT ?")
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            {"id": r[0], "content": r[1], "type": r[2],
             "subject": r[3], "speaker": r[4], "date": r[5]}
            for r in rows
        ]

    def get_all_types(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT type FROM memories ORDER BY type").fetchall()
        return [r[0] for r in rows if r[0]]

    def get_all_subjects(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT subject FROM memories ORDER BY subject").fetchall()
        return [r[0] for r in rows if r[0]]

    def get_all_speakers(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT speaker FROM memories ORDER BY speaker"
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    def get_preference_sequence(
        self, subject: str, start_date: str, end_date: str,
        speaker: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """
        Returns [(date, content), ...] for preference-type memories of a subject,
        ordered chronologically — used for temporal injection in summaries.
        """
        clauses = [
            "type = 'preference'", "subject = ?",
            "source_date >= ?", "source_date <= ?",
        ]
        params = [subject.lower(), start_date, end_date]
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
        """Store a raw conversation chunk in the conversation FAISS index."""
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

    def get_recent_conversations(self, k: int = 10) -> List[Document]:
        """Return the k most recent conversation chunks by source_date metadata."""
        if self._conv_store is None:
            return []
        all_docs = list(self._conv_store.docstore._dict.values())
        all_docs.sort(key=lambda d: d.metadata.get("source_date", ""), reverse=True)
        return all_docs[:k]

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

    # ── Helpers ─────────────────────────────────────────────────────

    def _load_faiss(self, path: str) -> Optional[FAISS]:
        if os.path.exists(os.path.join(path, "index.faiss")):
            try:
                return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Warning: could not load FAISS at {path}: {e}")
        return None
