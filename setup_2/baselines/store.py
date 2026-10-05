"""Durable observations and metadata inside a checkpoint's rollback directory."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from experiments.core.types import Session, Turn


class Sessions:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'baseline.sqlite3'
        with self.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, date TEXT NOT NULL, time TEXT, turns TEXT NOT NULL)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)''')

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _session(row) -> Session:
        return Session(row['id'], row['date'], [Turn(**t) for t in json.loads(row['turns'])], row['time'])

    def get(self, identifier: str) -> Session | None:
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM sessions WHERE id = ?', (identifier,)).fetchone()
        return self._session(row) if row else None

    def all(self) -> list[Session]:
        with self.connect() as conn:
            rows = conn.execute('SELECT * FROM sessions ORDER BY date, id').fetchall()
        return [self._session(row) for row in rows]

    def add(self, session: Session) -> bool:
        existing = self.get(session.id)
        if existing is not None:
            if existing != session:
                raise ValueError(f'Observed session changed: {session.id}')
            return False
        turns = json.dumps([{'speaker': t.speaker, 'text': t.text} for t in session.turns], ensure_ascii=False)
        with self.connect() as conn:
            conn.execute('INSERT INTO sessions VALUES (?, ?, ?, ?)',
                         (session.id, session.date, session.time, turns))
        return True

    def metadata(self, key: str, default=None):
        with self.connect() as conn:
            row = conn.execute('SELECT value FROM metadata WHERE key = ?', (key,)).fetchone()
        return json.loads(row['value']) if row else default

    def set_metadata(self, key: str, value) -> None:
        with self.connect() as conn:
            conn.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', (key, json.dumps(value)))
