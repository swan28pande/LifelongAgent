# Proposed SQL Schema Update

The `memories` table in the SQL store currently contains redundant fields since the system has moved to a preference-only structured memory model.

## Changes

1.  **Remove `type` field**: Since the `memories` table now only stores preferences, the `type` column (always set to "preference") is redundant.
2.  **Remove `extracted_at` field**: The timestamp of extraction is not used for retrieval or reasoning and adds unnecessary bloat. (Referred to as "content file" metadata).

- `memory_v2/store.py`: Update `_init_db`, `add_memory`, `add_memories`, and `query_memories`. Added `get_unique_preferences()`.
- `memory_v2/extractor.py`: Update extraction prompt and logic to use unique preferences for consistency.
- `memory_v2/context_builder.py`: Update SQL queries to remove the `type` filter.
- `memory_v2/agent.py`: Update any inspection helpers.

## Affected Files and Diffs

### 1. `memory_v2/store.py`
```diff
--- a/memory_v2/store.py
+++ b/memory_v2/store.py
@@ -46,14 +46,11 @@
                 CREATE TABLE IF NOT EXISTS memories (
                     id           INTEGER PRIMARY KEY AUTOINCREMENT,
                     content      TEXT    NOT NULL,
-                    type         TEXT,
                     subject      TEXT,
                     speaker      TEXT    DEFAULT 'user',
                     source_date  TEXT,
-                    extracted_at TEXT    DEFAULT (datetime('now'))
                 )
             """)
-            conn.execute("CREATE INDEX IF NOT EXISTS idx_type    ON memories(type)")
             conn.execute("CREATE INDEX IF NOT EXISTS idx_subject ON memories(subject)")
             conn.execute("CREATE INDEX IF NOT EXISTS idx_speaker ON memories(speaker)")
             conn.execute("CREATE INDEX IF NOT EXISTS idx_date    ON memories(source_date)")
@@ -81,32 +78,32 @@
 
-    def add_memory(self, content: str, type: str, subject: str, source_date: str,
+    def add_memory(self, content: str, subject: str, source_date: str,
                    speaker: str = "user"):
         with self._conn() as conn:
             conn.execute(
-                "INSERT INTO memories (content, type, subject, speaker, source_date) VALUES (?,?,?,?,?)",
-                (content, type.lower().strip(), subject.lower().strip(),
+                "INSERT INTO memories (content, subject, speaker, source_date) VALUES (?,?,?,?)",
+                (content, subject.lower().strip(),
                  speaker.lower().strip(), source_date),
             )
 
     def add_memories(self, memories: List[Dict], source_date: str):
-        """Bulk insert. Each dict: {content, type, subject, speaker}."""
+        """Bulk insert. Each dict: {content, subject, speaker}."""
         with self._conn() as conn:
             conn.executemany(
-                "INSERT INTO memories (content, type, subject, speaker, source_date) VALUES (?,?,?,?,?)",
+                "INSERT INTO memories (content, subject, speaker, source_date) VALUES (?,?,?,?)",
                 [
                     (
                         m["content"],
-                        m.get("type", "general").lower().strip(),
                         m.get("subject", "").lower().strip(),
                         m.get("speaker", "user").lower().strip(),
                         source_date,
                     )
                     for m in memories
                 ],
             )
 
     def query_memories(
         self,
-        type: Optional[str] = None,
         subject: Optional[str] = None,
         speaker: Optional[str] = None,
         start_date: Optional[str] = None,
@@ -118,8 +115,6 @@
         """Flexible SQL query — any combination of filters."""
         clauses, params = [], []
-        if type:
-            clauses.append("type = ?"); params.append(type.lower())
         if subject:
             clauses.append("subject = ?"); params.append(subject.lower())
         if speaker:
@@ -130,8 +125,8 @@
         where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
-        sql = (f"SELECT id, content, type, subject, speaker, source_date FROM memories "
+        sql = (f"SELECT id, content, subject, speaker, source_date FROM memories "
                f"{where} ORDER BY source_date LIMIT ?")
         params.append(limit)
 
         with self._conn() as conn:
             rows = conn.execute(sql, params).fetchall()
 
         return [
-            {"id": r[0], "content": r[1], "type": r[2],
-             "subject": r[3], "speaker": r[4], "date": r[5]}
+            {"id": r[0], "content": r[1],
+             "subject": r[2], "speaker": r[3], "date": r[4]}
             for r in rows
         ]
 
-    def get_all_types(self) -> List[str]:
-        with self._conn() as conn:
-            rows = conn.execute("SELECT DISTINCT type FROM memories ORDER BY type").fetchall()
-        return [r[0] for r in rows if r[0]]
-
     def get_all_subjects(self) -> List[str]:
```

### 2. `memory_v2/extractor.py`
```diff
--- a/memory_v2/extractor.py
+++ b/memory_v2/extractor.py
@@ -21,11 +21,10 @@
-Each preference memory has four fields:
-  "type"    : always "preference"
+Each preference memory has three fields:
   "subject" : the main category the preference belongs to
   "speaker" : who this preference belongs to (usually "user")
   "content" : the CORE choice only (noun/noun-phrase).
@@ -103,7 +102,7 @@
         raw_text = self._format_conversations(date, conversations)
         existing_subjects = self.store.get_all_subjects()
-        # Fetch some recent preferences for context to maintain consistency
-        recent = self.store.query_memories(type="preference", limit=100)
-        existing_prefs = [f"[{m['subject']}] {m['content']}" for m in recent]
+        # Fetch unique existing preferences to maintain consistency
+        unique_prefs = self.store.get_unique_preferences()
+        existing_prefs = [f"[{m['subject']}] {m['content']}" for m in unique_prefs]
         
         extracted = self.extract_memories(raw_text, date, existing_subjects, existing_prefs)
@@ -195,7 +194,7 @@
             memories = result.get("memories", result) if isinstance(result, dict) else result
             if not isinstance(memories, list):
                 return []
-            return [m for m in memories if isinstance(m, dict) and m.get("type") == "preference" and m.get("content")]
+            return [m for m in memories if isinstance(m, dict) and m.get("content")]
         except Exception as e:
```

### 3. `memory_v2/context_builder.py`
```diff
--- a/memory_v2/context_builder.py
+++ b/memory_v2/context_builder.py
@@ -151,7 +151,6 @@
         for subj in subjects or [None]:
             rows = self.store.query_memories(
-                type="preference",
                 subject=subj,
                 speaker=speaker,
                 limit=k * 3,
```
