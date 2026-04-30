import os
import sys
import sqlite3
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Add memory/ directory to sys.path to import managers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memory"))
from memory_manager import MemoryManager
from structured_memory import StructuredMemoryManager


class InsightGenerator:

    def __init__(self, dataset_path: str = "dataset_3"):
        self.dataset_path = dataset_path
        self.faiss_path = os.path.join(dataset_path, "faiss_index")
        self.sql_path = os.path.join(dataset_path, "lifelong_memory.db")
        self.conv_path = os.path.join(dataset_path, "learning_conversations.json")
        self.insights_json_path = os.path.join(dataset_path, "summary_insights.json")

        self.rag_manager = MemoryManager(index_path=self.faiss_path)
        self.sql_manager = StructuredMemoryManager(db_path=self.sql_path)

        if "OPENAI_API_KEY" in os.environ:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        else:
            self.llm = None
            print("Warning: OPENAI_API_KEY not found.")

        self.insights_data = self._load_insights_json()

        # Dynamic normalization maps – populated once per run via LLM
        self._entity_norm: Optional[Dict[str, str]] = None
        self._noise_values: Optional[set] = None

    # ── Persistence helpers ─────────────────────────────────────────

    def _load_insights_json(self) -> Dict:
        defaults = {"weekly": {}, "monthly": {}, "yearly": {}, "lifetime": {}, "trajectories": {}, "routines": {}}
        if os.path.exists(self.insights_json_path):
            with open(self.insights_json_path, 'r') as f:
                data = json.load(f)
            for k, v in defaults.items():
                data.setdefault(k, v)
            return data
        return defaults

    def _save_insights_json(self):
        with open(self.insights_json_path, 'w') as f:
            json.dump(self.insights_data, f, indent=2)

    def _save_to_rag(self, insight_type: str, title: str, content: str, identifier: str, metadata_ext: Dict = {}):
        """Saves a single insight document to the RAG summary index, replacing any old version."""
        metadata = {
            "type": "summary_document",
            "insight_type": insight_type,
            "generated_at": datetime.now().strftime("%Y-%m-%d"),
            "title": title,
            "identifier": identifier
        }
        metadata.update(metadata_ext)
        doc = Document(
            page_content=f"TITLE: {title}\nTYPE: {insight_type}\n\n{content}",
            metadata=metadata
        )

        # Remove existing if present
        self.rag_manager.delete_summary_by_identifier(identifier)

        if self.rag_manager.summary_vector_store is None:
            from langchain_community.vectorstores import FAISS
            self.rag_manager.summary_vector_store = FAISS.from_documents([doc], self.rag_manager.embeddings)
        else:
            self.rag_manager.summary_vector_store.add_documents([doc])

        self.rag_manager.summary_vector_store.save_local(self.rag_manager.summary_index_path)

    def _get_iso_week(self, date_str: str) -> str:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        year, week, _ = dt.isocalendar()
        return f"{year}-W{week:02d}"

    # ── Dynamic normalization ───────────────────────────────────────

    def _build_normalization_map(self):
        """
        Query all unique (entity, preference) pairs from SQL, send them to the LLM,
        and get back:
          1. An entity normalization map  (raw_entity -> canonical_entity)
          2. A set of noisy preference values to discard
        Result is cached for the lifetime of this InsightGenerator instance.
        """
        if self._entity_norm is not None:
            return  # Already built

        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT entity FROM preferences ORDER BY entity")
        all_entities = [r[0] for r in cursor.fetchall()]

        cursor.execute(
            "SELECT DISTINCT entity, preference FROM preferences ORDER BY entity, preference"
        )
        all_pairs = [{"entity": r[0], "preference": r[1]} for r in cursor.fetchall()]
        conn.close()

        if not self.llm or not all_entities:
            self._entity_norm = {e.lower(): e.lower() for e in all_entities}
            self._noise_values = set()
            return

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a data-cleaning analyst. You will receive a list of raw entity names "
             "and (entity, preference) pairs from a user-preference database. Your job is to:\n"
             "1. Group synonymous or overlapping entities under one canonical name "
             "(e.g., 'tennis shoes', 'hiking boots', 'dress shoes', 'shoes' -> canonical 'shoes').\n"
             "2. Identify preference values that are noise – vague sentiments, adjectives, "
             "or non-specific text that do NOT represent an actual concrete preference "
             "(e.g., 'good', 'perfect', 'comfortable', 'thinking about wearing')."),
            ("human", (
                "Entities:\n{entities_json}\n\n"
                "(Entity, Preference) pairs:\n{pairs_json}\n\n"
                "Return ONLY a JSON object with two keys:\n"
                "1. \"entity_map\": an object mapping every raw entity (lowercased) to its canonical entity name (lowercased). Every entity from the input list MUST appear as a key.\n"
                "2. \"noise_values\": a list of preference strings (lowercased) that should be discarded as noise.\n\n"
                "Example:\n"
                "{{\"entity_map\": {{\"tennis shoes\": \"shoes\", \"hiking boots\": \"shoes\", \"shoes\": \"shoes\", \"coffee\": \"coffee\"}}, "
                "\"noise_values\": [\"good\", \"perfect\", \"comfortable\"]}}"
            ))
        ])

        try:
            chain = prompt | self.llm | JsonOutputParser()
            res = chain.invoke({
                "entities_json": json.dumps(all_entities),
                "pairs_json": json.dumps(all_pairs),
            })

            entity_map = res.get("entity_map", {})
            noise_list = res.get("noise_values", [])

            self._entity_norm = {k.lower(): v.lower() for k, v in entity_map.items()}
            self._noise_values = {v.lower() for v in noise_list}

            print(f"Dynamic normalization built: {len(self._entity_norm)} entity mappings, "
                  f"{len(self._noise_values)} noise values identified.")
        except Exception as e:
            print(f"Warning: LLM normalization failed ({e}). Using identity map.")
            self._entity_norm = {ent.lower(): ent.lower() for ent in all_entities}
            self._noise_values = set()

    # ── Deterministic sequence builder ──────────────────────────────

    def _build_sequences_for_range(self, start_date: str, end_date: str) -> Dict[str, List[tuple]]:
        """
        Query SQL for all preferences in [start_date, end_date], normalize entities
        using the dynamic LLM-built map, filter noise, and return
        {canonical_entity: [(date, preference), ...]} sorted by date.
        Only keeps the first clean preference per (entity, date) to avoid duplicates.
        """
        self._build_normalization_map()

        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT source_date, entity, preference FROM preferences "
            "WHERE source_date >= ? AND source_date <= ? ORDER BY source_date",
            (start_date, end_date),
        )
        rows = cursor.fetchall()
        conn.close()

        sequences: Dict[str, List[tuple]] = {}
        seen: set = set()

        for date, entity, pref in rows:
            canonical = self._entity_norm.get(entity.lower())
            if canonical is None:
                continue
            pref_clean = pref.strip().lower()
            if pref_clean in self._noise_values or len(pref_clean) < 2:
                continue
            key = (canonical, date)
            if key in seen:
                continue
            seen.add(key)
            sequences.setdefault(canonical, []).append((date, pref_clean))

        return sequences

    @staticmethod
    def _format_sequence(pairs: List[tuple]) -> str:
        """
        Turn [(date, pref), ...] into a human-readable string like:
        'espresso(Mar 01) -> espresso(Mar 02) -> latte(Mar 03). Runs: espresso×2, latte×1.'
        """
        if not pairs:
            return ""
        arrow = " -> ".join(
            f"{p}({datetime.strptime(d, '%Y-%m-%d').strftime('%b %d')})" for d, p in pairs
        )
        # Compute run-length encoding
        runs = []
        cur_val, cur_count = pairs[0][1], 1
        for _, v in pairs[1:]:
            if v == cur_val:
                cur_count += 1
            else:
                runs.append((cur_val, cur_count))
                cur_val, cur_count = v, 1
        runs.append((cur_val, cur_count))
        runs_str = ", ".join(f"{v}x{c}" for v, c in runs)
        return f"{arrow}. Runs: {runs_str}."

    def _format_all_sequences(self, sequences: Dict[str, List[tuple]]) -> str:
        """Format all entity sequences into a single block of text."""
        lines = []
        for entity in sorted(sequences):
            seq_str = self._format_sequence(sequences[entity])
            if seq_str:
                lines.append(f"[{entity}] {seq_str}")
        return "\n".join(lines)

    # ── Week date-range helper ──────────────────────────────────────

    @staticmethod
    def _week_date_range(week_id: str):
        """Return (start_date, end_date) strings for an ISO week like '2026-W10'."""
        year, w = week_id.split("-W")
        monday = datetime.strptime(f"{year} {int(w)} 1", "%G %V %u")
        sunday = monday + timedelta(days=6)
        return monday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d")

    # ── Weekly summaries ────────────────────────────────────────────

    def generate_weekly_summaries(self, force: bool = False):
        """Generates summaries for each week using deterministic SQL sequences."""
        if not self.llm or not os.path.exists(self.conv_path):
            return

        with open(self.conv_path, 'r') as f:
            all_convs = json.load(f)

        # Identify weeks and their month anchors
        weeks = {}
        week_months = {}
        for date_str in sorted(all_convs.keys()):
            week_id = self._get_iso_week(date_str)
            weeks.setdefault(week_id, True)
            if week_id not in week_months:
                week_months[week_id] = date_str[:7]

        for week_id in sorted(weeks.keys()):
            if not force and week_id in self.insights_data["weekly"]:
                print(f"  Summary for week {week_id} already exists, skipping.")
                continue
            print(f"Generating summary for week {week_id}...")
            start, end = self._week_date_range(week_id)
            sequences = self._build_sequences_for_range(start, end)
            seq_block = self._format_all_sequences(sequences)

            if not seq_block:
                print(f"  No clean preference data for {week_id}, skipping.")
                continue

            # Fetch raw conversations for this week from FAISS
            raw_docs = self.rag_manager.get_conversations_by_date_range(start, end)
            raw_conv_block = "\n\n".join(doc.page_content for doc in raw_docs) if raw_docs else ""

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a memory analyst. You will receive deterministic chronological "
                 "preference sequences and the raw conversations they were derived from. "
                 "Use the sequences as the ground truth for preference patterns, and the "
                 "raw conversations as context to understand the reasons behind them."),
                ("human", (
                    "Below are the exact day-by-day preference sequences for a week, "
                    "built from database records, followed by the raw conversations.\n\n"
                    f"Week: {week_id}\n"
                    f"Preference Sequences:\n{seq_block}\n\n"
                    f"Raw Conversations:\n{raw_conv_block}\n\n"
                    "Using the sequences as ground truth and the conversations as context:\n"
                    "1. State each entity's chronological flow exactly as given in the sequences "
                    "(do not recount or reorder).\n"
                    "2. Identify any temporal patterns "
                    "(consistency, mid-week shifts, oscillations, etc.).\n"
                    "3. Use the raw conversations to explain the reasons or context behind "
                    "notable preference shifts where relevant.\n"
                    "4. Provide a high-level interpretation of what these patterns "
                    "suggest about the user.\n\n"
                    "The 'summary' field MUST be a single string paragraph. "
                    "DO NOT use nested objects or lists.\n"
                    "Return ONLY a JSON object: "
                    "{{\"title\": \"...\", \"summary\": \"(single string paragraph)\"}}"
                ))
            ])

            try:
                chain = prompt | self.llm | JsonOutputParser()
                res = chain.invoke({})

                title = res.get("title", f"Weekly Summary: {week_id}")
                content = res.get("summary", "")
                month_id = week_months[week_id]

                self.insights_data["weekly"][week_id] = {
                    "title": title,
                    "content": content,
                    "sequences": seq_block,
                    "month_id": month_id,
                }
                self._save_to_rag("weekly-summary", title, content, week_id, {"month_id": month_id})
                self._save_insights_json()
            except Exception as e:
                print(f"Error for week {week_id}: {e}")

    # ── Monthly summaries ───────────────────────────────────────────

    def generate_monthly_summaries(self, force: bool = False):
        """Combines weekly sequences into month-long sequences and asks LLM to interpret."""
        if not self.llm:
            return

        months: Dict[str, List[str]] = {}
        for week_id, data in self.insights_data["weekly"].items():
            month_id = data.get("month_id")
            if not month_id:
                continue
            months.setdefault(month_id, []).append(week_id)

        for month_id, week_ids in sorted(months.items()):
            if not force and month_id in self.insights_data["monthly"]:
                print(f"  Summary for month {month_id} already exists, skipping.")
                continue
            print(f"Generating summary for month {month_id}...")

            start = f"{month_id}-01"
            y, m = int(month_id[:4]), int(month_id[5:7])
            if m == 12:
                end_next = f"{y+1}-01-01"
            else:
                end_next = f"{y}-{m+1:02d}-01"
            end_dt = datetime.strptime(end_next, "%Y-%m-%d") - timedelta(days=1)
            end = end_dt.strftime("%Y-%m-%d")

            sequences = self._build_sequences_for_range(start, end)
            seq_block = self._format_all_sequences(sequences)

            if not seq_block:
                continue

            weekly_narratives = "\n\n".join(
                f"Week {wid}: {self.insights_data['weekly'][wid].get('content', '')}"
                for wid in sorted(week_ids)
                if wid in self.insights_data["weekly"]
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a master memory analyst. You receive month-long, day-by-day "
                 "preference sequences and weekly narrative summaries. Use the sequences "
                 "as ground truth for what happened and the weekly summaries as context "
                 "for interpreting macro patterns."),
                ("human", (
                    "Below are the full-month preference sequences and the weekly summaries.\n\n"
                    f"Month: {month_id}\n\n"
                    f"Preference Sequences:\n{seq_block}\n\n"
                    f"Weekly Summaries:\n{weekly_narratives}\n\n"
                    "Using the sequences as ground truth and the weekly summaries as context:\n"
                    "1. State each entity's full chronological flow exactly as given in the sequences.\n"
                    "2. Identify the dominant runs: 'user preferred X for Y consecutive "
                    "days, then moved to Z for Q days'.\n"
                    "3. Identify any macro temporal patterns (gradual drift, periodic "
                    "oscillation, sudden switch, etc.).\n"
                    "4. Use the weekly summaries to add context or reasons behind notable shifts.\n\n"
                    "The 'summary' field MUST be a single string paragraph. "
                    "DO NOT use nested objects or lists.\n"
                    "Return ONLY a JSON object: "
                    "{{\"title\": \"...\", \"summary\": \"(single string paragraph)\"}}"
                ))
            ])

            try:
                chain = prompt | self.llm | JsonOutputParser()
                res = chain.invoke({})

                title = res.get("title", f"Monthly Snapshot: {month_id}")
                content = res.get("summary", "")

                self.insights_data["monthly"][month_id] = {
                    "title": title,
                    "content": content,
                    "sequences": seq_block,
                }
                self._save_to_rag("monthly-summary", title, content, month_id)
                self._save_insights_json()
            except Exception as e:
                print(f"Error for month {month_id}: {e}")

    # ── Yearly summaries ────────────────────────────────────────────

    def generate_yearly_summaries(self, force: bool = False):
        """Aggregates monthly sequences into year-long patterns."""
        if not self.llm:
            return

        years: Dict[str, List[str]] = {}
        for month_id in self.insights_data.get("monthly", {}):
            year = month_id[:4]
            years.setdefault(year, []).append(month_id)

        for year, month_ids in sorted(years.items()):
            if not force and year in self.insights_data["yearly"]:
                print(f"  Summary for year {year} already exists, skipping.")
                continue
            print(f"Generating summary for year {year}...")

            start = f"{year}-01-01"
            end = f"{year}-12-31"
            sequences = self._build_sequences_for_range(start, end)
            seq_block = self._format_all_sequences(sequences)

            if not seq_block:
                continue

            monthly_narratives = "\n\n".join(
                f"Month {mid}: {self.insights_data['monthly'][mid].get('content', '')}"
                for mid in sorted(month_ids)
                if mid in self.insights_data["monthly"]
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a master memory analyst. You receive year-long preference "
                 "sequences and monthly narrative summaries. Use the sequences as ground "
                 "truth for what happened and the monthly summaries as context for "
                 "interpreting long-term evolution."),
                ("human", (
                    f"Year: {year}\n\n"
                    f"Full-year Preference Sequences:\n{seq_block}\n\n"
                    f"Monthly Summaries:\n{monthly_narratives}\n\n"
                    "Using the sequences as ground truth and the monthly summaries as context:\n"
                    "1. Identify multi-month preference arcs (e.g., 'user drank espresso "
                    "for all of March, shifted to latte in April').\n"
                    "2. Quantify the dominant runs across months.\n"
                    "3. Use the monthly summaries to add context or reasons behind major shifts.\n"
                    "4. Give a high-level narrative of the user's preference evolution "
                    "over the year.\n\n"
                    "The 'summary' field MUST be a single string paragraph. "
                    "DO NOT use nested objects or lists.\n"
                    "Return ONLY a JSON object: "
                    "{{\"title\": \"...\", \"summary\": \"(single string paragraph)\"}}"
                ))
            ])

            try:
                chain = prompt | self.llm | JsonOutputParser()
                res = chain.invoke({})

                title = res.get("title", f"Yearly Summary: {year}")
                content = res.get("summary", "")

                self.insights_data["yearly"][year] = {"title": title, "content": content}
                self._save_to_rag("yearly-summary", title, content, f"year-{year}")
                self._save_insights_json()
            except Exception as e:
                print(f"Error for year {year}: {e}")

    # ── Lifetime summaries ──────────────────────────────────────────

    def generate_lifetime_summary(self, force: bool = False):
        """Aggregates all yearly summaries into a single comprehensive 'Till Date' summary."""
        if not self.llm:
            return

        if not force and "main" in self.insights_data.get("lifetime", {}):
            print("  Lifetime summary already exists, skipping.")
            return

        yearly_summaries = self.insights_data.get("yearly", {})
        if not yearly_summaries:
            print("No yearly summaries found for lifetime aggregation.")
            return

        print("Generating comprehensive lifetime summary (Till Date)...")

        # Sort years to ensure chronological narrative
        sorted_years = sorted(yearly_summaries.keys())
        yearly_blob = "\n\n".join(
            f"Year {year}: {yearly_summaries[year].get('content', '')}"
            for year in sorted_years
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are the ultimate memory archivist. Your task is to synthesize multiple "
             "yearly summaries into a single, comprehensive 'Lifetime Summary till date'. "
             "Identify the grand narrative of the user's life and preference evolution."),
            ("human", (
                "Below are the yearly summaries for the user's interaction history.\n\n"
                f"Yearly Summaries:\n{yearly_blob}\n\n"
                "Synthesize these into a single master summary that:\n"
                "1. Traces the major life-long preference arcs and shifts.\n"
                "2. Identifies the most enduring habits that have spanned multiple years.\n"
                "3. Highlights the most significant points of transformation or evolution.\n"
                "4. Provides a high-level philosophical or narrative conclusion about the user's journey so far.\n\n"
                "The 'summary' field MUST be a single string paragraph. "
                "DO NOT use nested objects or lists.\n"
                "Return ONLY a JSON object: "
                "{{\"title\": \"...\", \"summary\": \"(single string paragraph)\"}}"
            ))
        ])

        try:
            chain = prompt | self.llm | JsonOutputParser()
            res = chain.invoke({})

            title = res.get("title", "Lifetime Memory Synthesis (Till Date)")
            content = res.get("summary", "")

            self.insights_data["lifetime"]["main"] = {"title": title, "content": content}
            self._save_to_rag("lifetime-summary", title, content, "lifetime-overall")
            self._save_insights_json()
        except Exception as e:
            print(f"Error for lifetime summary: {e}")

    # ── Trajectories ────────────────────────────────────────────────

    def generate_trajectories(self, force: bool = False):
        """Generates preference trajectories using the full history for each canonical entity."""
        if not self.llm:
            return

        # Build sequences across ALL time
        sequences = self._build_sequences_for_range("0000-01-01", "9999-12-31")

        if not sequences:
            print("No clean preference data found for trajectories.")
            return

        for entity, pairs in sorted(sequences.items()):
            if len(pairs) < 2:
                continue  # Need at least 2 data points for a trajectory
            print(f"Analyzing trajectory for {entity} ({len(pairs)} data points)...")

            seq_str = self._format_sequence(pairs)

            # Load existing trajectory if available for incremental update
            existing_data = self.insights_data["trajectories"].get(entity)
            existing_traj = existing_data.get("content") if existing_data else None

            if existing_traj:
                sys_msg = (
                    "You are a preference analyst. You receive the complete chronological "
                    "sequence of a user's preference for a single entity AND its previous trajectory analysis. "
                    "Your goal is to UPDATE and REFINE the analysis to incorporate the latest trends while maintaining continuity."
                )
                user_msg = (
                    f"Analyze the COMPLETE trajectory of user's preference for '{entity}'.\n\n"
                    f"Previous Analysis:\n{existing_traj}\n\n"
                    f"Full Current Preference Sequence:\n[{entity}] {seq_str}\n\n"
                    "Using BOTH the previous analysis and the current sequence:\n"
                    "1. Incorporate the previous findings while highlighting NEW shifts or reinforcements found in the latest data.\n"
                    "2. State the full chronological flow clearly.\n"
                    "3. Identify dominant runs and macro patterns (drift, cycles, etc.).\n"
                    "4. Provide an updated prediction of what the user might prefer next.\n\n"
                    "The 'trajectory' field MUST be a single string paragraph. "
                    "DO NOT use nested objects or lists. Ensure the narrative flows naturally as an update.\n"
                    "Return ONLY a JSON object: "
                    "{{\"title\": \"...\", \"trajectory\": \"(single string paragraph)\"}}"
                )
            else:
                sys_msg = (
                    "You are a preference analyst. You receive the complete chronological "
                    "sequence of a user's preference for a single entity. "
                    "Your goal is to create an initial trajectory analysis tracing its evolution."
                )
                user_msg = (
                    f"Analyze the COMPLETE trajectory of user's preference for '{entity}'.\n\n"
                    f"Full Preference Sequence:\n[{entity}] {seq_str}\n\n"
                    "Using ONLY the provided sequence:\n"
                    "1. Trace the complete evolution from start to finish.\n"
                    "2. State the full chronological flow clearly.\n"
                    "3. Identify dominant runs and macro patterns (drift, cycles, etc.).\n"
                    "4. Provide a high-level prediction of what the user might prefer next.\n\n"
                    "The 'trajectory' field MUST be a single string paragraph. "
                    "DO NOT use nested objects or lists.\n"
                    "Return ONLY a JSON object: "
                    "{{\"title\": \"...\", \"trajectory\": \"(single string paragraph)\"}}"
                )

            prompt = ChatPromptTemplate.from_messages([
                ("system", sys_msg),
                ("human", user_msg)
            ])

            try:
                chain = prompt | self.llm | JsonOutputParser()
                res = chain.invoke({})

                title = res.get("title", f"Trajectory: {entity}")
                content = res.get("trajectory", "")

                self.insights_data["trajectories"][entity] = {"title": title, "content": content}
                self._save_to_rag("trajectory", title, content, f"traj-{entity}")
                self._save_insights_json()
            except Exception as e:
                print(f"Error for trajectory {entity}: {e}")

    # ── Routines ────────────────────────────────────────────────────

    def generate_routines(self, force: bool = False):
        """Generates/Updates temporal routine fingerprints (multi-preference synthesis)."""
        if not self.llm:
            return

        if not force and "main" in self.insights_data.get("routines", {}):
            print("  Routine analysis already exists, skipping.")
            return

        print("Analyzing multi-preference temporal routines...")

        # Query RAG for time-of-day clusters
        time_slots = ["Morning", "Afternoon", "Evening", "Late Night"]
        routine_blob = ""

        for slot in time_slots:
            docs = self.rag_manager.query(f"What does the user typically do in the {slot}?", n_results=15)
            routine_blob += f"\n--- {slot} Observations ---\n"
            routine_blob += "\n".join([d.page_content for d in docs]) + "\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a routine analyst. Synthesize multi-preference patterns across different times of day."),
            ("human", (
                "Based on these observations across different times of day, identify the user's core routines.\n"
                "Combine info about multiple preferences (e.g., '8 AM Coffee + T-Shirt sequence').\n\n"
                f"Observations:\n{routine_blob}\n\n"
                "Return ONLY a JSON object: {{\"title\": \"...\", \"routine_analysis\": \"...\"}}"
            ))
        ])

        try:
            chain = prompt | self.llm | JsonOutputParser()
            res = chain.invoke({})

            title = res.get("title", "Temporal Routine Fingerprint")
            content = res.get("routine_analysis", "")

            self.insights_data["routines"]["main"] = {"title": title, "content": content}
            self._save_to_rag("routine", title, content, "routine-fingerprint")
            self._save_insights_json()
        except Exception as e:
            print(f"Error for routines: {e}")

    # ── Pipeline ────────────────────────────────────────────────────

    def run_full_pipeline(self, force: bool = False):
        """Executes all insight generation steps."""
        self.generate_weekly_summaries(force=force)
        self.generate_monthly_summaries(force=force)
        self.generate_yearly_summaries(force=force)
        self.generate_lifetime_summary(force=force)

        self.generate_trajectories(force=force)

        self.generate_routines(force=force)
        print("Full insight pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate insights from memory.")
    parser.add_argument("--weekly", action="store_true", help="Generate weekly summaries")
    parser.add_argument("--monthly", action="store_true", help="Generate monthly summaries")
    parser.add_argument("--trajectories", action="store_true", help="Generate preference trajectories")
    parser.add_argument("--yearly", action="store_true", help="Generate yearly summaries")
    parser.add_argument("--lifetime", action="store_true", help="Generate lifetime summary")
    parser.add_argument("--routines", action="store_true", help="Generate routine analysis")
    parser.add_argument("--all", action="store_true", help="Generate all insights (default if no other flags)")
    parser.add_argument("--dataset", type=str, default="dataset_3", help="Dataset path (default: dataset_3)")
    parser.add_argument("--force", action="store_true", help="Force regeneration of existing insights")

    args = parser.parse_args()

    # If no specific flags, default to all
    if not any([args.weekly, args.monthly, args.trajectories, args.yearly, args.lifetime, args.routines]):
        args.all = True

    generator = InsightGenerator(dataset_path=args.dataset)

    if args.all:
        generator.run_full_pipeline(force=args.force)
    else:
        if args.weekly:
            generator.generate_weekly_summaries(force=args.force)
        if args.monthly:
            generator.generate_monthly_summaries(force=args.force)
        if args.yearly:
            generator.generate_yearly_summaries(force=args.force)
        if args.lifetime:
            generator.generate_lifetime_summary(force=args.force)
        if args.trajectories:
            generator.generate_trajectories(force=args.force)
        if args.routines:
            generator.generate_routines(force=args.force)
