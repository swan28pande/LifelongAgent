import sqlite3
import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import dotenv

dotenv.load_dotenv()

class StructuredMemoryManager:
    def __init__(self, db_path: str = "memory/lifelong_memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        
        # Initialize LLM for extraction
        if "OPENAI_API_KEY" in os.environ:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        else:
            self.llm = None
            print("Warning: OPENAI_API_KEY not found. Structured extraction will be unavailable.")

    def _init_db(self):
        """Initializes the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for Preferences
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT,
                preference TEXT,
                category TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_date TEXT,
                time_of_day TEXT
            )
        ''')
        
        # Table for Tasks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_description TEXT,
                task_type TEXT, -- 'ad-hoc', 'repetitive'
                status TEXT DEFAULT 'pending',
                due_date TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_date TEXT,
                time_of_day TEXT
            )
        ''')
        
        # Migration: Ensure time_of_day column exists in older DBs
        for table in ["preferences", "tasks"]:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]
            if "time_of_day" not in columns:
                print(f"Migrating table '{table}': Adding 'time_of_day' column...")
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN time_of_day TEXT DEFAULT 'Unknown'")
        
        conn.commit()
        conn.close()

    def extract_and_store(self, date_str: str, interactions: List[Dict]):
        """
        Uses an LLM agent to extract preferences and tasks from a day's interactions
        and stores them in the SQL database.
        """
        if not self.llm:
            return

        # Prepare text for LLM
        full_text = f"Date: {date_str}\n"
        for inter in interactions:
            turns = inter.get("turns", [])
            full_text += f"\nTime: {inter.get('time_of_day')}\n"
            full_text += "\n".join([f"{t['speaker']}: {t['text']}" for t in turns]) + "\n"

        system_prompt = (
            "You are a structured memory extraction agent. Your goal is to identify user preferences "
            "and tasks (ad-hoc or repetitive) from the conversation provided.\n\n"
            "Return a JSON object with two keys:\n"
            "1. 'preferences': A list of objects with {{'entity', 'preference', 'category'}}.\n"
            "2. 'tasks': A list of objects with {{'description', 'type', 'due_date'}}.\n\n"
            "TASK NAMES:\n"
            "- Keep task descriptions extremely concise and normalized.\n"
            "- Example: Instead of 'do some coding' or 'coding for today', use 'coding'.\n"
            "- Example: Instead of 'buy some milk from the store', use 'buy milk'.\n\n"
            "TASK TYPES:\n"
            "- 'ad-hoc': One-time tasks (e.g., 'buy eggs today').\n"
            "- 'repetitive': Recurring tasks (e.g., 'exercise every Monday').\n\n"
            "CATEGORIES for preferences: 'food', 'routine', 'clothing', 'hobbies', 'other'.\n\n"
            "If no preferences or tasks are found, return empty lists."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Extract from these conversations:\n\n{text}")
        ])

        try:
            chain = prompt | self.llm | JsonOutputParser()
            
            total_extracted = {"preferences": [], "tasks": []}
            
            # Extract per time_block to ensure accurate temporal metadata
            for interaction in interactions:
                time_val = interaction.get("time_of_day", "Unknown")
                # Prepare text for just this block
                turns = interaction.get("turns", [])
                block_text = f"Date: {date_str}\nTime: {time_val}\n"
                block_text += "\n".join([f"{t['speaker']}: {t['text']}" for t in turns])
                
                extracted = chain.invoke({"text": block_text})
                self._save_to_db(date_str, time_val, extracted)
                
                total_extracted["preferences"].extend(extracted.get("preferences", []))
                total_extracted["tasks"].extend(extracted.get("tasks", []))
                
            return total_extracted
        except Exception as e:
            print(f"Error extracting structured memory for {date_str}: {e}")
            return None

    def delete_by_date(self, date_str: str):
        """Deletes all entries for a specific date from preferences and tasks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM preferences WHERE source_date = ?", (date_str,))
        cursor.execute("DELETE FROM tasks WHERE source_date = ?", (date_str,))
        
        conn.commit()
        conn.close()
        print(f"Deleted SQL entries for date {date_str}.")

    def _save_to_db(self, date_str: str, time_of_day: str, data: Dict):
        """Persists extracted data to SQLite with semantic consolidation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. Handle Preferences Consolidation
        raw_preferences = data.get("preferences", [])
        if raw_preferences:
            # Fetch targeted context for LLM comparison
            existing_prefs = self._get_relevant_context(raw_preferences)
            consolidated_prefs = self._consolidate_preferences_with_llm(raw_preferences, existing_prefs)
            
            for pref in consolidated_prefs:
                entity = str(pref.get("entity", "")).lower().strip()
                preference = str(pref.get("preference", "")).lower().strip()
                category = str(pref.get("category", "")).lower().strip()

                cursor.execute(
                    "INSERT INTO preferences (entity, preference, category, source_date, time_of_day) VALUES (?, ?, ?, ?, ?)",
                    (entity, preference, category, date_str, time_of_day)
                )
            
        # 2. Save Tasks
        for task in data.get("tasks", []):
            description = task.get("description")
            new_due = task.get("due_date")
            new_type = task.get("type")
            
            # Check if this task exists in the DB already to reuse the due_date and task_type
            cursor.execute(
                "SELECT task_type, due_date FROM tasks WHERE task_description = ? ORDER BY source_date DESC LIMIT 1",
                (description,)
            )
            existing = cursor.fetchone()
            
            # Reuse existing values if found for structural consistency
            final_due = new_due
            final_type = new_type
            if existing:
                final_type = existing[0]
                final_due = existing[1]

            cursor.execute(
                "INSERT INTO tasks (task_description, task_type, due_date, source_date, time_of_day) VALUES (?, ?, ?, ?, ?)",
                (description, final_type, final_due, date_str, time_of_day)
            )
            
        conn.commit()
        conn.close()

    def _get_relevant_context(self, raw_preferences: List[Dict]) -> List[Dict]:
        """Fetches 10-15 most relevant historical records based on keywords in new data."""
        keywords = set()
        for p in raw_preferences:
            if p.get("entity"): keywords.add(str(p["entity"]).lower())
            if p.get("preference"): keywords.add(str(p["preference"]).lower())
        
        if not keywords: return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        relevant_context = []
        seen = set()
        
        for word in keywords:
            if len(word) < 3: continue 
            cursor.execute(
                "SELECT DISTINCT entity, preference, category FROM preferences "
                "WHERE entity LIKE ? OR preference LIKE ? LIMIT 5",
                (f"%{word}%", f"%{word}%")
            )
            for row in cursor.fetchall():
                triple = (row[0], row[1], row[2])
                if triple not in seen:
                    relevant_context.append({"entity": row[0], "preference": row[1], "category": row[2]})
                    seen.add(triple)
        conn.close()
        return relevant_context[:15]

    def _consolidate_preferences_with_llm(self, raw_preferences: List[Dict], existing_context: List[Dict]) -> List[Dict]:
        """Uses the LLM to map raw preference extractions to existing ones semantically."""
        if not self.llm or not raw_preferences:
            return raw_preferences

        system_prompt = (
            "You are a memory consolidation agent. Your goal is to map new preference extractions "
            "to our existing database schema to ensure consistency. Do not hardcode rules; "
            "instead, look at the existing entries provided and decide if the new extraction is a synonym, sub-item, "
            "or a specialized version of something we already track.\n\n"
            "EXISTING PREFERENCES (Entity | Preference | Category):\n"
            "{context}\n\n"
            "CONSOLIDATION GUIDELINES:\n"
            "1. If a new extraction (e.g., 'espresso') is semantically a type of an existing entity (e.g., 'coffee'), "
            "use the existing entity as 'entity' (Coffee) and the new extraction as 'preference' (Espresso).\n"
            "2. If the new 'entity' (e.g., 'flip flops') is already a known 'preference' for something else (e.g., 'shoes'), "
            "standardize the entity to the parent ('shoes') and set the preference to 'flip flops'.\n"
            "3. For categories, use the EXISTING category if it matches the entity.\n"
            "4. Return ONLY a JSON list of objects with {{'entity', 'preference', 'category'}}."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Consolidate these extractions:\n{raw_json}")
        ])

        try:
            context_str = "\n".join([f"- {p['entity']} | {p['preference']} | {p['category']}" for p in existing_context])
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({"context": context_str, "raw_json": json.dumps(raw_preferences)})
            return result.get("preferences", result) if isinstance(result, dict) else result
        except Exception as e:
            print(f"Warning: Consolidation failed: {e}")
            return raw_preferences

    def query_preferences(self, category: Optional[str] = None, time_of_day: Optional[str] = None):
        """Query stored preferences with optional category and time_of_day filtering."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM preferences WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        if time_of_day:
            query += " AND time_of_day = ?"
            params.append(time_of_day)
            
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return rows

    def query_tasks(self, task_type: Optional[str] = None):
        """Query stored tasks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if task_type:
            cursor.execute("SELECT * FROM tasks WHERE task_type = ?", (task_type,))
        else:
            cursor.execute("SELECT * FROM tasks")
        rows = cursor.fetchall()
        conn.close()
        return rows


    def get_schema_info(self) -> str:
        """Returns a string description of the SQLite schema."""
        return (
            "TABLE preferences:\n"
            "- entity (TEXT): The thing the preference is about (e.g., 'coffee', 't-shirt')\n"
            "- preference (TEXT): The specific choice (e.g., 'espresso', 'red')\n"
            "- category (TEXT): 'food', 'routine', 'clothing', 'hobbies', 'other'\n"
            "- source_date (TEXT): YYYY-MM-DD\n\n"
            "TABLE tasks:\n"
            "- task_description (TEXT): What the task is (e.g., 'coding', 'running')\n"
            "- task_type (TEXT): 'ad-hoc' or 'repetitive'\n"
            "- status (TEXT): 'pending', 'completed'\n"
            "- due_date (TEXT): YYYY-MM-DD\n"
            "- source_date (TEXT): YYYY-MM-DD"
        )

    def generate_insight_questions(self, num_questions: int = 10) -> List[str]:
        """Generates 10 insightful questions based on the schema."""
        if not self.llm:
            return []
            
        system_prompt = (
            "You are a data analyst for a lifelong memory system. Based on the database schema provided, "
            f"generate {num_questions} insightful, analytical questions that would help a user understand their habits, "
            "consistency, and patterns over time.\n\n"
            "SCHEMA:\n{schema}\n\n"
            "Return ONLY a JSON list of strings."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Generate {num_questions} questions.")
        ])
        
        try:
            chain = prompt | self.llm | JsonOutputParser()
            return chain.invoke({"schema": self.get_schema_info(), "num_questions": num_questions})
        except Exception as e:
            print(f"Error generating insight questions: {e}")
            return []

    def execute_ai_sql(self, question: str) -> Dict:
        """Generates and executes a SQL query to answer a specific question."""
        if not self.llm:
            return {"error": "LLM not available"}

        system_prompt = (
            "You are a SQL expert. Given a question about a user's memory and the schema below, "
            "generate a valid SQLite query to answer the question.\n\n"
            "SCHEMA:\n{schema}\n\n"
            "RETURN ONLY THE SQL QUERY. NO MARKDOWN. NO EXPLANATION."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Question: {question}")
        ])
        
        try:
            chain = prompt | self.llm
            sql_query = chain.invoke({"schema": self.get_schema_info(), "question": question}).content.strip()
            # Clean markdown if present
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].split("```")[0].strip()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            
            return {
                "question": question,
                "sql": sql_query,
                "columns": columns,
                "results": rows
            }
        except Exception as e:
            return {"question": question, "error": str(e)}

if __name__ == "__main__":
    # Test initialization
    manager = StructuredMemoryManager()
    print("Structured Memory Manager initialized and DB created.")
