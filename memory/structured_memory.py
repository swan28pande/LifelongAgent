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
                source_date TEXT
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
                source_date TEXT
            )
        ''')
        
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
            extracted = chain.invoke({"text": full_text})
            
            self._save_to_db(date_str, extracted)
            return extracted
        except Exception as e:
            print(f"Error extracting structured memory for {date_str}: {e}")
            return None

    def _save_to_db(self, date_str: str, data: Dict):
        """Persists extracted data to SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save Preferences
        for pref in data.get("preferences", []):
            cursor.execute(
                "INSERT INTO preferences (entity, preference, category, source_date) VALUES (?, ?, ?, ?)",
                (pref.get("entity"), pref.get("preference"), pref.get("category"), date_str)
            )
            
        # Save Tasks
        for task in data.get("tasks", []):
            cursor.execute(
                "INSERT INTO tasks (task_description, task_type, due_date, source_date) VALUES (?, ?, ?, ?)",
                (task.get("description"), task.get("type"), task.get("due_date"), date_str)
            )
            
        conn.commit()
        conn.close()

    def query_preferences(self, category: Optional[str] = None):
        """Query stored preferences."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if category:
            cursor.execute("SELECT * FROM preferences WHERE category = ?", (category,))
        else:
            cursor.execute("SELECT * FROM preferences")
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
