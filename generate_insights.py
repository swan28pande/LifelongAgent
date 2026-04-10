import os
import sys
import sqlite3
import json
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
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        else:
            self.llm = None
            print("Warning: OPENAI_API_KEY not found.")

        self.insights_data = self._load_insights_json()

    def _load_insights_json(self) -> Dict:
        if os.path.exists(self.insights_json_path):
            with open(self.insights_json_path, 'r') as f:
                return json.load(f)
        return {"weekly": {}, "monthly": {}, "trajectories": {}, "routines": {}}

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

    def generate_weekly_summaries(self):
        """Generates summaries for each week that hasn't been summarized yet."""
        if not self.llm or not os.path.exists(self.conv_path):
            return
            
        with open(self.conv_path, 'r') as f:
            all_convs = json.load(f)
            
        # Group by week
        weeks = {}
        # Keep track of which month each week primarily belongs to
        week_months = {}
        for date_str, info in all_convs.items():
            week_id = self._get_iso_week(date_str)
            if week_id not in weeks:
                weeks[week_id] = []
                # Anchor the week to the month of its first occurrence in the dataset
                week_months[week_id] = date_str[:7] # YYYY-MM
            
            day_text = f"--- Date: {date_str} ({info.get('day')}) ---\n"
            for inter in info.get("interactions", []):
                turns = inter.get("turns", [])
                day_text += f"Time: {inter.get('time_of_day')}\n"
                day_text += "\n".join([f"{t['speaker']}: {t['text']}" for t in turns]) + "\n"
            weeks[week_id].append(day_text)

        for week_id, day_texts in sorted(weeks.items()):
            # Re-generate even if exists to ensure latest info (Redundancy check handled by _save_to_rag)
            print(f"Generating summary for week {week_id}...")
            full_text = "\n".join(day_texts)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a memory analyst. Summarize this specific week of user interactions."),
                ("human", (
                    "Below is a week of user interactions. Generate a concise but insightful summary highlighting:\n"
                    "1. Key events and activities.\n"
                    "2. Notable user preferences mentioned.\n"
                    "3. Any progression or changes in the user's routine.\n\n"
                    f"Week Content:\n{full_text}\n\n"
                    "Return ONLY a JSON object: {{\"title\": \"...\", \"summary\": \"...\"}}"
                ))
            ])
            
            try:
                chain = prompt | self.llm | JsonOutputParser()
                res = chain.invoke({})
                
                title = res.get("title", f"Weekly Summary: {week_id}")
                content = res.get("summary", "")
                month_id = week_months[week_id]
                
                # Store
                self.insights_data["weekly"][week_id] = {
                    "title": title, 
                    "content": content,
                    "month_id": month_id
                }
                self._save_to_rag("weekly-summary", title, content, week_id, {"month_id": month_id})
                self._save_insights_json()
            except Exception as e:
                print(f"Error for week {week_id}: {e}")

    def generate_monthly_summaries(self):
        """Generates monthly snapshots based on existing weekly summaries."""
        if not self.llm:
            return
            
        # Group weekly summaries by month using the stored month_id
        months = {}
        for week_id, data in self.insights_data["weekly"].items():
            month_id = data.get("month_id")
            if not month_id:
                continue # Skip if no month data (compat for old runs)
            
            if month_id not in months:
                months[month_id] = []
            months[month_id].append(f"{week_id}: {data['content']}")

        for month_id, weekly_texts in sorted(months.items()):
            # Re-generate even if exists to ensure latest info
            print(f"Generating summary for month {month_id}...")
            weekly_blob = "\n\n".join(weekly_texts)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a master memory analyst. Synthesize monthly snapshots from weekly summaries."),
                ("human", (
                    "Below are weekly summaries for the month. Provide a high-level monthly snapshot of the user's life and evolution.\n\n"
                    f"Weekly Summaries:\n{weekly_blob}\n\n"
                    "Return ONLY a JSON object: {{\"title\": \"...\", \"summary\": \"...\"}}"
                ))
            ])
            
            try:
                chain = prompt | self.llm | JsonOutputParser()
                res = chain.invoke({})
                
                title = res.get("title", f"Monthly Snapshot: {month_id}")
                content = res.get("summary", "")
                
                # Store
                self.insights_data["monthly"][month_id] = {"title": title, "content": content}
                self._save_to_rag("monthly-summary", title, content, month_id)
                self._save_insights_json()
            except Exception as e:
                print(f"Error for month {month_id}: {e}")

    def generate_trajectories(self, entities: List[str]):
        """Generates/Updates preference trajectories for specific entities (e.g. coffee)."""
        if not self.llm:
            return
            
        for entity in entities:
            print(f"Analyzing trajectory for {entity}...")
            
            # 1. Fetch last 10 from SQL
            conn = sqlite3.connect(self.sql_path)
            cursor = conn.cursor()
            cursor.execute("SELECT source_date, preference FROM preferences WHERE entity = ? ORDER BY source_date DESC LIMIT 10", (entity,))
            rows = cursor.fetchall()
            conn.close()
            
            if not rows: continue
            
            # 2. Fetch context from RAG
            docs = self.rag_manager.query(f"user preference for {entity}", n_results=10)
            rag_blob = "\n".join([d.page_content for d in docs])
            
            history = "\n".join([f"{r[0]}: {r[1]}" for r in reversed(rows)])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a preference analyst. Trace the evolution of a user's choice over time. Identify temporal patterns and quantifiable shifts."),
                ("human", (
                    f"Analyze the trajectory of user's preference for '{entity}'.\n\n"
                    f"Recorded Preferences (SQL):\n{history}\n\n"
                    f"Additional Context (RAG):\n{rag_blob}\n\n"
                    "Identify the pattern or evolution (e.g., 'Moving from casual to expert', 'Consistent but switching colors').\n"
                    "Provide a 'guess in pattern' that is quantifiable where possible (e.g. 'User preferred espresso for this week but changed to latte because...').\n\n"
                    "Return ONLY a JSON object: {{\"title\": \"...\", \"trajectory\": \"...\"}}"
                ))
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

    def generate_routines(self):
        """Generates/Updates temporal routine fingerprints (multi-preference synthesis)."""
        if not self.llm:
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

    def run_full_pipeline(self):
        """Executes all insight generation steps."""
        self.generate_weekly_summaries()
        self.generate_monthly_summaries()
        
        # Dynamically fetch last 10 unique entities from SQL
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()
        cursor.execute("SELECT entity FROM preferences GROUP BY entity ORDER BY MAX(source_date) DESC LIMIT 10")
        entities = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"Generating trajectories for detected entities: {entities}")
        self.generate_trajectories(entities)
        
        self.generate_routines()
        print("Full insight pipeline complete.")

if __name__ == "__main__":
    generator = InsightGenerator()
    generator.run_full_pipeline()
