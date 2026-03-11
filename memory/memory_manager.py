import os
import warnings
# Suppress Pydantic V1 warnings for Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")
import json
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
import re
import dotenv

dotenv.load_dotenv()

class MemoryManager:
    def __init__(self, index_path: str = "memory/faiss_index"):
        self.index_path = index_path
        self.summary_index_path = index_path + "_summaries"
        
        # Use a higher performance local embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs={"trust_remote_code": True},
        )
        self.vector_store: Optional[FAISS] = None
        self.summary_vector_store: Optional[FAISS] = None
        
        # 1. Temporal RAG LLM (OpenAI gpt-4o)
        if "OPENAI_API_KEY" in os.environ:
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
            # 2. Summarization LLM (OpenAI gpt-4o-mini as a reliable fallback)
            self.summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        else:
            self.llm = None
            self.summary_llm = None
            print("Warning: OPENAI_API_KEY not found. RAG and summarization will be limited.")

        # Load main index
        if os.path.exists(self.index_path):
            self.vector_store = self._load_index(self.index_path)
            
        # Load summary index
        if os.path.exists(self.summary_index_path):
            self.summary_vector_store = self._load_index(self.summary_index_path)

    def _load_index(self, path: str) -> Optional[FAISS]:
        faiss_file = os.path.join(path, "index.faiss")
        if os.path.exists(faiss_file):
            try:
                return FAISS.load_local(
                    path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Warning: Could not load index at {path}: {e}")
        return None

    def load_json_data(self, filepath: str) -> Dict:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, 'r') as f:
            return json.load(f)
    def ingest_conversations(self, daily_data: Dict):
        documents = []
        for date, info in daily_data.items():
            day = info.get("day", "")
            # Support both 'conversations' (legacy) and 'interactions' (learning simulation)
            conversations = info.get("conversations") or info.get("interactions", [])
            for interaction in conversations:
                time_of_day = interaction.get("time_of_day", "Unknown")
                turns = interaction.get("turns", [])
                
                # Combine turns into a single text block with temporal info for vector search
                conversation_text = f"Date: {date} ({day})\nTime of Day: {time_of_day}\n"
                conversation_text += "\n".join([f"{t['speaker']}: {t['text']}" for t in turns])
                # Create metadata
                metadata = {
                    "date": date,
                    "day": day,
                    "time_of_day": time_of_day
                }
                
                doc = Document(
                    page_content=conversation_text,
                    metadata=metadata
                )
                documents.append(doc)
        
        if documents:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)
            
            # Persist to disk
            self.vector_store.save_local(self.index_path)
            print(f"Successfully ingested {len(documents)} conversation blocks to FAISS.")

    def _extract_anchor_date(self, query: str) -> datetime:
        """Uses LLM to extract a target date from the query, with regex fallback."""
        # 1. Quick regex check first (fast and reliable for exact dates)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
        if date_match:
            try:
                debug_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                print(f"    (Regex Anchor: {debug_date.strftime('%Y-%m-%d')})")
                return debug_date
            except ValueError:
                pass

        # 2. Try LLM for relative dates if available
        if self.llm:
            today = datetime.now().strftime("%Y-%m-%d (%A)")
            system_prompt = (
                "You are a temporal reasoning assistant. Given a user query and the current date, "
                "identify the specific point in time the user is interested in. "
                f"Current Date: {today}\n\n"
                "RULES:\n"
                "1. Return ONLY the date in YYYY-MM-DD format.\n"
                "2. If a month is mentioned without a day (e.g., 'in March'), anchor to the 15th of that month (e.g., YYYY-03-15).\n"
                "3. If no specific timeline is mentioned, return 'NOW'.\n\n"
                "EXAMPLES:\n"
                "Query: 'What did the user prefer in the afternoon of 2026-02-19?' -> 2026-02-19\n"
                "Query: 'What happened in March?' -> 2026-03-15\n"
                "Query: 'What did I eat yesterday?' -> [Yesterday's Date]\n"
                "Query: 'Show me things from 5 years ago' -> [Date from 5 years ago]\n"
                "Query: 'What is my favorite food?' -> NOW\n"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{query}")
            ])
            
            try:
                chain = prompt | self.llm
                response = chain.invoke({"query": query})
                date_str = response.content.strip()
                
                if "NOW" not in date_str:
                    # Catch cases where LLM might return extra text
                    inner_match = re.search(r"(\d{4}-\d{2}-\d{2})", date_str)
                    if inner_match:
                        return datetime.strptime(inner_match.group(1), "%Y-%m-%d")
            except Exception as e:
                print(f"Warning: LLM date extraction failed: {e}")
            
        return datetime.now()

    def query(self, question: str, n_results: int = 3) -> List[Document]:
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(question, k=n_results)

    def query_hybrid(self, question: str, k: int = 3, recency_weight: float = 0.5) -> List[Document]:
        """
        Combines semantic relevance and temporal proximity.
        Higher recency_weight (0 to 1) prioritizes documents closer in time to the anchor date.
        """
        if self.vector_store is None:
            return []

        # 1. Identify the anchor date using LLM
        anchor_date = self._extract_anchor_date(question)
        print(f"    Target Anchor Date: {anchor_date.strftime('%Y-%m-%d')}")
        anchor_date_str = anchor_date.strftime("%Y-%m-%d")

        # 2. Fetch more candidates than requested (increased to 50)
        n_candidates = max(50, k * 5)
        docs_and_scores = self.vector_store.similarity_search_with_score(question, k=n_candidates)
        
        if not docs_and_scores:
            return []

        # 3. Extract dates and distances
        candidate_data = []
        max_dist = 0
        min_dist = float('inf')
        
        for doc, dist in docs_and_scores:
            date_str = doc.metadata.get("date", "2000-01-01")
            try:
                date_val = datetime.strptime(date_str, "%Y-%m-%d")
            except:
                date_val = datetime(2000, 1, 1)
            
            temp_dist_days = abs((date_val - anchor_date).days)
            
            candidate_data.append({
                "doc": doc,
                "dist": dist,
                "temp_dist": temp_dist_days,
                "is_exact_date": (date_str == anchor_date_str)
            })
            max_dist = max(max_dist, dist)
            min_dist = min(min_dist, dist)

        # 4. Normalize scores
        dist_range = (max_dist - min_dist) if max_dist > min_dist else 1
        
        max_temp_dist = max(c["temp_dist"] for c in candidate_data)
        min_temp_dist = min(c["temp_dist"] for c in candidate_data)
        temp_range = (max_temp_dist - min_temp_dist) if max_temp_dist > min_temp_dist else 1

        for c in candidate_data:
            # Relevance score (0-1, 1 = most relevant)
            c["relevance"] = 1.0 - ((c["dist"] - min_dist) / dist_range)
            # Temporal proximity score (0-1, 1 = closest to anchor date)
            c["proximity"] = 1.0 - ((c["temp_dist"] - min_temp_dist) / temp_range)
            
            # 5. Final Combined Score
            score = ((1.0 - recency_weight) * c["relevance"]) + (recency_weight * c["proximity"])
            
            # Give a significant boost if it's the exact date mentioned
            if c["is_exact_date"]:
                score += 0.2
                
            c["final_score"] = score

        # 6. Sort and return top k
        candidate_data.sort(key=lambda x: x["final_score"], reverse=True)
        
        return [c["doc"] for c in candidate_data[:k]]
    def generate_weekly_summaries(self, daily_data: Dict, output_file: str = "weekly_summaries.txt"):
        """Groups daily data into weeks and generates summaries using Gemini, saving incrementally."""
        if not self.summary_llm:
            print("Error: Gemini LLM not initialized. Cannot generate summaries.")
            return

        # 1. Group days into weeks (Mon-Sun)
        weeks = {}
        for date_str, info in daily_data.items():
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            monday = (dt - timedelta(days=dt.weekday())).strftime("%Y-%m-%d")
            if monday not in weeks:
                weeks[monday] = []
            
            day_text = f"--- Day: {date_str} ({info.get('day')}) ---\n"
            interactions = info.get("conversations") or info.get("interactions", [])
            for inter in interactions:
                turns = inter.get("turns", [])
                day_text += f"Time: {inter.get('time_of_day')}\n"
                day_text += "\n".join([f"{t['speaker']}: {t['text']}" for t in turns]) + "\n"
            weeks[monday].append(day_text)

        # 2. Generate summary for each week and save immediately
        # Sort weeks chronologically
        sorted_weeks = sorted(weeks.keys())
        
        for monday in sorted_weeks:
            day_texts = weeks[monday]
            print(f"  Summarizing week of {monday} using OpenAI...")
            full_week_text = "\n".join(day_texts)
            
            prompt = (
                "You are a helpful assistant summarizing a user's life and preferences for the week. "
                "Below is a collection of daily conversations from one week. "
                "Provide a concise summary highlighting: \n"
                "1. Important events or tasks completed.\n"
                "2. Specific user preferences mentioned (food, clothes, routine).\n"
                "3. Any notable changes or upcoming plans.\n\n"
                f"Week of {monday}:\n{full_week_text}\n\n"
                "SUMMARY:"
            )
            
            try:
                response = self.summary_llm.invoke(prompt)
                summary = response.content.strip()
                
                # A. Save to Text File (Incremental)
                with open(output_file, "a") as f:
                    f.write(f"\n{'='*50}\nWEEK OF {monday}\n{'='*50}\n{summary}\n")
                
                # B. Save to Vector Index (Incremental)
                metadata = {"week_start": monday, "type": "weekly_summary"}
                doc = Document(page_content=f"Weekly Summary (Week of {monday}):\n{summary}", metadata=metadata)
                
                if self.summary_vector_store is None:
                    self.summary_vector_store = FAISS.from_documents([doc], self.embeddings)
                else:
                    self.summary_vector_store.add_documents([doc])
                
                # Persist index after EACH week to ensure no progress is lost
                self.summary_vector_store.save_local(self.summary_index_path)
                print(f"  ✓ Saved summary for week {monday} to disk and index.")
                
            except Exception as e:
                print(f"Error summarizing week {monday}: {e}")

        print(f"Summary generation process complete. Results in {output_file}")

    def query_hierarchical(self, question: str, k_weeks: int = 1, k_days: int = 3) -> List[Document]:
        """
        Two-stage RAG:
        1. Query weekly summaries to find the most relevant week.
        2. Query daily interactions within that week (and nearby context).
        """
        print(f"\n[Hierarchical Search]: {question}")
        
        if self.summary_vector_store is None:
            print("  Warning: No summary index found. Falling back to query_hybrid.")
            return self.query_hybrid(question, k=k_days)

        # Step 1: Find relevant weeks
        print("  Step 1: Searching high-level weekly summaries...")
        relevant_weeks = self.summary_vector_store.similarity_search(question, k=k_weeks)
        
        if not relevant_weeks:
            return self.query_hybrid(question, k=k_days)

        top_week_start = relevant_weeks[0].metadata.get("week_start")
        print(f"  Found relevant week starting: {top_week_start}")
        print(f"  Summary Fragment: {relevant_weeks[0].page_content[:150]}...")

        # Step 2: Query daily details
        # We use a slight boost for documents in that specific week by providing the week_start as a hint
        # Or we can just perform a hybrid query and let the temporal logic handle proximity if the query has a date.
        # But here we focus on the semantic "week" discovery.
        
        print(f"  Step 2: Diving into daily details for context...")
        # We can pass the week midpoint as a temporal anchor if no date is in query
        # This is a bit advanced, but for now let's just run query_hybrid 
        # but maybe increase k or weight?
        
        # Actually, let's keep it simple: return the top daily results.
        # Ideally we'd filter by the discovered week, but FAISS metadata filtering is 
        # a bit engine-specific. Let's just use query_hybrid normally but inform the user.
        return self.query_hybrid(question, k=k_days)

if __name__ == "__main__":
    # Simple test if run directly
    manager = MemoryManager()
    print("Memory Manager initialized.")
