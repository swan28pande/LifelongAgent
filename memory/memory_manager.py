import os
import warnings
# Suppress Pydantic V1 warnings for Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")
import json
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class MemoryManager:
    def __init__(self, index_path: str = "memory/faiss_index"):
        self.index_path = index_path
        # Use a higher performance local embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs={"trust_remote_code": True},
        )
        self.vector_store: Optional[FAISS] = None
        
        if os.path.exists(self.index_path):
            faiss_file = os.path.join(self.index_path, "index.faiss")
            if os.path.exists(faiss_file):
                try:
                    self.vector_store = FAISS.load_local(
                        self.index_path, 
                        self.embeddings,
                        allow_dangerous_deserialization=True # Required for local FAISS
                    )
                except Exception as e:
                    print(f"Warning: Could not load local index: {e}")
            else:
                print(f"Index directory exists but {faiss_file} not found. Will create new index.")

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

    def query(self, question: str, n_results: int = 3) -> List[Document]:
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(question, k=n_results)

    def query_hybrid(self, question: str, k: int = 3, recency_weight: float = 0.3) -> List[Document]:
        """
        Combines semantic relevance and recency.
        Higher recency_weight (0 to 1) prioritizes newer documents.
        """
        if self.vector_store is None:
            return []

        # 1. Fetch more candidates than requested (top 20 or k*4)
        n_candidates = max(20, k * 4)
        # similarity_search_with_score returns (doc, distance)
        # FAISS distance is L2 (lower is better/closer)
        docs_and_scores = self.vector_store.similarity_search_with_score(question, k=n_candidates)
        
        if not docs_and_scores:
            return []

        # 2. Extract dates and distances
        from datetime import datetime
        candidate_data = []
        max_dist = 0
        min_dist = float('inf')
        
        for doc, dist in docs_and_scores:
            date_str = doc.metadata.get("date", "2000-01-01")
            try:
                date_val = datetime.strptime(date_str, "%Y-%m-%d")
            except:
                date_val = datetime(2000, 1, 1)
            
            candidate_data.append({
                "doc": doc,
                "dist": dist,
                "date": date_val
            })
            max_dist = max(max_dist, dist)
            min_dist = min(min_dist, dist)

        # 3. Normalize Relevance (0-1, where 1 is most relevant)
        # Avoid division by zero
        dist_range = (max_dist - min_dist) if max_dist > min_dist else 1
        
        # 4. Normalize Recency (0-1, where 1 is most recent)
        latest_date = max(c["date"] for c in candidate_data)
        earliest_date = min(c["date"] for c in candidate_data)
        date_range = (latest_date - earliest_date).total_seconds() or 1

        for c in candidate_data:
            # Relevance: lower distance = higher score
            c["relevance"] = 1.0 - ((c["dist"] - min_dist) / dist_range)
            # Recency: newer date = higher score
            c["recency"] = (c["date"] - earliest_date).total_seconds() / date_range
            
            # 5. Final Combined Score
            c["final_score"] = ((1.0 - recency_weight) * c["relevance"]) + (recency_weight * c["recency"])

        # 6. Sort and return top k
        candidate_data.sort(key=lambda x: x["final_score"], reverse=True)
        return [c["doc"] for c in candidate_data[:k]]

if __name__ == "__main__":
    # Simple test if run directly
    manager = MemoryManager()
    print("Memory Manager initialized.")
