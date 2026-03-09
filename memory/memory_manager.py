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
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
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

if __name__ == "__main__":
    # Simple test if run directly
    manager = MemoryManager()
    print("Memory Manager initialized.")
