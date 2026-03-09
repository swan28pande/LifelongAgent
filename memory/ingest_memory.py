import os
import warnings
# Suppress Pydantic V1 warnings for Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")
from memory_manager import MemoryManager

def main():
    json_path = "learning_conversations.json"
    index_path = "memory/faiss_index"
    
    print(f"Initializing Memory Manager with index at {index_path}...")
    manager = MemoryManager(index_path=index_path)
    
    print(f"Loading data from {json_path}...")
    try:
        data = manager.load_json_data(json_path)
        print(f"Found {len(data)} days of conversations.")
        
        print("Starting ingestion into vector store...")
        manager.ingest_conversations(data)
        print("Ingestion complete.")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    main()
