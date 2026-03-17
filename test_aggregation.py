import os
import sys
from datetime import datetime

# Add memory/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memory"))
from memory_manager import MemoryManager

def test_aggregation():
    # Use existing dataset_2 index
    index_path = "dataset_2/faiss_index"
    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}. Please run ingest_memory.py first.")
        return

    manager = MemoryManager(index_path=index_path)
    
    # Test query
    question = "How many times in the past month did I have coffee?"
    print(f"Testing Query: {question}")
    
    result = manager.query_analytics(question)
    
    print("\n--- AGGREGATION RESULT ---")
    print(result)
    print("--------------------------")

if __name__ == "__main__":
    test_aggregation()
