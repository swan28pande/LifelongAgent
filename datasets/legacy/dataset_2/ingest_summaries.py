import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")

# Allow importing MemoryManager from the shared memory/ module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))
from memory_manager import MemoryManager

def main():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    # Dataset 2 uses learning_conversations.json
    json_path  = os.path.join(base_dir, "learning_conversations.json")
    index_path = os.path.join(base_dir, "faiss_index")

    print(f"Initializing Memory Manager for Summarization...")
    manager = MemoryManager(index_path=index_path)

    if not manager.summary_llm:
        print("Error: GOOGLE_API_KEY not found in environment. Cannot proceed.")
        return

    print(f"Loading data from {json_path}...")
    try:
        data = manager.load_json_data(json_path)
        print(f"Found {len(data)} days of conversations.")

        print("Generating and Indexing Weekly Summaries (High-Level Memory)...")
        manager.generate_weekly_summaries(data)
        print("Ingestion complete.")

    except Exception as e:
        print(f"Error during summary ingestion: {e}")

if __name__ == "__main__":
    main()
