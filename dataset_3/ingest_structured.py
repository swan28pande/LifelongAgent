import os
import sys
import json
from typing import Dict

# Add memory/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))
from structured_memory_pref import StructuredMemoryManagerPref

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "learning_conversations.json")
    db_path = os.path.join(base_dir, "lifelong_memory.db")

    print(f"Initializing Structured Memory Manager with DB at {db_path}...")
    manager = StructuredMemoryManagerPref(db_path=db_path)

    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Found {len(data)} days of interactions to process.")
    
    # Process only a few days for testing if the dataset is large
    # or process all if it's manageable. 
    # For now, let's process all but add a progress tracker.
    
    sorted_dates = sorted(data.keys())
    
    for i, date_str in enumerate(sorted_dates):
        print(f"[{i+1}/{len(data)}] Extracting structured data for {date_str}...")
        info = data[date_str]
        interactions = info.get("interactions", [])
        
        extracted = manager.extract_and_store(date_str, interactions)
        
        if extracted:
            prefs_count = len(extracted.get("preferences", []))
            tasks_count = len(extracted.get("tasks", []))
            print(f"  ✓ Extracted {prefs_count} preferences and {tasks_count} tasks.")
        else:
            print(f"  × Extraction failed for {date_str}.")

    print("\nIngestion into SQL complete.")

if __name__ == "__main__":
    main()
