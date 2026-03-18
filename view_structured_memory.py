import os
import sys

# Add memory/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memory"))
from structured_memory import StructuredMemoryManager

def view_tables():
    db_path = "dataset_2/lifelong_memory.db"
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}. Please run ingest_structured.py first.")
        return

    manager = StructuredMemoryManager(db_path=db_path)
    
    print("\n--- RECENT PREFERENCES (Top 10) ---")
    prefs = manager.query_preferences()
    for row in prefs[:10]:
        print(f"ID: {row[0]} | Entity: {row[1]} | Preference: {row[2]} | Category: {row[3]} | Source: {row[5]}")

    print("\n--- EXTRACTED TASKS (Top 10) ---")
    tasks = manager.query_tasks()
    for row in tasks[:10]:
        print(f"ID: {row[0]} | Task: {row[1]} | Type: {row[2]} | Due: {row[4]} | Priority: {row[5]} | Date: {row[7]}")

if __name__ == "__main__":
    view_tables()
