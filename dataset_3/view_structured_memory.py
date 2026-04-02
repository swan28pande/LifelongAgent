import os
import sys

# Add memory/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../memory"))
from structured_memory import StructuredMemoryManager

def view_tables():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lifelong_memory.db")
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}. Please run ingest_structured.py first.")
        return

    manager = StructuredMemoryManager(db_path=db_path)
    
    print("\n--- RECENT PREFERENCES (Top 10) ---")
    prefs = manager.query_preferences()
    for row in prefs[:10]:
        print(f"ID: {row[0]} | Entity: {row[1]} | Preference: {row[2]} | Category: {row[3]} | Source: {row[5]}")

if __name__ == "__main__":
    view_tables()
