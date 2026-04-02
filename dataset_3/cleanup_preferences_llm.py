import os
import sys
import json
import sqlite3
from typing import List, Dict

# Add memory/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))
from structured_memory_pref import StructuredMemoryManagerPref

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "lifelong_memory.db")
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    manager = StructuredMemoryManagerPref(db_path=db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Fetch all existing preferences
    cursor.execute("SELECT id, entity, preference, category FROM preferences")
    rows = cursor.fetchall()
    
    if not rows:
        print("No preferences found to clean.")
        return
        
    print(f"Found {len(rows)} preference entries. Starting LLM-based cleanup...")
    
    # 2. Extract unique triples for the LLM to consolidate
    unique_prefs = []
    seen = set()
    for r in rows:
        triple = (r[1], r[2], r[3])
        if triple not in seen:
            unique_prefs.append({"entity": r[1], "preference": r[2], "category": r[3]})
            seen.add(triple)

    print(f"Unique preferences to consolidate: {len(unique_prefs)}")
    
    # 3. Use the manager's LLM consolidation method
    # Since we are cleaning the whole DB, we don't have "existing_context" yet, 
    # so we pass the unique prefs as both raw and context to find internal merges.
    consolidated = manager._consolidate_preferences_with_llm(unique_prefs, [])
    
    if not consolidated:
        print("Consolidation failed or returned no data.")
        return

    # Create a mapping from (old_entity, old_pref) -> (new_entity, new_pref)
    # We'll match based on the LLM's returned order or semantic matching if order isn't guaranteed.
    # To be safe, we'll ask the LLM to return the mapping explicitly or we'll do it manually if it matches 1:1.
    
    # Actually, let's refine the LLM call to be specific about returning a mapping for cleanup.
    # But for a small set, we can just look at the categories or do a second pass.
    
    # Let's use a more robust cleanup prompt for this one-time script.
    cleanup_prompt = (
        "You are cleaning a user's memory database. Group these preferences semantically. "
        "For each input, return the standardized 'new_entity' and 'new_preference'.\n\n"
        "INPUTS:\n{inputs}\n\n"
        "RULES:\n"
        "1. Map synonyms to a single entity (e.g., 'espresso' and 'latte' -> entity: 'coffee').\n"
        "2. If an entity is itself a preference for something else (e.g., 'flip flops' for 'shoes'), merge them.\n"
        "3. Return a JSON list of objects: {{'old_entity', 'old_preference', 'new_entity', 'new_preference', 'category'}}\n\n"
        "Return ONLY the JSON list."
    )
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", cleanup_prompt),
        ("human", "Process these entries.")
    ])
    
    try:
        chain = prompt | manager.llm | JsonOutputParser()
        mapping_list = chain.invoke({"inputs": json.dumps(unique_prefs)})
        
        # 4. Apply updates
        updates_count = 0
        for entry in mapping_list:
            old_ent = entry.get("old_entity")
            old_pref = entry.get("old_preference")
            new_ent = entry.get("new_entity")
            new_pref = entry.get("new_preference")
            new_cat = entry.get("category")
            
            if new_ent and new_pref:
                cursor.execute(
                    "UPDATE preferences SET entity = ?, preference = ?, category = ? WHERE entity = ? AND preference = ?",
                    (new_ent.lower(), new_pref.lower(), new_cat.lower() if new_cat else "other", old_ent, old_pref)
                )
                if cursor.rowcount > 0:
                    updates_count += cursor.rowcount
                    print(f"  Merged '{old_ent}/{old_pref}' -> '{new_ent}/{new_pref}'")

        conn.commit()
        print(f"\nCleanup complete. Total rows updated: {updates_count}")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
