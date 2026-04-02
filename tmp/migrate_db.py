import sqlite3
import os

db_path = "/Users/swanandpande/Documents/Coding/Projects/LifelongAgent/dataset_3/lifelong_memory.db"

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Converting all entries to lowercase...")
# Lowercase Preferences
cursor.execute("UPDATE preferences SET entity = LOWER(TRIM(entity)), preference = LOWER(TRIM(preference)), category = LOWER(TRIM(category))")
# Lowercase Tasks
cursor.execute("UPDATE tasks SET task_description = LOWER(TRIM(task_description)), task_type = LOWER(TRIM(task_type)), due_date = LOWER(TRIM(due_date))")
conn.commit()

print("Fixing entity consistency for preferences...")
# Get all current preference-entity mappings
cursor.execute("SELECT preference, entity, id FROM preferences ORDER BY source_date ASC, id ASC")
rows = cursor.fetchall()

pref_to_entity = {}
updates = []

# Populate the preference-to-entity map from all historical data
for pref, entity, row_id in rows:
    if pref not in pref_to_entity:
        pref_to_entity[pref] = entity

# Identify needed updates based on:
# 1. Preference mismatch
# 2. Level 2 Hierarchical Check (If the entity itself is a preference, normalize both)
for pref, entity, row_id in rows:
    new_entity = entity
    new_pref = pref
    
    # Check 1: Preference Consistency
    if pref in pref_to_entity:
        new_entity = pref_to_entity[pref]
    
    # Check 2: Level 2 Hierarchy (Is this entity actually a preference for something else?)
    if new_entity in pref_to_entity:
        # Standardize the hierarchy (e.g., 'flip flops' -> 'shoes')
        parent_entity = pref_to_entity[new_entity]
        if parent_entity != new_entity:
            new_pref = new_entity # Standard preference (e.g. 'flip flops')
            new_entity = parent_entity # Parent entity (e.g. 'shoes')

    if new_entity != entity or new_pref != pref:
        updates.append((new_entity, new_pref, row_id))

if updates:
    print(f"Found {len(updates)} entries to normalize.")
    for new_ent, new_pr, row_id in updates:
        cursor.execute("UPDATE preferences SET entity = ?, preference = ? WHERE id = ?", (new_ent, new_pr, row_id))
    conn.commit()
    print("Normalization complete.")
else:
    print("All preferences are already consistent.")

conn.close()
print("Migration script finished.")
