import os
import json
from datetime import datetime, timedelta
import sys

# Ensure memory_v2 is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from memory_v2 import LifelongAgent
import shutil

# Ground Truth Patterns from generate_eval_dataset.py
BASE = datetime(2026, 3, 1)
def date_of(day: int) -> str:
    return (BASE + timedelta(days=day - 1)).strftime("%Y-%m-%d")

def week(day: int) -> int: return (day - 1) // 7 + 1
def dow(day: int) -> int: return ((BASE + timedelta(days=day - 1)).weekday()) + 1

def coffee_pattern(day):
    return "oat milk latte" if week(day) % 2 == 1 else "black coffee"

def clothing_pattern(day):
    return "fitted t-shirt" if (day - 1) % 6 < 3 else "oversized hoodie"

def exercise_pattern(day):
    d = dow(day)
    return {1: "morning yoga", 2: "evening run", 3: "morning yoga", 
            4: "evening run", 5: "morning yoga", 6: "climbing gym", 7: "rest day"}[d]

def verify():
    tmp_dir = "results/verify_run"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    # Load Dataset
    with open(os.path.join(PROJECT_ROOT, "datasets", "eval", "conversations.json")) as f:
        user_data = json.load(f)["user_1"]

    agent = LifelongAgent(base_dir=tmp_dir)
    
    print("Ingesting sessions...")
    dates = sorted(user_data["sessions"].keys())
    for date in dates:
        session = user_data["sessions"][date]
        turns = [{"speaker": t["speaker"], "text": t["text"]} for t in session["turns"]]
        agent.ingest(date, [{"time_of_day": "Morning", "turns": turns}])
        print(f"  Processed {date}")

    # Query all memories
    memories = agent.store.query_memories(limit=1000)
    
    print("\n" + "="*60)
    print("EXTRACTION VERIFICATION REPORT")
    print("="*60)
    
    errors = []
    successes = 0
    total_checks = 0

    # Group extracted memories by date
    extracted_by_date = {}
    for m in memories:
        d = m["date"]
        if d not in extracted_by_date: extracted_by_date[d] = []
        extracted_by_date[d].append(m)

    for day in range(1, 22):
        date = date_of(day)
        expected = {
            "coffee": coffee_pattern(day),
            "clothing": clothing_pattern(day),
            "exercise": exercise_pattern(day)
        }
        
        mems = extracted_by_date.get(date, [])
        
        for domain, exp_val in expected.items():
            if exp_val == "rest day":
                continue
                
            total_checks += 1
            # Simple keyword match
            found = any(exp_val.lower() in m["content"].lower() for m in mems)
            
            if found:
                successes += 1
            else:
                errors.append(f"Day {day:02d} ({date}): Missing {domain:8s} preference. Expected '{exp_val}'")

    # Check for domain force-fitting (e.g. food in beverage)
    for m in memories:
        content = m["content"].lower()
        subject = m["subject"].lower()
        if any(food in content for food in ["meal", "taco", "stir-fry", "vegetarian"]):
            if subject == "beverage":
                errors.append(f"FORCE-FIT ERROR: '{m['content']}' was extracted as subject 'beverage' on {m['date']}")

    print(f"\nScore: {successes}/{total_checks} preferences correctly identified.")
    if errors:
        print("\nIssues Found:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\nNo issues found! Extraction is perfectly aligned with ground truth.")

if __name__ == "__main__":
    verify()
