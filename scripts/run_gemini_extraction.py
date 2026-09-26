import os
import sys
import shutil
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from memory_v2.agent import LifelongAgent, TokenCounterCallback

# Set environment variable to fix MPS memory limit issues on Mac
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Using the requested model
MODEL = "gemini-3.1-pro-preview"

# Pricing (USD per 1M tokens)
INPUT_PRICE_PER_M  = 2.00
OUTPUT_PRICE_PER_M = 12.00

def run():
    run_dir   = "results/legacy/gemini_extraction_run"
    store_dir = os.path.join(run_dir, "store")

    # Clear existing data for a fresh run
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    tracker = TokenCounterCallback()

    print(f"--- INITIALIZING {MODEL.upper()} AGENT ---")
    agent = LifelongAgent(
        base_dir=store_dir,
        chat_model=MODEL,
        extract_model=MODEL,
        callbacks=[tracker],
    )

    dataset_path = os.path.join(PROJECT_ROOT, "datasets", "eval", "conversations.json")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"--- PHASE 1: EXTRACTION FROM {dataset_path} ---")
    with open(dataset_path) as f:
        data = json.load(f)

    user_sessions = data["user_1"]["sessions"]
    # Run on the first 30 days of the dataset
    dates = sorted(user_sessions.keys())[:30]
    for date in dates:
        session = user_sessions[date]
        turns = [{"speaker": t["speaker"], "text": t["text"]} for t in session["turns"]]
        print(f"Ingesting {date}...")
        agent.ingest(date, [{"time_of_day": "All Day", "turns": turns}])

    print("\n--- PHASE 2: BUILDING SUMMARY HIERARCHY ---")
    # This automatically calls consolidate_memories first
    agent.build_summaries(force=True)

    print("\n--- EXPORTING EXTRACTED MEMORIES ---")
    memories = agent.store.query_memories(limit=5000)
    output_path = os.path.join(run_dir, "extracted_memories.json")
    with open(output_path, "w") as f:
        json.dump(memories, f, indent=2)
    print(f"Extracted memories saved to {output_path}")
    
    print("\n--- EXPORTING ALL SUMMARIES (WEEKLY, MONTHLY, YEARLY, LIFETIME) ---")
    summary_path = os.path.join(run_dir, "all_summaries.txt")
    with open(summary_path, "w") as f:
        if agent.store._summary_store:
            # Sort the summaries by their identifiers (e.g., week:2026-W09, month:2026-03)
            # This ensures they are printed chronologically and hierarchically
            sorted_docs = sorted(
                agent.store._summary_store.docstore._dict.values(),
                key=lambda doc: doc.metadata.get("identifier", "")
            )
            for doc in sorted_docs:
                f.write("=" * 80 + "\n")
                f.write(doc.page_content.strip() + "\n")
                f.write("=" * 80 + "\n\n")
            print(f"All summaries saved to {summary_path}")
        else:
            print("No summaries found in the database.")

    # Cost Report
    report        = tracker.get_report()
    input_tokens  = report["prompt_tokens"]
    output_tokens = report["completion_tokens"]
    total_tokens  = report["total_tokens"]

    input_cost  = (input_tokens  / 1_000_000) * INPUT_PRICE_PER_M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
    total_cost  = input_cost + output_cost

    print("\n" + "=" * 52)
    print(f"  💰  COST REPORT  ({MODEL})")
    print("=" * 52)
    print(f"  Input  tokens : {input_tokens:>12,}")
    print(f"  Output tokens : {output_tokens:>12,}")
    print(f"  Total  tokens : {total_tokens:>12,}")
    print(f"  Input  cost   : ${input_cost:>12.6f}  (@ ${INPUT_PRICE_PER_M:.2f} / 1M)")
    print(f"  Output cost   : ${output_cost:>12.6f}  (@ ${OUTPUT_PRICE_PER_M:.2f} / 1M)")
    print(f"  TOTAL  COST   : ${total_cost:>12.6f}")
    print("=" * 52)

    # Call the external evaluation module
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from memory_v2.evaluate import run_evaluation
    
    qa_path = os.path.join(PROJECT_ROOT, "datasets", "eval", "qa_pairs.json")
    run_evaluation(store_dir, qa_path, MODEL)

    print("\n--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    run()

