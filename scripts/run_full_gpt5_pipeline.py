import os
import sys
import shutil
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from memory_v2.agent import LifelongAgent, TokenCounterCallback

# Set environment variable to fix MPS memory limit issues on Mac
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ── Model + Pricing ──────────────────────────────────────────────────────────
MODEL = "gemini-3-flash-preview"

# Gemini 3 Flash pricing (USD per 1M tokens)
INPUT_PRICE_PER_M  = 0.50   # $0.50 / 1M input tokens
OUTPUT_PRICE_PER_M = 3.00   # $3.00 / 1M output tokens
# ─────────────────────────────────────────────────────────────────────────────


def run():
    run_dir   = "results/legacy/memory_v2_run"
    store_dir = os.path.join(run_dir, "store")

    # Check if we should resume or start fresh
    db_exists = os.path.exists(os.path.join(store_dir, "memories.db"))

    if not db_exists:
        if os.path.exists(store_dir):
            shutil.rmtree(store_dir)
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

    # Skip ingestion if DB already has memories
    start, end = agent.store.get_date_range()
    if not start:
        print(f"--- INGESTING DATA FROM {dataset_path} ---")
        with open(dataset_path) as f:
            data = json.load(f)

        user_sessions = data["user_1"]["sessions"]
        for date in sorted(user_sessions.keys()):
            session = user_sessions[date]
            turns = [{"speaker": t["speaker"], "text": t["text"]} for t in session["turns"]]
            print(f"Ingesting {date}...")
            agent.ingest(date, [{"time_of_day": "All Day", "turns": turns}])
    else:
        print(f"--- DATABASE ALREADY CONTAINS DATA ({start} to {end}). SKIPPING INGESTION. ---")

    print("\n--- BUILDING SUMMARY HIERARCHY (RESUMING) ---")
    agent.build_summaries(force=False)

    print("\n--- EXPORTING FINAL RESULTS ---")
    summary_text = agent.store.get_lifetime_summary(speaker="user")
    if summary_text:
        output_path = os.path.join(run_dir, "lifetime_summary_math.txt")
        with open(output_path, "w") as f:
            f.write(summary_text)
        print(f"Final Lifetime Summary saved to {output_path}")

    # ── Cost Report ──────────────────────────────────────────────────────────
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
    print(f"  Input  cost   : ${input_cost:>12.6f}  (@ $0.50 / 1M)")
    print(f"  Output cost   : ${output_cost:>12.6f}  (@ $3.00 / 1M)")
    print(f"  TOTAL  COST   : ${total_cost:>12.6f}")
    print("=" * 52)

    print("\n--- PIPELINE COMPLETE ---")


if __name__ == "__main__":
    run()
