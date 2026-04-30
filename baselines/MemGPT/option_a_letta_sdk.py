"""
Option A: MemGPT baseline using the official Letta SDK.

Setup:
  1. Install Letta server: pip install letta
  2. Start server: letta server (runs on http://localhost:8283)
  3. Install client: pip install letta-client
  4. Run: python option_a_letta_sdk.py

Pipeline:
  1. For each user, create a Letta agent
  2. Load all conversation turns into archival memory
  3. Send each probing question as a message
  4. Collect agent responses
  5. Save results to eval_results_memgpt_option_a.json
"""

import os
import sys
import json
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "..", "MemoryBank-SiliconFriend", "eval_data", "en", "memory_bank_en.json")
PROBING_PATH = os.path.join(os.path.dirname(__file__), "..", "MemoryBank-SiliconFriend", "eval_data", "en", "probing_questions_en.jsonl")
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "eval_results_memgpt_option_a.json")

LETTA_BASE_URL = "http://localhost:8283"
LLM_MODEL      = "gpt-4o-mini"


def load_data():
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        memory = json.load(f)
    probing = {}
    with open(PROBING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            probing.update(json.loads(line.strip()))
    return memory, probing


def build_archival_docs(user_mem: dict, user: str) -> list[str]:
    """Convert a user's memory into flat text documents for archival storage."""
    docs = []
    history  = user_mem.get("history", {})
    summaries = user_mem.get("summary", {})

    for date, dialogs in history.items():
        for i, dialog in enumerate(dialogs):
            if not isinstance(dialog, dict):
                continue
            q = dialog.get("query", "").strip()
            r = dialog.get("response", "").strip()
            docs.append(f"[{date}] User: {q} | AI: {r}")

        if date in summaries:
            entry = summaries[date]
            content = entry.get("content", entry) if isinstance(entry, dict) else entry
            if content:
                docs.append(f"[{date}] Summary: {content}")

    overall = user_mem.get("overall_history", "")
    if overall:
        docs.append(f"[Overall History] {overall}")

    personality = user_mem.get("overall_personality", "")
    if personality:
        docs.append(f"[Overall Personality] {personality}")

    return docs


def evaluate():
    try:
        from letta_client import Letta
        from letta_client.types import LlmConfig, EmbeddingConfig
    except ImportError:
        print("ERROR: letta-client not installed. Run: pip install letta-client")
        sys.exit(1)

    memory, probing = load_data()

    client = Letta(base_url=LETTA_BASE_URL)
    print(f"Connected to Letta server at {LETTA_BASE_URL}")

    results = {}
    total_q = sum(len(q) for q in probing.values())
    answered = 0

    for user, questions in probing.items():
        print(f"\nUser: {user} ({len(questions)} questions)")

        user_mem = memory.get(user, {})
        if not user_mem.get("history"):
            print(f"  No memory for {user}, skipping.")
            results[user] = [{"question": q, "answer": "No memory available."} for q in questions]
            continue

        # Create a fresh agent for this user
        persona = (
            f"You are a personal AI companion for {user}. "
            "Answer questions using only information from your memory. "
            "Search archival memory when you need to recall specific facts."
        )
        agent = client.agents.create(
            name=f"memgpt_{user.replace(' ', '_')}",
            system=persona,
            llm_config=LlmConfig(model=LLM_MODEL, model_endpoint_type="openai"),
            embedding_config=EmbeddingConfig(
                embedding_model="text-embedding-ada-002",
                embedding_endpoint_type="openai",
            ),
        )
        agent_id = agent.id
        print(f"  Created agent: {agent_id}")

        # Load conversations into archival memory
        docs = build_archival_docs(user_mem, user)
        for doc in docs:
            client.agents.archival_memory.insert(agent_id=agent_id, text=doc)
        print(f"  Loaded {len(docs)} documents into archival memory.")

        # Answer each probing question
        results[user] = []
        for q in questions:
            response = client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "user", "content": q}],
            )
            # Extract the assistant's final text response
            answer = ""
            for msg in response.messages:
                if hasattr(msg, "role") and msg.role == "assistant":
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        answer = msg.content.strip()
                elif hasattr(msg, "message_type") and msg.message_type == "assistant_message":
                    answer = getattr(msg, "content", "").strip()

            results[user].append({"question": q, "answer": answer})
            answered += 1
            print(f"    Q: {q[:70]}...")
            print(f"    A: {answer[:100]}...")

        # Clean up agent
        client.agents.delete(agent_id=agent_id)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Users: {len(probing)}, Total: {total_q}, Answered: {answered}")


if __name__ == "__main__":
    evaluate()
