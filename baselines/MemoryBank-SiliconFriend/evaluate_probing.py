"""
Evaluation script for MemoryBank on probing questions.

Pipeline:
  1. Summarize memory (event summaries + personality) via summarize_memory.py
  2. Apply Ebbinghaus forgetting curve per dialog entry
  3. Build FAISS index from surviving memories
  4. Retrieve relevant memories per question
  5. Ask GPT to answer using retrieved context
  6. Save results to eval_results.json
"""

import os
import sys
import json
import math
import random
import openai
import argparse
import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Add memory_bank to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memory_bank"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memory_bank", "memory_retrieval"))

from memory_bank.summarize_memory import summarize_memory

# ── Config ──────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
GPT_MODEL = "gpt-4o-mini"
TOP_K = 6

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RANDOM_SEED = 42


# ── Ebbinghaus forgetting curve (exact implementation from forget_memory.py) ─

def forgetting_curve(t: float, S: float) -> float:
    return math.exp(-t / (5 * S))


def days_between(date1: str, date2: str) -> int:
    fmt = "%Y-%m-%d"
    d1 = datetime.datetime.strptime(date1, fmt)
    d2 = datetime.datetime.strptime(date2, fmt)
    return (d2 - d1).days


# ── Build FAISS index for a user with forgetting curve applied ───────────────

def build_user_index(memory: dict, user: str, embeddings, ref_date: str) -> FAISS | None:
    user_mem = memory.get(user, {})
    history = user_mem.get("history", {})
    summaries = user_mem.get("summary", {})

    docs = []
    surviving_dates = set()

    for date, dialogs in history.items():
        date_has_survivor = False
        for i, dialog in enumerate(dialogs):
            if not isinstance(dialog, dict):
                continue

            query = dialog.get("query", "")
            response = dialog.get("response", "")
            memory_strength = dialog.get("memory_strength", 1)
            last_recall_date = dialog.get("last_recall_date", date)
            memory_id = dialog.get("memory_id", f"{user}_{date}_{i}")

            t = days_between(last_recall_date, ref_date)
            retention = forgetting_curve(t, memory_strength)

            # Keep memory with probability = retention (drop if random > retention)
            if random.random() > retention:
                continue

            text = (f"Conversation content on {date}:"
                    f" [|User|]: {query.strip()};"
                    f" [|AI|]: {response.strip()}")
            docs.append(Document(
                page_content=text,
                metadata={"source": memory_id, "date": date},
            ))
            date_has_survivor = True

        if date_has_survivor:
            surviving_dates.add(date)

        # Summaries: always included for surviving dates (no forgetting check on summaries)
        if date in surviving_dates and date in summaries:
            summary_entry = summaries[date]
            content = (summary_entry.get("content", summary_entry)
                       if isinstance(summary_entry, dict) else summary_entry)
            if content:
                memory_strength = (summary_entry.get("memory_strength", 1)
                                   if isinstance(summary_entry, dict) else 1)
                last_recall_date = (summary_entry.get("last_recall_date", date)
                                    if isinstance(summary_entry, dict) else date)
                text = f"The summary of the conversation on {date} is: {content}"
                docs.append(Document(
                    page_content=text,
                    metadata={"source": f"{user}_{date}_summary", "date": date},
                ))

    if not docs:
        return None

    return FAISS.from_documents(docs, embeddings)


# ── Answer generation ───────────────────────────────────────────────────────

def answer_with_memory(question: str, user: str, retrieved_memories: list) -> str:
    if not retrieved_memories:
        context = "No relevant memory found."
    else:
        context = "\n\n".join(retrieved_memories)

    messages = [
        {"role": "system", "content": (
            f"You are a personal AI companion for {user}. "
            "Answer the user's question using only the memory context provided. "
            "Be specific and accurate."
        )},
        {"role": "user", "content": f"Memory Context:\n{context}\n\nQuestion: {question}"},
    ]

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate(memory_path: str, probing_path: str, output_path: str,
             language: str = "en", reference_date: str = None):

    random.seed(RANDOM_SEED)

    # Step 1 — Summarize memory if not already done
    print("Step 1: Checking/generating summaries...")
    with open(memory_path, "r", encoding="utf-8") as f:
        memory = json.load(f)

    needs_summary = any(
        not v.get("summary") or not v.get("personality")
        for v in memory.values()
        if v.get("history")
    )
    if needs_summary:
        print("  Summaries missing — running summarize_memory...")
        summarize_memory(memory_path, language=language)
        with open(memory_path, "r", encoding="utf-8") as f:
            memory = json.load(f)
        print("  Summarization complete.")
    else:
        print("  Summaries already present, skipping.")

    # Determine reference date (last conversation date so t is minimal)
    if reference_date:
        ref_date = reference_date
    else:
        all_dates = [
            d for v in memory.values() if v.get("history")
            for d in v["history"].keys()
        ]
        ref_date = max(all_dates) if all_dates else datetime.date.today().strftime("%Y-%m-%d")
    print(f"  Reference date: {ref_date}")

    # Step 2 — Load probing questions
    print("Step 2: Loading probing questions...")
    probing = {}
    with open(probing_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            probing.update(entry)
    total_q = sum(len(q) for q in probing.values())
    print(f"  {total_q} questions across {len(probing)} users.")

    # Step 3 — Build embeddings model once
    print("Step 3: Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )

    # Step 4 — Evaluate per user
    print("Step 4: Running evaluation...")
    results = {}
    answered = 0

    for user, questions in probing.items():
        print(f"\n  User: {user} ({len(questions)} questions)")

        vector_store = build_user_index(memory, user, embeddings, ref_date)
        if vector_store is None:
            print(f"  No memories for {user}, skipping.")
            results[user] = [{"question": q, "answer": "No memory available."} for q in questions]
            continue

        results[user] = []
        for q in questions:
            docs_with_score = vector_store.similarity_search_with_score(q, k=TOP_K)
            retrieved = [doc.page_content for doc, _ in docs_with_score]
            answer = answer_with_memory(q, user, retrieved)
            results[user].append({
                "question": q,
                "retrieved_context": retrieved,
                "answer": answer,
            })
            answered += 1
            print(f"    Q: {q[:70]}...")
            print(f"    A: {answer[:100]}...")

    # Step 5 — Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    print(f"Users: {len(probing)}, Total questions: {total_q}, Answered: {answered}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", default="eval_data/en/memory_bank_en.json")
    parser.add_argument("--probing", default="eval_data/en/probing_questions_en.jsonl")
    parser.add_argument("--output", default="eval_results_en.json")
    parser.add_argument("--language", default="en", choices=["en", "cn"])
    parser.add_argument("--date", default=None, help="Reference date YYYY-MM-DD (default: last memory date)")
    args = parser.parse_args()

    evaluate(
        memory_path=args.memory,
        probing_path=args.probing,
        output_path=args.output,
        language=args.language,
        reference_date=args.date,
    )
