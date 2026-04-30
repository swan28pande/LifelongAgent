"""
Option B: MemGPT baseline — manual reimplementation of core MemGPT mechanism.

Replicates the four core functions from the Letta/MemGPT codebase:
  - archival_storage_search(query, page)  → FAISS vector search
  - recall_storage_search(query, page)    → keyword search over conversation history
  - working_context_replace(old, new)     → update the mutable working memory block
  - send_message(message)                 → return final answer to user

Architecture (from MemGPT paper, Fig 3):
  Main context  = System Instructions + Working Context + FIFO Queue
  External      = Archival Storage (FAISS) + Recall Storage (chronological DB)
  Function calls move data between main and external context.

Run: python option_b_manual.py
"""

import os
import sys
import json
import math
import openai
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MemoryBank-SiliconFriend", "memory_bank"))

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "..", "MemoryBank-SiliconFriend", "eval_data", "en", "memory_bank_en.json")
PROBING_PATH = os.path.join(os.path.dirname(__file__), "..", "MemoryBank-SiliconFriend", "eval_data", "en", "probing_questions_en.jsonl")
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "eval_results_memgpt_option_b.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
LLM_MODEL = "gpt-4o-mini"
ARCHIVAL_PAGE_SIZE = 5   # results per archival search page
RECALL_PAGE_SIZE   = 5   # results per recall search page
MAX_TOOL_ROUNDS    = 6   # max function-call iterations before forcing answer

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── MemGPT Function Definitions (tool schemas) ───────────────────────────────
# These match the function signatures in the Letta/MemGPT codebase.

MEMGPT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "archival_storage_search",
            "description": (
                "Search archival memory using semantic similarity. "
                "Returns a page of relevant memory passages. "
                "Use this to find specific facts, events, or conversations from the past."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant archival memories.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number of results (0-indexed). Use to get more results.",
                        "default": 0,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_storage_search",
            "description": (
                "Search recall storage (chronological conversation history) by keyword. "
                "Returns a page of matching conversation turns in chronological order. "
                "Use this to find conversations from specific dates or about specific topics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword or phrase to search in conversation history.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number of results (0-indexed).",
                        "default": 0,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "working_context_replace",
            "description": (
                "Edit the working context (persistent memory block). "
                "Replace old_content with new_content to update key facts about the user. "
                "Use this to record important information you want to remember."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_content": {
                        "type": "string",
                        "description": "The text to replace in working context (must match exactly).",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The new text to insert in place of old_content.",
                    },
                },
                "required": ["old_content", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": (
                "Send a message to the user. This is the ONLY way to return a response. "
                "Call this when you have enough information to answer the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The response message to send to the user.",
                    },
                },
                "required": ["message"],
            },
        },
    },
]


# ── Memory Storage Classes ───────────────────────────────────────────────────

class ArchivalStorage:
    """FAISS-backed vector store — analogous to disk storage in MemGPT."""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None
        self.docs = []

    def insert(self, text: str, metadata: dict = None):
        doc = Document(page_content=text, metadata=metadata or {})
        self.docs.append(doc)

    def build_index(self):
        if self.docs:
            self.vector_store = FAISS.from_documents(self.docs, self.embeddings)

    def search(self, query: str, page: int = 0) -> str:
        if not self.vector_store:
            return "Archival storage is empty."
        k = ARCHIVAL_PAGE_SIZE * (page + 1)
        results = self.vector_store.similarity_search(query, k=k)
        page_results = results[page * ARCHIVAL_PAGE_SIZE : (page + 1) * ARCHIVAL_PAGE_SIZE]
        if not page_results:
            return f"No results found on page {page}."
        total_pages = math.ceil(len(self.vector_store.docstore._dict) / ARCHIVAL_PAGE_SIZE)
        output = f"Showing {len(page_results)} results (page {page+1}/{total_pages}):\n"
        for i, doc in enumerate(page_results):
            output += f"[{i+1}] {doc.page_content}\n"
        return output.strip()


class RecallStorage:
    """Chronological conversation DB — analogous to recall storage in MemGPT."""

    def __init__(self):
        self.records = []  # list of {"date": str, "text": str}

    def insert(self, date: str, text: str):
        self.records.append({"date": date, "text": text})

    def search(self, query: str, page: int = 0) -> str:
        query_lower = query.lower()
        matches = [r for r in self.records if query_lower in r["text"].lower()]
        page_matches = matches[page * RECALL_PAGE_SIZE : (page + 1) * RECALL_PAGE_SIZE]
        if not page_matches:
            return f"No recall results found for '{query}' on page {page}."
        total_pages = math.ceil(len(matches) / RECALL_PAGE_SIZE)
        output = f"Showing {len(page_matches)} results (page {page+1}/{total_pages}):\n"
        for r in page_matches:
            output += f"[{r['date']}] {r['text']}\n"
        return output.strip()


class WorkingContext:
    """Fixed-size mutable text block in main context — writable by LLM."""

    def __init__(self, initial: str = ""):
        self.content = initial

    def replace(self, old: str, new: str) -> str:
        if old not in self.content:
            return f"Error: '{old}' not found in working context."
        self.content = self.content.replace(old, new, 1)
        return "Working context updated."

    def __str__(self):
        return self.content if self.content else "(empty)"


# ── Build memory stores for a user ───────────────────────────────────────────

def build_memory_stores(user_mem: dict, user: str, embeddings) -> tuple:
    archival = ArchivalStorage(embeddings)
    recall   = RecallStorage()

    history   = user_mem.get("history", {})
    summaries = user_mem.get("summary", {})

    for date, dialogs in history.items():
        for i, dialog in enumerate(dialogs):
            if not isinstance(dialog, dict):
                continue
            q = dialog.get("query", "").strip()
            r = dialog.get("response", "").strip()
            text = f"[{date}] User: {q} | AI: {r}"
            archival.insert(text, metadata={"date": date, "idx": i})
            recall.insert(date, text)

        if date in summaries:
            entry = summaries[date]
            content = entry.get("content", entry) if isinstance(entry, dict) else entry
            if content:
                text = f"[{date}] Summary: {content}"
                archival.insert(text, metadata={"date": date, "type": "summary"})
                recall.insert(date, text)

    overall = user_mem.get("overall_history", "")
    if overall:
        archival.insert(f"[Overall History] {overall}", metadata={"type": "overall"})

    archival.build_index()

    # Working context seeded with overall personality
    personality = user_mem.get("overall_personality", "")
    working_ctx = WorkingContext(
        initial=f"User: {user}\nPersonality: {personality}" if personality else f"User: {user}"
    )

    return archival, recall, working_ctx


# ── MemGPT agent loop ─────────────────────────────────────────────────────────

def run_memgpt_agent(question: str, user: str,
                     archival: ArchivalStorage,
                     recall: RecallStorage,
                     working_ctx: WorkingContext) -> str:
    """
    Core MemGPT agentic loop:
      LLM receives question → calls tools to search memory → calls send_message to answer.
    """
    system_prompt = f"""You are MemGPT, a personal AI companion for {user} with access to memory tools.

[Working Context]
{working_ctx}

[Instructions]
- Use archival_storage_search to find specific facts or events from long-term memory.
- Use recall_storage_search to find conversations by keyword or date.
- Use working_context_replace to update key facts you want to remember.
- You MUST call send_message to return your final answer. Do NOT respond directly.
- Search memory before answering. Be specific and accurate."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=MEMGPT_TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=500,
        )

        msg = response.choices[0].message
        messages.append(msg)

        # No tool call → model responded directly (shouldn't happen with instructions)
        if not msg.tool_calls:
            return msg.content.strip() if msg.content else "No answer."

        # Execute each tool call
        final_answer = None
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if fn_name == "archival_storage_search":
                result = archival.search(args.get("query", ""), args.get("page", 0))
            elif fn_name == "recall_storage_search":
                result = recall.search(args.get("query", ""), args.get("page", 0))
            elif fn_name == "working_context_replace":
                result = working_ctx.replace(args.get("old_content", ""), args.get("new_content", ""))
            elif fn_name == "send_message":
                final_answer = args.get("message", "")
                result = "Message sent."
            else:
                result = f"Unknown function: {fn_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        if final_answer is not None:
            return final_answer

    return "Max iterations reached without answer."


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate():
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        memory = json.load(f)

    probing = {}
    with open(PROBING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            probing.update(json.loads(line.strip()))

    total_q = sum(len(q) for q in probing.values())
    print(f"Loaded {total_q} questions across {len(probing)} users.")

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )

    results = {}
    answered = 0

    for user, questions in probing.items():
        print(f"\nUser: {user} ({len(questions)} questions)")

        user_mem = memory.get(user, {})
        if not user_mem.get("history"):
            print(f"  No memory for {user}, skipping.")
            results[user] = [{"question": q, "answer": "No memory available."} for q in questions]
            continue

        archival, recall, working_ctx = build_memory_stores(user_mem, user, embeddings)
        print(f"  Archival: {len(archival.docs)} docs | Recall: {len(recall.records)} records")

        results[user] = []
        for q in questions:
            answer = run_memgpt_agent(q, user, archival, recall, working_ctx)
            results[user].append({
                "question": q,
                "answer": answer,
            })
            answered += 1
            print(f"    Q: {q[:70]}...")
            print(f"    A: {answer[:100]}...")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Users: {len(probing)}, Total: {total_q}, Answered: {answered}")


if __name__ == "__main__":
    evaluate()
