"""
MemGPT evaluation on MSC-Self-Instruct dataset (Deep Memory Retrieval task).

Matches the evaluation setup from the MemGPT paper (Packer et al., 2024), Table 2:
  - Task: Given 4 prior conversation sessions as memory, answer a probing question
    that requires recalling a specific fact from those sessions.
  - Metrics:
      Accuracy  : 1 if ground-truth answer appears in generated response, else 0
      ROUGE-L R : Recall-oriented ROUGE-L between generated and ground-truth answer

Pipeline:
  1. Load previous_dialogs (4 sessions) into archival + recall storage
  2. Run MemGPT agentic loop: LLM calls archival/recall search tools, then send_message
  3. Score against self_instruct['A'] (ground truth)
  4. Save per-example results + aggregate metrics

Run: python evaluate_msc.py [--limit N] [--output PATH]
"""

import os
import sys
import json
import math
import argparse
import openai
from rouge_score import rouge_scorer
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

MSC_PATH    = os.path.join(os.path.dirname(__file__), "msc_data", "msc_self_instruct.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "eval_results_msc_memgpt.json")

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
LLM_MODEL        = "gpt-4-0613"  # matches paper exactly — 8k context window
ARCHIVAL_PAGE_SIZE = 5
RECALL_PAGE_SIZE   = 5
MAX_TOOL_ROUNDS    = 8

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── MemGPT Tool Schemas ───────────────────────────────────────────────────────

MEMGPT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "archival_storage_search",
            "description": (
                "Search archival memory using semantic similarity. "
                "Returns relevant passages from past conversations. "
                "Use this to find specific facts mentioned in prior sessions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic search query."},
                    "page":  {"type": "integer", "description": "Result page (0-indexed).", "default": 0},
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
                "Use this to find conversations from a specific session or about a specific topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword or phrase to search."},
                    "page":  {"type": "integer", "description": "Result page (0-indexed).", "default": 0},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "working_context_replace",
            "description": "Update the working memory block by replacing old_content with new_content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_content": {"type": "string"},
                    "new_content": {"type": "string"},
                },
                "required": ["old_content", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send the final answer to the user. Call this ONLY when ready to answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Final answer to send."},
                },
                "required": ["message"],
            },
        },
    },
]


# ── Memory Storage ────────────────────────────────────────────────────────────

class ArchivalStorage:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None
        self.docs = []

    def insert(self, text: str, metadata: dict = None):
        self.docs.append(Document(page_content=text, metadata=metadata or {}))

    def build_index(self):
        if self.docs:
            self.vector_store = FAISS.from_documents(self.docs, self.embeddings)

    def search(self, query: str, page: int = 0) -> str:
        if not self.vector_store:
            return "Archival storage is empty."
        k = ARCHIVAL_PAGE_SIZE * (page + 1)
        results = self.vector_store.similarity_search(query, k=k)
        page_results = results[page * ARCHIVAL_PAGE_SIZE:(page + 1) * ARCHIVAL_PAGE_SIZE]
        if not page_results:
            return f"No results on page {page}."
        total = math.ceil(len(self.docs) / ARCHIVAL_PAGE_SIZE)
        out = f"Showing {len(page_results)} results (page {page+1}/{total}):\n"
        for i, doc in enumerate(page_results):
            out += f"[{i+1}] {doc.page_content}\n"
        return out.strip()


class RecallStorage:
    def __init__(self):
        self.records = []

    def insert(self, session: int, text: str):
        self.records.append({"session": session, "text": text})

    def search(self, query: str, page: int = 0) -> str:
        q = query.lower()
        matches = [r for r in self.records if q in r["text"].lower()]
        page_matches = matches[page * RECALL_PAGE_SIZE:(page + 1) * RECALL_PAGE_SIZE]
        if not page_matches:
            return f"No recall results for '{query}' on page {page}."
        total = math.ceil(len(matches) / RECALL_PAGE_SIZE)
        out = f"Showing {len(page_matches)} results (page {page+1}/{total}):\n"
        for r in page_matches:
            out += f"[Session {r['session']}] {r['text']}\n"
        return out.strip()


class WorkingContext:
    def __init__(self, content: str = ""):
        self.content = content

    def replace(self, old: str, new: str) -> str:
        if old not in self.content:
            return f"Error: '{old}' not found in working context."
        self.content = self.content.replace(old, new, 1)
        return "Working context updated."

    def __str__(self):
        return self.content or "(empty)"


# ── Build memory from MSC previous_dialogs ────────────────────────────────────

def build_memory(example: dict, embeddings) -> tuple:
    archival = ArchivalStorage(embeddings)
    recall   = RecallStorage()

    for s_idx, session in enumerate(example["previous_dialogs"], start=1):
        session_turns = session.get("dialog", [])
        for turn in session_turns:
            speaker = turn.get("id", f"Speaker {s_idx}")
            text    = turn.get("text", "").strip()
            doc_text = f"[Session {s_idx}] {speaker}: {text}"
            archival.insert(doc_text, metadata={"session": s_idx, "speaker": speaker})
            recall.insert(s_idx, doc_text)

    archival.build_index()

    # Seed working context with persona of Speaker 1 (the agent's persona)
    personas = example.get("init_personas", [])
    persona_str = " ".join(personas[0]) if personas else ""
    working_ctx = WorkingContext(content=f"My persona: {persona_str}" if persona_str else "")

    return archival, recall, working_ctx


# ── MemGPT agentic loop ───────────────────────────────────────────────────────

def run_memgpt(question: str, archival: ArchivalStorage,
               recall: RecallStorage, working_ctx: WorkingContext) -> str:
    system_prompt = f"""You are a conversational agent with access to memory tools.

[Working Context]
{working_ctx}

[Instructions]
- You are Speaker 1 in a multi-session conversation.
- The user (Speaker 2) is asking you to recall something specific from your past conversations.
- Search archival_storage_search and recall_storage_search to find the relevant fact.
- You MUST call send_message with your final answer. Do NOT respond directly.
- Be specific — give the exact fact, name, or detail asked for."""

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
            temperature=0.0,
            max_tokens=512,
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content.strip() if msg.content else ""

        final_answer = None
        for tc in msg.tool_calls:
            fn   = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if fn == "archival_storage_search":
                result = archival.search(args.get("query", ""), args.get("page", 0))
            elif fn == "recall_storage_search":
                result = recall.search(args.get("query", ""), args.get("page", 0))
            elif fn == "working_context_replace":
                result = working_ctx.replace(args.get("old_content", ""), args.get("new_content", ""))
            elif fn == "send_message":
                final_answer = args.get("message", "")
                result = "Message sent."
            else:
                result = f"Unknown function: {fn}"

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if final_answer is not None:
            return final_answer

    return ""


# ── Scoring ───────────────────────────────────────────────────────────────────

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
judge_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm_judge(question: str, prediction: str, ground_truth: str) -> int:
    prompt = (
        f"Question: {question}\n"
        f"Gold answer: {ground_truth}\n"
        f"Agent response: {prediction}\n\n"
        "Is the agent response consistent with the gold answer? Answer only 'yes' or 'no'."
    )
    try:
        resp = judge_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        return 1 if resp.choices[0].message.content.strip().lower().startswith("yes") else 0
    except Exception:
        return 0


def score(question: str, prediction: str, ground_truth: str) -> dict:
    rouge_l = scorer.score(ground_truth, prediction)["rougeL"].recall
    accuracy = llm_judge(question, prediction, ground_truth)
    return {"accuracy": accuracy, "rouge_l_recall": round(rouge_l, 4)}


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(limit: int = None, output_path: str = OUTPUT_PATH):
    with open(MSC_PATH, "r") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    print(f"Evaluating MemGPT on {len(data)} MSC-Self-Instruct examples...")
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )

    results = []
    total_acc   = 0.0
    total_rouge = 0.0

    for i, example in enumerate(data):
        question = example["self_instruct"]["B"]
        gt       = example["self_instruct"]["A"]

        archival, recall, working_ctx = build_memory(example, embeddings)
        prediction = run_memgpt(question, archival, recall, working_ctx)
        metrics    = score(question, prediction, gt)

        total_acc   += metrics["accuracy"]
        total_rouge += metrics["rouge_l_recall"]

        results.append({
            "id":           i,
            "question":     question,
            "ground_truth": gt,
            "prediction":   prediction,
            "accuracy":     metrics["accuracy"],
            "rouge_l_recall": metrics["rouge_l_recall"],
        })

        if (i + 1) % 10 == 0 or i == 0:
            acc   = total_acc / (i + 1)
            rouge = total_rouge / (i + 1)
            print(f"  [{i+1}/{len(data)}] Accuracy: {acc:.3f} | ROUGE-L R: {rouge:.3f}")
        print(f"    Q: {question[:70]}...")
        print(f"    GT: {gt[:60]} | Pred: {prediction[:60]}...")

    n = len(results)
    aggregate = {
        "n":                n,
        "accuracy":         round(total_acc / n, 4),
        "rouge_l_recall":   round(total_rouge / n, 4),
    }

    output = {"aggregate": aggregate, "results": results}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Accuracy:     {aggregate['accuracy']:.4f}")
    print(f"ROUGE-L R:    {aggregate['rouge_l_recall']:.4f}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=None, help="Limit number of examples (default: all 500)")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    evaluate(limit=args.limit, output_path=args.output)
