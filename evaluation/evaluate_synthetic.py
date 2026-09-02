"""
Evaluate memory_v2, naive_rag, rsum, memory_bank, and memgpt on the synthetic eval dataset.
Runs both gemini-3.1-flash-lite and gemini-3.1-pro-preview over exactly 30 days of data.
Outputs are saved to the complete_comparison/ directory.
Evaluation metric: F1 score + LLM Judge.
"""

import os
import re
import sys
import json
import string
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from nltk.stem import PorterStemmer
from tqdm import tqdm
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Insert paths to allow importing memory_v2 and baselines
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "baselines", "NaiveRAG"))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler

# Pricing (USD per 1M tokens)
PRICING = {
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00}
}

# Initialize embedding model once (used for NaiveRAG, MemoryBank, and MemGPT stores)
print("Loading HuggingFaceEmbeddings nomic-embed-text-v1...")
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True},
)

# Set directories
DATASET_DIR  = os.path.join(PROJECT_ROOT, "datasets", "eval")
RSUM_PROMPTS = os.path.join(PROJECT_ROOT, "baselines", "Rsum", "prompt.json")
OUTPUT_DIR   = "complete_comparison"

ps = PorterStemmer()

QA_SYSTEM = """\
Answer the question based on the provided memory/context.
Be concise — answer in as few words as possible.
If the information is not available, say "No information available."\
"""

# ── Callback for tracking tokens across all LLM invocations ──────────

class TokenCounterCallback(BaseCallbackHandler):
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        for generation in response.generations:
            for g in generation:
                if hasattr(g, 'generation_info') and g.generation_info:
                    usage = g.generation_info.get('usage', {})
                    self.prompt_tokens += usage.get('prompt_tokens', 0)
                    self.completion_tokens += usage.get('completion_tokens', 0)
                    self.total_tokens += usage.get('total_tokens', 0)
                if hasattr(g.message, 'usage_metadata') and g.message.usage_metadata:
                    um = g.message.usage_metadata
                    self.prompt_tokens += um.get('input_tokens', 0)
                    self.completion_tokens += um.get('output_tokens', 0)
                    self.total_tokens += um.get('total_tokens', 0)

    def get_report(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

# ── F1 and LLM Judge Scoring ─────────────────────────────────────────

def normalize_date(s: str) -> str:
    s = str(s)
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return dt.strftime("%B %-d %Y").lower()
    return s

def normalize(s: str) -> str:
    s = normalize_date(str(s))
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s, flags=re.IGNORECASE)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.lower().split())

def token_f1(pred: str, ref: str) -> float:
    p = [ps.stem(w) for w in normalize(pred).split()]
    r = [ps.stem(w) for w in normalize(ref).split()]
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(p)
    rec  = n / len(r)
    return 2 * prec * rec / (prec + rec)

def llm_judge(query: str, pred: str, ref: str, call_gemini_func) -> float:
    prompt_text = f"""
Query: {query}
Ground Truth: {ref}
System Prediction: {pred}

Is the System Prediction semantically correct and factually consistent with the Ground Truth? 
Ignore minor phrasing differences or verbosity. 

EVALUATION GUIDELINES:
- Temporal Evolution: Facts can evolve over time. If the Ground Truth describes a state as current/upcoming but the System Prediction describes it as past/completed (or vice-versa), treat it as consistent if they refer to the same event.
- State Changes: If the Ground Truth describes a historical preference or state that subsequently changed, the prediction is correct if it lists either the relevant state or mentions the transition.
- List/Group Overlap: If the Ground Truth is a single item and the System Prediction provides a category, list, or set of options that clearly contains the Ground Truth item, it is correct.
- Extra Details: Do not penalize the System Prediction for including extra reasoning, context, or details, as long as the correct core information is present.

If the System says "I don't know" but the Ground Truth has a correct answer, it is WRONG.
If the System provides the correct core information (even if embedded in a larger correct context or list), it is RIGHT.

Return only "RIGHT" or "WRONG".
"""
    res = call_gemini_func(prompt_text, system="You are a strict grading assistant.")
    return 1.0 if "RIGHT" in res.upper() else 0.0

# ── Data Loading ──────────────────────────────────────────────────────

def load_dataset():
    with open(os.path.join(DATASET_DIR, "conversations.json")) as f:
        convs = json.load(f)
    with open(os.path.join(DATASET_DIR, "qa_pairs.json")) as f:
        qa = json.load(f)
    return convs["user_1"], qa["user_1"]["qa_pairs"]

# ── Rsum memory builder ───────────────────────────────────────────────

def build_rsum_memory(user_data: dict, prompts: dict, dates: list, call_gemini_func) -> str:
    instruction = prompts["msc"]["gpt-3.5-turbo"]["update_memory"]
    system = "You are an advanced AI language model with the ability to keep track of dialog information between speakers."
    memory = "Empty"
    for date in dates:
        session = user_data["sessions"][date]
        session_text = " ".join(
            f"{t['speaker']}: {t['text']}"
            for t in session["turns"]
        )[:3000]
        prompt = (
            f"**Instruction** {instruction}\n"
            f"**Test** [Previous Memory] {memory} "
            f"[Dialogue Context] {session_text} [Updated Memory]"
        )
        result = call_gemini_func(prompt, system=system)
        for prefix in ("Updated memory:", "[Updated Memory]"):
            if prefix in result:
                result = result.split(prefix)[-1]
        memory = result.replace("\n", " ").strip() or memory
    return memory

# ── Memory Bank baseline ──────────────────────────────────────────────

class MemoryBankBaseline:
    def __init__(self, embeddings, call_gemini_func):
        self.embeddings = embeddings
        self.call_gemini = call_gemini_func
        self.summaries = {}  # date -> daily summary
        self.store = None

    def ingest(self, date: str, turns: list):
        turns_text = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
        prompt = (
            f"Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. "
            f"Dialogue content:\n{turns_text}\n\nSummarization:"
        )
        summary = self.call_gemini(prompt, system="You are a helpful assistant.")
        self.summaries[date] = summary
        
        doc = Document(page_content=f"Date: {date}\nSummary: {summary}", metadata={"date": date})
        if self.store is None:
            self.store = FAISS.from_documents([doc], self.embeddings)
        else:
            self.store.add_documents([doc])

    def search(self, query: str, k: int = 3) -> str:
        if self.store is None:
            return ""
        hits = self.store.similarity_search(query, k=k)
        return "\n\n---\n\n".join(h.page_content for h in hits)

# ── MemGPT baseline ───────────────────────────────────────────────────

class MemGPTBaseline:
    def __init__(self, embeddings, llm_model):
        self.embeddings = embeddings
        self.llm = llm_model
        self.archival_docs = []
        self.archival_store = None
        self.recall_records = []
        self.working_context = "User: Jordan\nFacts about User: None"

    def ingest_session(self, date: str, turns: list):
        session_lines = [f"Date: {date}"]
        for t in turns:
            session_lines.append(f"{t['speaker']}: {t['text']}")
        session_text = "\n".join(session_lines)
        
        # Recall storage (chronological turns)
        self.recall_records.append(session_text)
        
        # Archival storage (FAISS vector store, chunked in sizes of 5)
        chunk_size = 5
        for start in range(0, max(len(turns), 1), chunk_size):
            batch = turns[start: start + chunk_size]
            if not batch:
                continue
            lines = [f"Date: {date}"]
            for t in batch:
                lines.append(f"{t['speaker']}: {t['text']}")
            doc = Document(page_content="\n".join(lines), metadata={"date": date})
            self.archival_docs.append(doc)

    def build_index(self):
        if self.archival_docs:
            self.archival_store = FAISS.from_documents(self.archival_docs, self.embeddings)

    def run_agent(self, query: str, max_rounds: int = 5) -> str:
        @tool
        def archival_storage_search(search_query: str) -> str:
            """Search archival memory using semantic similarity. Returns relevant passages."""
            if self.archival_store is None:
                return "Archival storage is empty."
            hits = self.archival_store.similarity_search(search_query, k=3)
            return "\n\n---\n\n".join(h.page_content for h in hits)

        @tool
        def recall_storage_search(keyword: str) -> str:
            """Search recall storage (chronological conversation history) by keyword."""
            matches = [r for r in self.recall_records if keyword.lower() in r.lower()]
            if not matches:
                return f"No recall results found for '{keyword}'."
            return "\n\n---\n\n".join(matches[-3:])

        @tool
        def working_context_replace(old_content: str, new_content: str) -> str:
            """Edit the working context (persistent memory block) by replacing old_content with new_content."""
            if old_content not in self.working_context:
                return f"Error: '{old_content}' not found in working context."
            self.working_context = self.working_context.replace(old_content, new_content, 1)
            return "Working context updated."

        @tool
        def send_message(message: str) -> str:
            """Send a response message to the user. This is the ONLY way to answer the user query."""
            return message

        tools = [archival_storage_search, recall_storage_search, working_context_replace, send_message]
        llm_with_tools = self.llm.bind_tools(tools)

        system_prompt = f"""You are MemGPT, a personal AI companion for Jordan with access to memory tools.

[Working Context]
{self.working_context}

[Instructions]
- Use archival_storage_search to find specific facts or events from long-term memory.
- Use recall_storage_search to find conversations by keyword or date.
- Use working_context_replace to update key facts you want to remember.
- You MUST call send_message to return your final answer. Do NOT respond directly.
- Search memory before answering. Be specific and accurate."""

        messages = [
            ("system", system_prompt),
            ("human", query)
        ]

        final_answer = None
        for _ in range(max_rounds):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                print(f"Error in MemGPT agent invoke: {e}")
                return "Error in agent execution."

            messages.append(response)

            if response.tool_calls:
                for tc in response.tool_calls:
                    fn_name = tc["name"]
                    args = tc["args"]
                    
                    if fn_name == "archival_storage_search":
                        tool_out = archival_storage_search.invoke(args)
                    elif fn_name == "recall_storage_search":
                        tool_out = recall_storage_search.invoke(args)
                    elif fn_name == "working_context_replace":
                        tool_out = working_context_replace.invoke(args)
                    elif fn_name == "send_message":
                        final_answer = args.get("message", "") or args.get("send_message", "") or str(args)
                        tool_out = "Message sent."
                    else:
                        tool_out = f"Unknown function: {fn_name}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": fn_name,
                        "content": str(tool_out)
                    })
                if final_answer is not None:
                    return final_answer
            else:
                if response.content:
                    if isinstance(response.content, list):
                        return " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in response.content]).strip()
                    return str(response.content).strip()

        return "Max iterations reached without final answer."

# ── Evaluator Run for a Single Model ───────────────────────────────────

def run_model_evaluation(model_name: str, user_data: dict, qa_pairs: list, rsum_prompts: dict, dates: list):
    print(f"\n=======================================================")
    print(f"STARTING EVALUATION FOR MODEL: {model_name.upper()}")
    print(f"=======================================================")
    
    tracker = TokenCounterCallback()
    model_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, callbacks=[tracker])
    
    def call_gemini(prompt: str, system: str = "You are a helpful assistant.") -> str:
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", prompt)
            ])
            chain = chat_prompt | model_llm
            response = chain.invoke({})
            if isinstance(response.content, list):
                return " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in response.content]).strip()
            return str(response.content).strip()
        except Exception as e:
            print(f"Error in Gemini call: {e}")
            return ""

    def ask_gemini(question: str, context: str = "") -> str:
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            prompt = f"Question: {question}"
        return call_gemini(prompt, system=QA_SYSTEM)

    # 1. Build naive_rag
    print("\n--- BUILDING NAIVE RAG STORE ---")
    from naive_rag import NaiveRAG
    naive_rag = NaiveRAG()
    naive_sessions = []
    for date in dates:
        session = user_data["sessions"][date]
        turns = [{"speaker": t["speaker"], "text": t["text"]} for t in session["turns"]]
        naive_sessions.append((date, turns, date, "Assistant"))
    naive_rag.ingest_sessions(naive_sessions)

    # 2. Build rsum memory
    print("\n--- BUILDING RSUM SUMMARY ---")
    rsum_mem = build_rsum_memory(user_data, rsum_prompts, dates, call_gemini)

    # 3. Build memory_bank
    print("\n--- BUILDING MEMORY BANK BASELINE ---")
    memory_bank = MemoryBankBaseline(embeddings, call_gemini)
    for date in dates:
        session = user_data["sessions"][date]
        turns = [{"speaker": t["speaker"], "text": t["text"]} for t in session["turns"]]
        memory_bank.ingest(date, turns)

    # 4. Build memgpt
    print("\n--- BUILDING MEMGPT BASELINE ---")
    memgpt = MemGPTBaseline(embeddings, model_llm)
    for date in dates:
        session = user_data["sessions"][date]
        turns = [{"speaker": t["speaker"], "text": t["text"]} for t in session["turns"]]
        memgpt.ingest_session(date, turns)
    memgpt.build_index()

    # 5. Build memory_v2
    print("\n--- BUILDING MEMORY V2 STORE ---")
    from memory_v2 import LifelongAgent
    tmp = tempfile.mkdtemp(prefix=f"synth_eval_{model_name}_")
    try:
        agent = LifelongAgent(base_dir=tmp, chat_model=model_name,
                              extract_model=model_name, callbacks=[tracker])
        for date in dates:
            session = user_data["sessions"][date]
            turns = [{"speaker": t["speaker"], "text": t["text"]} for t in session["turns"]]
            agent.ingest(date, [{"time_of_day": "All Day", "turns": turns}])

        print("Running memory_v2 consolidation and summarization...")
        agent.build_summaries(force=True)

        # ── Export Summaries ──
        print("\n--- EXPORTING SUMMARIES ---")
        suffix = "pro" if "pro" in model_name else "flash"
        
        # memory_v2 summaries
        sorted_docs = sorted(
            agent.store._summary_store.docstore._dict.values(),
            key=lambda doc: doc.metadata.get("identifier", "")
        )
        memory_v2_summaries = ""
        for doc in sorted_docs:
            memory_v2_summaries += "=" * 80 + "\n" + doc.page_content.strip() + "\n" + "=" * 80 + "\n\n"
        with open(os.path.join(OUTPUT_DIR, f"memory_v2_summaries_{suffix}.txt"), "w") as f:
            f.write(memory_v2_summaries)

        # rsum summary
        with open(os.path.join(OUTPUT_DIR, f"rsum_summary_{suffix}.txt"), "w") as f:
            f.write(rsum_mem)

        # memory_bank summaries
        with open(os.path.join(OUTPUT_DIR, f"memory_bank_summaries_{suffix}.json"), "w") as f:
            json.dump(memory_bank.summaries, f, indent=2)

        # memgpt final working context
        with open(os.path.join(OUTPUT_DIR, f"memgpt_working_context_{suffix}.txt"), "w") as f:
            f.write(memgpt.working_context)

        # ── Run Q&A Evaluation ──
        print(f"\n--- EVALUATING QA PAIRS ACROSS 5 METHODS ---")
        METHODS = ["naive_rag", "rsum", "memory_bank", "memgpt", "memory_v2"]
        
        all_f1_scores = defaultdict(list)
        all_llm_scores = defaultdict(list)
        
        diff_f1_scores = defaultdict(lambda: defaultdict(list))
        diff_llm_scores = defaultdict(lambda: defaultdict(list))
        
        type_f1_scores = defaultdict(lambda: defaultdict(list))
        type_llm_scores = defaultdict(lambda: defaultdict(list))
        
        detailed_records = []

        for qa in tqdm(qa_pairs, desc=f"QA Evaluator ({model_name})"):
            q, a = qa["question"], qa["answer"]
            diff, qtype = qa["difficulty"], qa["type"]
            
            # Predict
            # 1. naive_rag
            context_rag = naive_rag.search(q, k=5)
            resp_rag = ask_gemini(q, context_rag)
            
            # 2. rsum
            resp_rsum = ask_gemini(q, rsum_mem)
            
            # 3. memory_bank
            context_mb = memory_bank.search(q, k=3)
            resp_mb = ask_gemini(q, context_mb)
            
            # 4. memgpt
            resp_memgpt = memgpt.run_agent(q)
            
            # 5. memory_v2
            resp_v2 = agent.chat(q)
            
            responses = {
                "naive_rag": resp_rag,
                "rsum": resp_rsum,
                "memory_bank": resp_mb,
                "memgpt": resp_memgpt,
                "memory_v2": resp_v2
            }
            
            record = {
                "question": q,
                "answer": a,
                "difficulty": diff,
                "type": qtype,
                "predictions": {}
            }
            
            for m in METHODS:
                resp = responses[m]
                f1_score = token_f1(resp, a)
                llm_score = llm_judge(q, resp, a, call_gemini)
                
                all_f1_scores[m].append(f1_score)
                all_llm_scores[m].append(llm_score)
                
                diff_f1_scores[m][diff].append(f1_score)
                diff_llm_scores[m][diff].append(llm_score)
                
                type_f1_scores[m][qtype].append(f1_score)
                type_llm_scores[m][qtype].append(llm_score)
                
                record["predictions"][m] = {
                    "response": resp,
                    "f1_score": f1_score,
                    "llm_score": llm_score
                }
            
            detailed_records.append(record)

        # Build output stats
        summary_stats = {}
        for m in METHODS:
            summary_stats[m] = {
                "overall_f1": float(np.mean(all_f1_scores[m])),
                "overall_llm": float(np.mean(all_llm_scores[m])),
                "by_difficulty": {
                    d: {
                        "f1": float(np.mean(diff_f1_scores[m][d])) if diff_f1_scores[m][d] else 0.0,
                        "llm": float(np.mean(diff_llm_scores[m][d])) if diff_llm_scores[m][d] else 0.0
                    }
                    for d in ["simple", "mid", "difficult"]
                },
                "by_type": {
                    t: {
                        "f1": float(np.mean(type_f1_scores[m][t])) if type_f1_scores[m][t] else 0.0,
                        "llm": float(np.mean(type_llm_scores[m][t])) if type_llm_scores[m][t] else 0.0
                    }
                    for t in ["factual", "factual_evolving", "recall", "pattern_id", "prediction", "transition"]
                }
            }

        # Calculate costs
        report = tracker.get_report()
        input_tokens = report["prompt_tokens"]
        output_tokens = report["completion_tokens"]
        total_tokens = report["total_tokens"]
        
        pricing = PRICING[model_name]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        cost_stats = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost
        }

        # Save model-specific evaluation file
        with open(os.path.join(OUTPUT_DIR, f"comparative_evaluation_{suffix}.json"), "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "summary": summary_stats,
                "cost": cost_stats,
                "details": detailed_records
            }, f, indent=2)

        return summary_stats, cost_stats

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ── Main Script Entry point ───────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="Number of days to ingest (default: 30)")
    parser.add_argument("--num_questions", type=int, default=None, help="Number of questions to evaluate (default: all)")
    parser.add_argument("--models", nargs="+", default=["gemini-3.1-flash-lite", "gemini-3.1-pro-preview"],
                        help="Models to evaluate")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    user_data, qa_pairs = load_dataset()
    rsum_prompts = json.load(open(RSUM_PROMPTS))

    # Take first args.days days
    dates = sorted(user_data["sessions"].keys())[:args.days]

    # Take first args.num_questions questions
    if args.num_questions is not None:
        eval_qa_subset = qa_pairs[:args.num_questions]
    else:
        eval_qa_subset = qa_pairs

    print(f"Running comparative evaluation on {len(dates)} days and {len(eval_qa_subset)} questions.")

    all_results = {}
    all_costs = {}

    for model in args.models:
        stats, cost = run_model_evaluation(model, user_data, eval_qa_subset, rsum_prompts, dates)
        all_results[model] = stats
        all_costs[model] = cost

    # Save a combined overview
    with open(os.path.join(OUTPUT_DIR, "combined_evaluation_overview.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "days": args.days,
            "num_questions": len(eval_qa_subset),
            "results": all_results,
            "costs": all_costs
        }, f, indent=2)

    # Output both tables
    print_tables(all_results, all_costs)

def print_tables(all_results, all_costs):
    diffs = ["simple", "mid", "difficult"]
    
    print("\n" + "=" * 90)
    print("TABLE 1: PERFORMANCE COMPARISON (F1 / LLM Judge)")
    print("=" * 90)
    print(f"{'Model':<22} | {'Method':<12} | {'Overall F1':<10} | {'Overall LLM':<11} | " + " | ".join(f"{d.capitalize() + ' LLM':<13}" for d in diffs))
    print("-" * 90)
    for model in all_results.keys():
        stats = all_results[model]
        for m in stats.keys():
            overall_f1 = stats[m]["overall_f1"]
            overall_llm = stats[m]["overall_llm"]
            diff_vals = " | ".join(f"{stats[m]['by_difficulty'][d]['llm']:<13.3f}" for d in diffs)
            model_disp = model if m == list(stats.keys())[0] else ""
            print(f"{model_disp:<22} | {m:<12} | {overall_f1:<10.3f} | {overall_llm:<11.3f} | {diff_vals}")
        print("-" * 90)

    print("\n" + "=" * 80)
    print("TABLE 2: TOTAL COST & TOKEN USAGE COMPARISON")
    print("=" * 80)
    print(f"{'Model':<25} | {'Prompt Tokens':<15} | {'Completion Tokens':<18} | {'Total Cost (USD)':<16}")
    print("-" * 80)
    for model in all_costs.keys():
        c = all_costs[model]
        print(f"{model:<25} | {c['prompt_tokens']:<15,} | {c['completion_tokens']:<18,} | ${c['total_cost']:<15.6f}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
