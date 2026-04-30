"""
Evaluate memory_v2, Rsum, and Naive RAG on the synthetic eval dataset.

Dataset: eval_dataset/
  - conversations.json : 21 sessions of Jordan's daily conversations
  - qa_pairs.json      : 65 QA pairs (factual, recall, prediction, transition, pattern_id)
  - schedules.json     : ground truth preference schedules

Methods:
  - naive_rag  : raw conversation chunks in FAISS, semantic search only
  - rsum       : recursive session summary as context
  - memory_v2  : extraction + hierarchical summaries + context_builder

Scoring: token F1 with Porter stemming (same as LoComo paper).
Results broken down by difficulty (simple/mid/difficult) and question type.
"""

import json
import os
import re
import shutil
import string
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from nltk.stem import PorterStemmer
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baselines", "NaiveRAG"))

# ── API key ──────────────────────────────────────────────────────────
for line in open(os.path.join(os.path.dirname(__file__), ".env")):
    if "OPENAI_API_KEY" in line and not line.strip().startswith("#"):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ps     = PorterStemmer()

DATASET_DIR  = os.path.join(os.path.dirname(__file__), "eval_dataset")
RSUM_PROMPTS = os.path.join(os.path.dirname(__file__), "baselines", "Rsum", "prompt.json")

# ── Scoring ───────────────────────────────────────────────────────────

def normalize_date(s: str) -> str:
    """Normalize date formats: '2026-03-08', 'March 8, 2026', 'March 8' → 'march 8 2026'."""
    s = str(s)
    # ISO format: 2026-03-08 → March 8 2026
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        from datetime import datetime
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

# ── LLM call ─────────────────────────────────────────────────────────

QA_SYSTEM = """\
Answer the question based on the provided memory/context.
Be concise — answer in as few words as possible.
If the information is not available, say "No information available."\
"""

def call(prompt, system=QA_SYSTEM, model="gpt-4o-mini", retries=5, max_tokens=200):
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": prompt}],
                temperature=0, max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if i < retries - 1:
                time.sleep(3)
    return ""

def ask(question: str, context: str = "") -> str:
    if context:
        return call(f"Context:\n{context}\n\nQuestion: {question}")
    return call(f"Question: {question}")

# ── Data loading ──────────────────────────────────────────────────────

def load_dataset():
    with open(os.path.join(DATASET_DIR, "conversations.json")) as f:
        convs = json.load(f)
    with open(os.path.join(DATASET_DIR, "qa_pairs.json")) as f:
        qa = json.load(f)
    return convs["user_1"], qa["user_1"]["qa_pairs"]

def sessions_as_list(user_data: dict):
    """Yield (date, conversations_fmt) in chronological order."""
    for date in sorted(user_data["sessions"].keys()):
        session = user_data["sessions"][date]
        turns = [{"speaker": t["speaker"], "text": t["text"]}
                 for t in session["turns"]]
        yield date, [{"time_of_day": "Morning", "turns": turns}]

# ── Rsum memory builder ───────────────────────────────────────────────

def build_rsum_memory(user_data: dict, prompts: dict) -> str:
    instruction = prompts["msc"]["gpt-3.5-turbo"]["update_memory"]
    system = "You are an advanced AI language model with the ability to keep track of dialog information between speakers."
    memory = "Empty"
    for date, convs in sessions_as_list(user_data):
        session_text = " ".join(
            f"{t['speaker']}: {t['text']}"
            for c in convs for t in c["turns"]
        )[:3000]
        prompt = (
            f"**Instruction** {instruction}\n"
            f"**Test** [Previous Memory] {memory} "
            f"[Dialogue Context] {session_text} [Updated Memory]"
        )
        result = call(prompt, system=system, max_tokens=500)
        for prefix in ("Updated memory:", "[Updated Memory]"):
            if prefix in result:
                result = result.split(prefix)[-1]
        memory = result.replace("\n", " ").strip() or memory
    return memory

# ── Naive RAG builder ─────────────────────────────────────────────────

def build_naive_rag_store(user_data: dict):
    from naive_rag import NaiveRAG
    rag = NaiveRAG()
    for date, convs in sessions_as_list(user_data):
        turns = convs[0]["turns"]
        # Mimic (date, turns, session_key, speaker_a) format
        rag.ingest_sessions([(date, [{"speaker": t["speaker"], "text": t["text"]}
                                     for t in turns], date, "Assistant")])
    return rag

# ── Main evaluation ───────────────────────────────────────────────────

def run():
    os.makedirs("results", exist_ok=True)
    print("Loading dataset...")
    user_data, qa_pairs = load_dataset()
    rsum_prompts = json.load(open(RSUM_PROMPTS))

    print(f"  {len(user_data['sessions'])} sessions, {len(qa_pairs)} QA pairs")

    # ── Build all memories once ───────────────────────────────────────
    print("\nBuilding Rsum memory (21 sessions)...")
    rsum_mem = build_rsum_memory(user_data, rsum_prompts)
    print("  Done.")

    print("\nBuilding Naive RAG index...")
    naive_rag = build_naive_rag_store(user_data)
    print("  Done.")

    print("\nBuilding memory_v2 (extraction + summaries)...")
    from memory_v2 import LifelongAgent
    tmp = tempfile.mkdtemp(prefix="synth_eval_")
    try:
        agent = LifelongAgent(base_dir=tmp, chat_model="gpt-4o-mini",
                              extract_model="gpt-4o-mini")
        for date, convs in sessions_as_list(user_data):
            agent.ingest(date, convs)
        agent.build_summaries()
        print("  Done.")

        # ── Evaluate ─────────────────────────────────────────────────
        METHODS = ["naive_rag", "rsum", "memory_v2"]
        all_scores  = {m: [] for m in METHODS}
        diff_scores = {m: defaultdict(list) for m in METHODS}
        type_scores = {m: defaultdict(list) for m in METHODS}
        records     = []

        print(f"\nAnswering {len(qa_pairs)} questions...")
        for qa in tqdm(qa_pairs, desc="QA"):
            q    = qa["question"]
            a    = qa["answer"]
            diff = qa["difficulty"]
            qtype = qa["type"]

            r_naive = naive_rag.search(q, k=5)
            r_naive = ask(q, r_naive)

            r_rsum  = ask(q, rsum_mem[:4000])
            r_mv2   = agent.chat(q)

            s_naive = token_f1(r_naive, a)
            s_rsum  = token_f1(r_rsum,  a)
            s_mv2   = token_f1(r_mv2,   a)

            for method, score, resp in [
                ("naive_rag",  s_naive, r_naive),
                ("rsum",       s_rsum,  r_rsum),
                ("memory_v2",  s_mv2,   r_mv2),
            ]:
                all_scores[method].append(score)
                diff_scores[method][diff].append(score)
                type_scores[method][qtype].append(score)

            records.append({
                "question": q, "answer": a,
                "difficulty": diff, "type": qtype,
                "naive_rag":  {"response": r_naive, "score": s_naive},
                "rsum":       {"response": r_rsum,  "score": s_rsum},
                "memory_v2":  {"response": r_mv2,   "score": s_mv2},
            })

        # ── Print results ─────────────────────────────────────────────
        print_results(all_scores, diff_scores, type_scores)

        with open("results/synthetic_results.json", "w") as f:
            json.dump({"summary": {
                m: {
                    "overall": float(np.mean(all_scores[m])),
                    "by_difficulty": {d: float(np.mean(v)) for d, v in diff_scores[m].items()},
                    "by_type":       {t: float(np.mean(v)) for t, v in type_scores[m].items()},
                    "n": len(all_scores[m]),
                } for m in METHODS
            }, "records": records}, f, indent=2)
        print("\nSaved → results/synthetic_results.json")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def print_results(all_scores, diff_scores, type_scores):
    METHODS = ["naive_rag", "rsum", "memory_v2"]
    diffs = ["simple", "mid", "difficult"]
    types = ["factual", "factual_evolving", "recall", "pattern_id", "prediction", "transition"]

    print(f"\n{'='*75}")
    print("OVERALL")
    print(f"{'Method':<15} {'Overall':>8}  " +
          "  ".join(f"{d:>10}" for d in diffs))
    print("─" * 75)
    for m in METHODS:
        overall = np.mean(all_scores[m])
        diff_vals = "  ".join(
            f"{np.mean(diff_scores[m][d]) if diff_scores[m][d] else 0:>10.3f}"
            for d in diffs
        )
        print(f"{m:<15} {overall:>8.3f}  {diff_vals}")

    print(f"\n{'='*75}")
    print("BY QUESTION TYPE")
    print(f"{'Method':<15} " + "  ".join(f"{t[:10]:>10}" for t in types))
    print("─" * 75)
    for m in METHODS:
        type_vals = "  ".join(
            f"{np.mean(type_scores[m][t]) if type_scores[m][t] else 0:>10.3f}"
            for t in types
        )
        print(f"{m:<15} {type_vals}")
    print("=" * 75)


if __name__ == "__main__":
    run()
