"""
Evaluate memory_v2 on the synthetic dataset and store all artefacts for review.

Saves to results/memory_v2_run/:
  store/               ← full memory_v2 data dir (SQLite + FAISS indices)
  memories.json        ← all extracted SQL memories
  summaries.json       ← all generated summaries (weekly/monthly/yearly/lifetime)
  qa_results.json      ← per-question responses and scores
  report.txt           ← human-readable summary report
"""

import json
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from nltk.stem import PorterStemmer
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

# ── API key ──────────────────────────────────────────────────────────
for line in open(os.path.join(os.path.dirname(__file__), ".env")):
    if "OPENAI_API_KEY" in line and not line.strip().startswith("#"):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ps     = PorterStemmer()

# ── Configuration ─────────────────────────────────────────────────────
# Choose metric: "f1" or "llm_judge"
EVAL_METRIC = "f1" 
if len(sys.argv) > 1 and sys.argv[1] in ["f1", "llm_judge"]:
    EVAL_METRIC = sys.argv[1]

print(f"Using evaluation metric: {EVAL_METRIC}")

DATASET_DIR = os.path.join(os.path.dirname(__file__), "eval_dataset")
RUN_DIR     = os.path.join(os.path.dirname(__file__), "results", "memory_v2_run")
STORE_DIR   = os.path.join(RUN_DIR, "store")

# ── Scoring ───────────────────────────────────────────────────────────

def normalize_date(s: str) -> str:
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(s))
    if m:
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return dt.strftime("%B %-d %Y").lower()
    return str(s)

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

def llm_judge(query: str, pred: str, ref: str) -> float:
    """Uses LLM to decide if the prediction is semantically correct given the reference."""
    prompt = f"""
Query: {query}
Ground Truth: {ref}
System Prediction: {pred}

Is the System Prediction semantically correct and factually consistent with the Ground Truth? 
Ignore minor phrasing differences or verbosity. 
If the System says "I don't know" but the Ground Truth has an answer, it is WRONG.
If the System provides the correct core information, it is RIGHT.

Return only "RIGHT" or "WRONG".
"""
    result = call(prompt, system="You are a strict grading assistant.")
    return 1.0 if "RIGHT" in result.upper() else 0.0

# ── LLM call ─────────────────────────────────────────────────────────

def call(prompt, system="Answer concisely based on context.", retries=5):
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": prompt}],
                temperature=0, max_tokens=200,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if i < retries - 1:
                time.sleep(3)
    return ""

# ── Data loading ──────────────────────────────────────────────────────

def load_dataset():
    with open(os.path.join(DATASET_DIR, "conversations.json")) as f:
        convs = json.load(f)
    with open(os.path.join(DATASET_DIR, "qa_pairs.json")) as f:
        qa = json.load(f)
    return convs["user_1"], qa["user_1"]["qa_pairs"]

def sessions_as_list(user_data: dict):
    for date in sorted(user_data["sessions"].keys()):
        session = user_data["sessions"][date]
        turns   = [{"speaker": t["speaker"], "text": t["text"]}
                   for t in session["turns"]]
        yield date, [{"time_of_day": "", "turns": turns}]

# ── Summary extraction ────────────────────────────────────────────────

def extract_summaries(store) -> dict:
    """Pull all summaries from the FAISS summary store into a dict."""
    summaries = {}
    if store._summary_store is None:
        return summaries
    for doc in store._summary_store.docstore._dict.values():
        ident = doc.metadata.get("identifier", "unknown")
        speaker = doc.metadata.get("speaker", "")
        # Strip the "[identifier] Title\n\n" prefix
        content = doc.page_content
        parts = content.split("\n\n", 1)
        text = parts[1] if len(parts) > 1 else content
        summaries[ident] = {
            "identifier": ident,
            "speaker": speaker,
            "title": doc.metadata.get("title", ""),
            "content": text,
        }
    return dict(sorted(summaries.items()))

# ── Main ─────────────────────────────────────────────────────────────

def run():
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(STORE_DIR, exist_ok=True)

    print("Loading dataset...")
    user_data, qa_pairs = load_dataset()
    print(f"  {len(user_data['sessions'])} sessions, {len(qa_pairs)} QA pairs\n")

    # ── Build memory_v2 into persistent store ─────────────────────────
    from memory_v2 import LifelongAgent
    agent = LifelongAgent(base_dir=STORE_DIR, chat_model="gpt-4o-mini",
                          extract_model="gpt-4o-mini")

    print("Ingesting sessions...")
    for date, convs in sessions_as_list(user_data):
        print(f"  {date}")
        agent.ingest(date, convs)

    print("\nBuilding summaries...")
    agent.build_summaries(force=True)

    # ── Save memories ─────────────────────────────────────────────────
    memories = agent.store.query_memories(limit=5000)
    with open(os.path.join(RUN_DIR, "memories.json"), "w") as f:
        json.dump(memories, f, indent=2)
    print(f"\nSaved {len(memories)} memories → results/memory_v2_run/memories.json")

    # ── Save summaries ────────────────────────────────────────────────
    summaries = extract_summaries(agent.store)
    with open(os.path.join(RUN_DIR, "summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved {len(summaries)} summaries → results/memory_v2_run/summaries.json")

    # ── Print summaries for quick review ─────────────────────────────
    print("\n" + "="*60)
    print("SUMMARIES OVERVIEW")
    print("="*60)
    for ident, s in summaries.items():
        level = ident.split(":")[0]
        spk   = f" [{s['speaker']}]" if s.get("speaker") else ""
        print(f"\n[{ident}]{spk}")
        print(f"  {s['content'][:200]}{'...' if len(s['content']) > 200 else ''}")

    # ── Evaluate QA ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Answering {len(qa_pairs)} questions...")
    print("="*60)

    all_scores  = []
    diff_scores = defaultdict(list)
    type_scores = defaultdict(list)
    records     = []

    for qa in tqdm(qa_pairs, desc="QA"):
        q    = qa["question"]
        a    = qa["answer"]
        diff = qa["difficulty"]
        qtype = qa["type"]

        resp  = agent.chat(q)
        
        if EVAL_METRIC == "llm_judge":
            score = llm_judge(q, resp, a)
        else:
            score = token_f1(resp, a)

        all_scores.append(score)
        diff_scores[diff].append(score)
        type_scores[qtype].append(score)

        records.append({
            "question": q, "answer": a,
            "difficulty": diff, "type": qtype,
            "response": resp, "score": score,
        })

    # ── Print results ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS — memory_v2")
    print(f"{'='*60}")
    print(f"  Overall:  {np.mean(all_scores):.3f}  (n={len(all_scores)})")
    print()
    print("  By difficulty:")
    for diff in ("simple", "mid", "difficult"):
        vals = diff_scores[diff]
        print(f"    {diff:12s}: {np.mean(vals):.3f}  (n={len(vals)})")
    print()
    print("  By type:")
    for qtype in ("factual", "factual_evolving", "recall",
                  "pattern_id", "prediction", "transition"):
        vals = type_scores[qtype]
        if vals:
            print(f"    {qtype:18s}: {np.mean(vals):.3f}  (n={len(vals)})")

    # ── Sample wrong answers ──────────────────────────────────────────
    wrong = [r for r in records if r["score"] < 0.1]
    print(f"\n  Examples with score < 0.1 ({len(wrong)} total):")
    for r in wrong[:5]:
        print(f"    [{r['difficulty']}/{r['type']}] Q: {r['question'][:60]}")
        print(f"      A:    {r['answer']}")
        print(f"      Got:  {r['response'][:80]}")
        print()

    # ── Save all results ──────────────────────────────────────────────
    results = {
        "overall": float(np.mean(all_scores)),
        "by_difficulty": {d: float(np.mean(v)) for d, v in diff_scores.items()},
        "by_type":       {t: float(np.mean(v)) for t, v in type_scores.items()},
        "n": len(all_scores),
    }
    suffix = f"_{EVAL_METRIC}"
    with open(os.path.join(RUN_DIR, f"qa_results{suffix}.json"), "w") as f:
        json.dump({"summary": results, "records": records}, f, indent=2)

    # ── Human-readable report ─────────────────────────────────────────
    report_lines = [
        f"memory_v2 Evaluation Report ({EVAL_METRIC})",
        "=" * 60,
        f"Sessions ingested: {len(user_data['sessions'])}",
        f"Memories extracted: {len(memories)}",
        f"Summaries generated: {len(summaries)}",
        f"QA pairs evaluated: {len(qa_pairs)}",
        f"Metric used: {EVAL_METRIC}",
        "",
        f"Overall Score: {results['overall']:.3f}",
        "",
        "By difficulty:",
    ] + [f"  {d}: {v:.3f}" for d, v in results["by_difficulty"].items()] + [
        "",
        "By type:",
    ] + [f"  {t}: {v:.3f}" for t, v in results["by_type"].items()] + [
        "",
        "Summaries generated:",
    ] + [f"  {ident}: {s['content'][:100]}..." for ident, s in summaries.items()]

    with open(os.path.join(RUN_DIR, f"report{suffix}.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nAll artefacts saved to results/memory_v2_run/")
    print(f"  store/           ← SQLite + FAISS (persistent)")
    print(f"  memories.json    ← {len(memories)} extracted memories")
    print(f"  summaries.json   ← {len(summaries)} summaries")
    print(f"  qa_results{suffix}.json  ← {len(records)} QA records")
    print(f"  report{suffix}.txt       ← human-readable summary")


if __name__ == "__main__":
    run()
