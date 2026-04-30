"""
Evaluate memory_v2 on LoComo and compare against baselines.

Dataset: baselines/locomo/data/locomo10.json
  - 10 conversations, up to 35 sessions each, real timestamps
  - 1,986 pre-annotated QA pairs (single-hop, multi-hop, temporal, commonsense, adversarial)

Methods:
  - no_memory : question asked with no conversation context
  - rsum      : recursive session summary (M_i = LLM(S_i, M_{i-1})) as context
  - memory_v2 : full extraction + hierarchical summaries + context_builder

Scoring (paper-exact from task_eval/evaluation.py):
  - cat 1 (multi-hop)    : split-F1 over comma-separated sub-answers
  - cat 2,3,4 (temporal, commonsense, single-hop) : token F1 with Porter stemming
  - cat 5 (adversarial)  : 1 if model says "no information"/"not mentioned", else 0

Paper reported baselines (GPT-3.5-turbo, full context, Table 1):
  Overall F1 ~0.27, single-hop ~0.35, temporal ~0.22, multi-hop ~0.19
"""

import argparse
import json
import os
import re
import shutil
import string
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime

import numpy as np
from nltk.stem import PorterStemmer
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baselines", "locomo"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baselines", "NaiveRAG"))

# ── API key ──────────────────────────────────────────────────────────
for line in open(os.path.join(os.path.dirname(__file__), ".env")):
    if "OPENAI_API_KEY" in line and not line.strip().startswith("#"):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ps = PorterStemmer()

LOCOMO_PATH = "baselines/locomo/data/locomo10.json"
RSUM_PROMPTS = "baselines/Rsum/prompt.json"

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "commonsense", 4: "single-hop", 5: "adversarial"}

# ── Paper-exact scoring ───────────────────────────────────────────────

def normalize(s):
    s = str(s).replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s, flags=re.IGNORECASE)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.lower().split())

def token_f1(pred, ref):
    p_toks = [ps.stem(w) for w in normalize(pred).split()]
    r_toks = [ps.stem(w) for w in normalize(ref).split()]
    common = Counter(p_toks) & Counter(r_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(p_toks)
    rec  = n / len(r_toks)
    return 2 * prec * rec / (prec + rec)

def score_answer(pred: str, answer, category: int) -> float:
    answer = str(answer)
    pred   = str(pred)
    if category in (2, 3, 4):                        # temporal, commonsense, single-hop
        return token_f1(pred, answer)
    elif category == 1:                               # multi-hop: split on comma
        preds = [p.strip() for p in pred.split(",")]
        refs  = [r.strip() for r in answer.split(",")]
        return float(np.mean([
            max(token_f1(p, r) for p in preds) for r in refs
        ]))
    elif category == 5:                               # adversarial: model should abstain
        abstain_phrases = ("no information", "not mentioned", "not provided",
                           "don't know", "do not know", "cannot find", "no record")
        return 1.0 if any(ph in pred.lower() for ph in abstain_phrases) else 0.0
    return 0.0

# ── LLM call helper ──────────────────────────────────────────────────

def call(prompt, system="You are a helpful assistant.", model="gpt-4o-mini",
         retries=5, max_tokens=200):
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

# ── Data helpers ─────────────────────────────────────────────────────

def parse_session_date(date_str: str) -> str:
    """Parse '1:56 pm on 8 May, 2023' → '2023-05-08'."""
    try:
        # Strip time part, keep date
        date_part = date_str.split(" on ")[-1].strip().rstrip(".")
        dt = datetime.strptime(date_part, "%d %B, %Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        try:
            date_part = date_str.split(" on ")[-1].strip().rstrip(".")
            dt = datetime.strptime(date_part, "%d %B %Y")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return "2020-01-01"

def get_sessions(conv: dict):
    """Yield (date_str, turns_list, session_key) for each session in chronological order."""
    c = conv["conversation"]
    speaker_a = c["speaker_a"]

    # Find all session_N keys
    session_keys = sorted(
        [k for k in c if re.match(r"^session_\d+$", k)],
        key=lambda k: int(k.split("_")[1])
    )
    for sk in session_keys:
        dt_key = sk + "_date_time"
        date   = parse_session_date(c.get(dt_key, "1 January 2020"))
        turns  = c[sk]
        yield date, turns, sk, speaker_a

def turns_to_conversations(turns: list, speaker_a: str):
    """Convert LoComo turns to memory_v2 conversation format."""
    conv_turns = []
    for t in turns:
        if "text" not in t:
            continue
        role = "User" if t["speaker"] == speaker_a else "System"
        text = t["text"]
        # Append image caption if present (skip actual image URL)
        if "blip_caption" in t:
            text += f" [shared image: {t['blip_caption']}]"
        conv_turns.append({"speaker": role, "text": text})
    return [{"time_of_day": "Morning", "turns": conv_turns}]

def build_full_context(conv: dict) -> str:
    """Concatenate all session turns as plain text (for no_memory baseline context)."""
    lines = []
    for date, turns, sk, speaker_a in get_sessions(conv):
        lines.append(f"\n[{date}]")
        for t in turns:
            if "text" in t:
                lines.append(f"{t['speaker']}: {t['text']}")
    return "\n".join(lines)

# ── Rsum memory builder ───────────────────────────────────────────────

def build_rsum_memory(conv: dict, prompts: dict) -> str:
    instruction = prompts["msc"]["gpt-3.5-turbo"]["update_memory"]
    system = "You are an advanced AI language model with the ability to keep track of dialog information between speakers."
    memory = "Empty"
    for date, turns, sk, speaker_a in get_sessions(conv):
        session_text = " ".join(
            f"{'User' if t['speaker']==speaker_a else 'System'}: {t.get('text','')}"
            for t in turns if "text" in t
        )
        prompt = (
            f"**Instruction** {instruction}\n"
            f"**Test** [Previous Memory] {memory} "
            f"[Dialogue Context] {session_text[:3000]} [Updated Memory]"
        )
        result = call(prompt, system=system, max_tokens=500)
        for prefix in ("Updated memory:", "[Updated Memory]"):
            if prefix in result:
                result = result.split(prefix)[-1]
        memory = result.replace("\n", " ").strip() or memory
    return memory

# ── Answer generation ─────────────────────────────────────────────────

QA_SYSTEM = """\
Answer the question based on the provided memory/context.
Be concise — answer in as few words as possible.
If the information is not available, say "No information available".\
"""

def ask(question: str, context: str = "") -> str:
    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        prompt = f"Question: {question}"
    return call(prompt, system=QA_SYSTEM)


# ── Naive RAG (imported from baselines/NaiveRAG) ─────────────────────────────

from naive_rag import NaiveRAG

def build_naive_rag(conv: dict) -> NaiveRAG:
    rag = NaiveRAG()
    rag.ingest_sessions(list(get_sessions(conv)))
    return rag

def ask_naive_rag(question: str, rag: NaiveRAG, k: int = 5) -> str:
    context = rag.search(question, k=k)
    if not context:
        return "No information available."
    return ask(question, context)

# ── Main evaluation ───────────────────────────────────────────────────

def run(args):
    os.makedirs("results", exist_ok=True)

    with open(LOCOMO_PATH) as f:
        all_convs = json.load(f)

    rsum_prompts = json.load(open(RSUM_PROMPTS))

    convs = all_convs[:args.max_convs]
    print(f"Evaluating {len(convs)} conversations")

    # Category filter
    eval_cats = set(args.categories) if args.categories else {1, 2, 3, 4, 5}

    METHODS = ["no_memory", "naive_rag", "rsum", "memory_v2"]
    all_scores = {m: [] for m in METHODS}
    cat_scores  = {m: {c: [] for c in eval_cats} for m in METHODS}
    records     = []

    for conv_idx, conv in enumerate(tqdm(convs, desc="Conversations")):
        qa_pairs = [q for q in conv["qa"] if q["category"] in eval_cats]
        if not qa_pairs:
            continue

        print(f"\n  Conv {conv_idx} ({conv['sample_id']}): "
              f"{len(get_sessions_list(conv))} sessions, {len(qa_pairs)} QA pairs")

        # ── Build naive RAG (baselines/NaiveRAG) ─────────────────────
        print("    Building Naive RAG...")
        naive_rag = build_naive_rag(conv)

        # ── Build rsum memory ─────────────────────────────────────────
        print("    Building Rsum memory...")
        rsum_mem = build_rsum_memory(conv, rsum_prompts)

        # ── Build memory_v2 ───────────────────────────────────────────
        from memory_v2 import LifelongAgent
        tmp = tempfile.mkdtemp(prefix="locomo_eval_")
        try:
            print("    Building memory_v2...")
            agent = LifelongAgent(base_dir=tmp, chat_model="gpt-4o-mini",
                                  extract_model="gpt-4o-mini")
            for date, turns, sk, speaker_a in get_sessions(conv):
                convs_fmt = turns_to_conversations(turns, speaker_a)
                agent.ingest(date, convs_fmt)
            agent.build_summaries()

            # ── Evaluate each QA pair ─────────────────────────────────
            print(f"    Answering {len(qa_pairs)} questions...")
            for qa in tqdm(qa_pairs, desc="    QA", leave=False):
                q, a, cat = qa["question"], qa["answer"], qa["category"]

                r_no_mem    = ask(q)
                r_naive_rag = ask_naive_rag(q, naive_rag, k=5)
                r_rsum      = ask(q, rsum_mem[:4000])
                r_mv2       = agent.chat(q)

                s_no_mem    = score_answer(r_no_mem,    a, cat)
                s_naive_rag = score_answer(r_naive_rag, a, cat)
                s_rsum      = score_answer(r_rsum,      a, cat)
                s_mv2       = score_answer(r_mv2,       a, cat)

                for method, score in [("no_memory",  s_no_mem),
                                       ("naive_rag",  s_naive_rag),
                                       ("rsum",       s_rsum),
                                       ("memory_v2",  s_mv2)]:
                    all_scores[method].append(score)
                    cat_scores[method][cat].append(score)

                records.append({
                    "conv_id": conv["sample_id"], "category": cat,
                    "question": q, "answer": a,
                    "no_memory":  {"response": r_no_mem,    "score": s_no_mem},
                    "naive_rag":  {"response": r_naive_rag, "score": s_naive_rag},
                    "rsum":       {"response": r_rsum,      "score": s_rsum},
                    "memory_v2":  {"response": r_mv2,       "score": s_mv2},
                })

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ── Print results ─────────────────────────────────────────────────
    print_results(all_scores, cat_scores, eval_cats)

    # Save
    summary = {
        m: {
            "overall": float(np.mean(all_scores[m])) if all_scores[m] else 0,
            "by_category": {
                CATEGORY_NAMES[c]: float(np.mean(cat_scores[m][c])) if cat_scores[m][c] else 0
                for c in eval_cats
            },
            "n": len(all_scores[m]),
        }
        for m in METHODS
    }
    with open("results/locomo_results.json", "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print("\nSaved → results/locomo_results.json")


def get_sessions_list(conv):
    c = conv["conversation"]
    return [k for k in c if re.match(r"^session_\d+$", k)]


def print_results(all_scores, cat_scores, eval_cats):
    PAPER = {   # GPT-3.5-turbo full-context from LoComo paper (approx)
        "single-hop": 0.354, "temporal": 0.224,
        "multi-hop":  0.191, "commonsense": 0.426,
    }
    w = 20
    cats_sorted = sorted(eval_cats)
    header_cats = "  ".join(f"{CATEGORY_NAMES[c]:>13}" for c in cats_sorted)
    print(f"\n{'='*80}")
    print(f"{'Method':<{w}}  {'Overall':>8}  {header_cats}")
    print("─" * 80)
    for method in ("no_memory", "naive_rag", "rsum", "memory_v2"):
        overall = np.mean(all_scores[method]) if all_scores[method] else 0
        cat_vals = "  ".join(
            f"{np.mean(cat_scores[method][c]) if cat_scores[method][c] else 0:>13.3f}"
            for c in cats_sorted
        )
        print(f"{method:<{w}}  {overall:>8.3f}  {cat_vals}")
    print("─" * 80)
    print(f"\nPaper reference (GPT-3.5-turbo, full context):")
    for cat, val in PAPER.items():
        print(f"  {cat}: {val:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_convs",   type=int, default=3,
                        help="Number of LoComo conversations to evaluate (max 10)")
    parser.add_argument("--categories",  type=int, nargs="+", default=[1, 2, 4],
                        help="QA categories to evaluate (1=multi-hop 2=temporal 3=commonsense 4=single-hop 5=adversarial)")
    args = parser.parse_args()
    run(args)
