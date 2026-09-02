"""
Probing-based evaluation on MSC personas.

For each MSC session-5 dialogue:
  1. Convert persona facts (Speaker 1 + Speaker 2) → factual Q&A pairs via LLM.
  2. Ingest sessions 0-4 into each method's memory.
  3. Ask each probing question to each method.
  4. LLM judge scores each answer 0 or 1.

Methods compared:
  - no_memory    : question asked with no context at all
  - rsum         : recursive summary (M_N = LLM(S_N, M_{N-1})) used as context
  - memory_v2    : full extraction + summaries + context_builder

Metric: Recall@1 — fraction of questions answered correctly (averaged over all Q&A pairs).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

# ── API key ──────────────────────────────────────────────────────────
for line in open(os.path.join(os.path.dirname(__file__), ".env")):
    if "OPENAI_API_KEY" in line and not line.strip().startswith("#"):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MSC_TEST = "baselines/Rsum/data/msc_dialogue/session_5/test.txt"
QA_CACHE  = "results/probing_qa_cache.json"

# ── Helpers ───────────────────────────────────────────────────────────

def call(prompt, system="You are a helpful assistant.", model="gpt-4o-mini", retries=5):
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": prompt}],
                temperature=0, max_tokens=300,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if i < retries - 1:
                time.sleep(3)
    return ""

def msc_date(session_idx: int) -> str:
    return (datetime(2020, 1, 1) + timedelta(weeks=session_idx)).strftime("%Y-%m-%d")

def load_dialogs(path, n):
    dialogs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n: break
            dialogs.append(json.loads(line))
    return dialogs

# ── Step 1: Generate Q&A pairs from persona facts ────────────────────

QA_SYSTEM = """\
Convert each persona fact into ONE factual probing question and its short answer.
The question should be answerable from memory if the AI read the conversations.
Output JSON: {"qa": [{"question": "...", "answer": "..."}]}
One item per fact. Keep answers short (1-5 words when possible).\
"""

def generate_qa(personas_0: list, personas_1: list, dial_id: int) -> list:
    """Generate Q&A pairs for both speakers' personas."""
    all_facts = (
        [f"[Speaker 1] {f}" for f in personas_0] +
        [f"[Speaker 2] {f}" for f in personas_1]
    )
    facts_text = "\n".join(all_facts)
    raw = call(
        f"Persona facts:\n{facts_text}\n\nGenerate one Q&A per fact.",
        system=QA_SYSTEM,
    )
    try:
        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
            raw = raw.rsplit("```", 1)[0].strip()
        return json.loads(raw).get("qa", [])
    except Exception:
        return []

def load_or_build_qa_cache(dialogs) -> dict:
    if os.path.exists(QA_CACHE):
        with open(QA_CACHE) as f:
            return json.load(f)

    print("Generating Q&A pairs from MSC persona facts...")
    cache = {}
    for i, d in enumerate(tqdm(dialogs, desc="Q&A generation")):
        p0, p1 = d["personas"][0], d["personas"][1]
        qa = generate_qa(p0, p1, i)
        cache[str(i)] = qa

    os.makedirs("results", exist_ok=True)
    with open(QA_CACHE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  Cached {sum(len(v) for v in cache.values())} Q&A pairs → {QA_CACHE}")
    return cache

# ── Step 2: Build memory per method ──────────────────────────────────

def sessions_as_conversations(dial):
    """Yield (date, conversations) for persona + each previous session."""
    # Persona as session 0
    p0 = " ".join(dial["personas"][0])
    p1 = " ".join(dial["personas"][1])
    persona_conv = [{"time_of_day": "Morning", "turns": [
        {"speaker": "User",   "text": f"About me: {p0}"},
        {"speaker": "System", "text": f"About me: {p1}"},
    ]}]
    yield msc_date(0), persona_conv

    for i, prev in enumerate(dial["previous_dialogs"]):
        turns = []
        for t in prev["dialog"]:
            speaker = "User" if t["id"] == "Speaker 1" else "System"
            turns.append({"speaker": speaker, "text": t["text"]})
        yield msc_date(i + 1), [{"time_of_day": "Morning", "turns": turns}]

def build_rsum_memory(dial, prompts) -> str:
    """Recursive summary: M_i = LLM(S_i, M_{i-1})."""
    instruction = prompts["msc"]["gpt-3.5-turbo"]["update_memory"]
    system = "You are an advanced AI language model with the ability to keep track of dialog information between speakers."
    memory = "Empty"
    for _, convs in sessions_as_conversations(dial):
        session_text = " ".join(
            f"{t['speaker']}: {t['text']}" for c in convs for t in c["turns"]
        )
        prompt = (
            f"**Instruction** {instruction}\n"
            f"**Test** [Previous Memory] {memory} "
            f"[Dialogue Context] {session_text} [Updated Memory]"
        )
        result = call(prompt, system=system, max_tokens=400)
        for prefix in ("Updated memory:", "[Updated Memory]"):
            if prefix in result:
                result = result.split(prefix)[-1]
        memory = result.replace("\n", " ").strip() or memory
    return memory

# ── Step 3: Ask probing questions ─────────────────────────────────────

ASK_SYSTEM = """\
You are a personalized AI assistant. Use the provided memory context to answer
the question as specifically as possible. If you don't know, say "I don't know".\
"""

def ask_no_memory(question: str) -> str:
    return call(question, system=ASK_SYSTEM)

def ask_with_context(question: str, context: str) -> str:
    prompt = f"Memory:\n{context}\n\nQuestion: {question}"
    return call(prompt, system=ASK_SYSTEM)

# ── Step 4: LLM judge ────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are a strict factual evaluator.
Given a question, the correct answer, and a model response, reply with exactly
"1" if the response correctly contains or implies the correct answer,
or "0" if it does not or says "I don't know".\
"""

def judge(question: str, correct_answer: str, model_response: str) -> int:
    prompt = (
        f"Question: {question}\n"
        f"Correct answer: {correct_answer}\n"
        f"Model response: {model_response}"
    )
    r = call(prompt, system=JUDGE_SYSTEM)
    return 1 if r.strip().startswith("1") else 0

# ── Main evaluation loop ──────────────────────────────────────────────

def run(args):
    os.makedirs("results", exist_ok=True)
    dialogs = load_dialogs(MSC_TEST, args.max_dialogs)
    print(f"Loaded {len(dialogs)} dialogues")

    qa_cache = load_or_build_qa_cache(dialogs)

    # Load Rsum prompts
    rsum_prompts = json.load(open("baselines/Rsum/prompt.json"))

    scores = {"no_memory": [], "rsum": [], "memory_v2": []}
    records = []

    for dial_idx, dial in enumerate(tqdm(dialogs, desc="Dialogues")):
        qa_pairs = qa_cache.get(str(dial_idx), [])
        if not qa_pairs:
            continue

        # ── Build Rsum memory (once per dialogue) ──────────────────
        rsum_mem = build_rsum_memory(dial, rsum_prompts)

        # ── Build memory_v2 (once per dialogue) ────────────────────
        from memory_v2 import LifelongAgent
        tmp = tempfile.mkdtemp(prefix="probe_eval_")
        try:
            agent = LifelongAgent(base_dir=tmp, chat_model="gpt-4o-mini",
                                  extract_model="gpt-4o-mini")
            for date, convs in sessions_as_conversations(dial):
                agent.ingest(date, convs)
            agent.build_summaries()

            # ── Ask all probing questions for this dialogue ─────────
            for qa in qa_pairs:
                q, a = qa["question"], qa["answer"]

                r_no_mem = ask_no_memory(q)
                r_rsum   = ask_with_context(q, rsum_mem)
                r_mv2    = agent.chat(q)

                s_no_mem = judge(q, a, r_no_mem)
                s_rsum   = judge(q, a, r_rsum)
                s_mv2    = judge(q, a, r_mv2)

                scores["no_memory"].append(s_no_mem)
                scores["rsum"].append(s_rsum)
                scores["memory_v2"].append(s_mv2)

                records.append({
                    "dial_idx": dial_idx,
                    "question": q, "answer": a,
                    "no_memory": {"response": r_no_mem, "score": s_no_mem},
                    "rsum":      {"response": r_rsum,   "score": s_rsum},
                    "memory_v2": {"response": r_mv2,    "score": s_mv2},
                })

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ── Results ─────────────────────────────────────────────────────
    total = len(scores["no_memory"])
    print(f"\n{'='*52}")
    print(f"{'Method':<20} {'Recall@1':>10} {'Correct':>8} {'Total':>6}")
    print(f"{'─'*52}")

    PAPER_MB = 0.654  # MemoryBank reported probing accuracy on SiliconFriend
    result_dict = {}
    for method in ("no_memory", "rsum", "memory_v2"):
        correct = sum(scores[method])
        recall  = correct / total if total else 0
        print(f"{method:<20} {recall:>10.3f} {correct:>8} {total:>6}")
        result_dict[method] = {"recall": recall, "correct": correct, "total": total}

    print(f"{'─'*52}")
    print(f"  MemoryBank (SiliconFriend, diff dataset): ~{PAPER_MB:.3f}")
    print(f"{'='*52}")
    print(f"\nTotal Q&A pairs evaluated: {total}")

    with open("results/probing_results.json", "w") as f:
        json.dump({"summary": result_dict, "records": records}, f, indent=2)
    print("Saved → results/probing_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_dialogs", type=int, default=20)
    parser.add_argument("--regenerate_qa", action="store_true",
                        help="Force regenerate Q&A cache even if it exists")
    args = parser.parse_args()

    if args.regenerate_qa and os.path.exists(QA_CACHE):
        os.remove(QA_CACHE)

    run(args)
