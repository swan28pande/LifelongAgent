"""
Evaluate memory_v2 on MSC session-5 and compare against baselines.

Baselines compared (all from baselines/Rsum/results/):
  - ChatGPT context-only
  - ChatGPT-Rsum

Our method:
  - memory_v2 (extraction + hierarchical summaries + context builder)

Metrics (paper-exact): F1, BLEU-1, BLEU-2
"""

import argparse
import collections
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

# ── Load API key ─────────────────────────────────────────────────────

for line in open(os.path.join(os.path.dirname(__file__), ".env")):
    if "OPENAI_API_KEY" in line and not line.strip().startswith("#"):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

MSC_TEST_FILE = "baselines/Rsum/data/msc_dialogue/session_5/test.txt"

# ── Metrics (paper-exact) ────────────────────────────────────────────

import re as _re
_re_art  = _re.compile(r"\b(a|an|the)\b")
_re_punc = _re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def _norm(s):
    s = s.lower()
    s = _re_punc.sub(" ", s)
    s = _re_art.sub(" ", s)
    return " ".join(s.split())

def compute_f1(preds, refs):
    total = 0
    for p, r in zip(preds, refs):
        pt = word_tokenize(_norm(p))
        rt = word_tokenize(_norm(r))
        common = collections.Counter(pt) & collections.Counter(rt)
        n = sum(common.values())
        if not pt or not rt or n == 0:
            continue
        prec, rec = n / len(pt), n / len(rt)
        total += 2 * prec * rec / (prec + rec)
    return total / len(preds) * 100

def compute_bleu(preds, refs):
    smooth = SmoothingFunction()
    hyps  = [word_tokenize(p) for p in preds]
    refs_ = [word_tokenize(r) for r in refs]
    b1 = corpus_bleu(refs_, hyps, weights=(0.5, 0.5),              smoothing_function=smooth.method7) * 100
    b2 = corpus_bleu(refs_, hyps, weights=(0.333, 0.333, 0.334),   smoothing_function=smooth.method7) * 100
    return b1, b2

def metrics(preds, refs):
    f1      = compute_f1(preds, refs)
    b1, b2  = compute_bleu(preds, refs)
    return {"F1": f1, "BLEU1": b1, "BLEU2": b2, "n": len(preds)}

# ── MSC data helpers ─────────────────────────────────────────────────

def load_msc_test(path, max_dialogs):
    dialogs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_dialogs:
                break
            dialogs.append(json.loads(line))
    return dialogs

def msc_to_conversations(session_dialog, session_idx):
    """Convert a MSC previous_dialog entry to memory_v2 conversation format."""
    turns = []
    for turn in session_dialog:
        speaker = "User" if turn["id"] == "Speaker 1" else "System"
        turns.append({"speaker": speaker, "text": turn["text"]})
    return [{"time_of_day": "Morning", "turns": turns}]

def msc_session_date(session_idx):
    """Assign a synthetic date per session so memory_v2 can track time."""
    base = datetime(2020, 1, 1)
    return (base + timedelta(weeks=session_idx)).strftime("%Y-%m-%d")

def extract_test_turns(current_dialog):
    """
    Return list of (context_so_far, ground_truth_response) for every
    System turn in the current session. context_so_far is all turns
    up to (but not including) the System turn.
    """
    turns, result = [], []
    for turn in current_dialog:
        if turn["id"] == "Speaker 2":          # System turn → generate here
            result.append((" ".join(turns), turn["text"]))
        turns.append(f"{'User' if turn['id']=='Speaker 1' else 'System'}: {turn['text']}")
    return result

# ── memory_v2 evaluation ─────────────────────────────────────────────

RESPONSE_SYSTEM = (
    "You are a personalized AI companion in a multi-session conversation. "
    "Use the memory context provided to respond naturally and consistently. "
    "Keep your reply to 1-3 sentences."
)

def run_memory_v2_dialogue(dial, model, n_workers):
    """
    For one MSC dialogue:
      1. Ingest persona facts + previous sessions into a fresh memory_v2 store.
      2. Build summaries.
      3. Generate responses for all System turns in the current session.
    Returns list of (prediction, reference) pairs.
    """
    from memory_v2 import LifelongAgent
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    tmp = tempfile.mkdtemp(prefix="msc_eval_")
    try:
        agent = LifelongAgent(base_dir=tmp, chat_model=model, extract_model="gpt-4o-mini")

        # Ingest persona facts as day-0 (very short conversation)
        personas = dial.get("personas", [[], []])
        persona_text = " ".join(personas[0] + personas[1])
        if persona_text.strip():
            persona_conv = [{"time_of_day": "Morning", "turns": [
                {"speaker": "System", "text": f"Background: {persona_text}"}
            ]}]
            agent.ingest(msc_session_date(0), persona_conv)

        # Ingest previous sessions
        for i, prev in enumerate(dial["previous_dialogs"]):
            convs = msc_to_conversations(prev["dialog"], i + 1)
            agent.ingest(msc_session_date(i + 1), convs)

        # Build summaries (lifetime is what matters most)
        agent.build_summaries()

        # Generate responses for current session
        llm    = ChatOpenAI(model=model, temperature=0)
        pairs  = []
        test_turns = extract_test_turns(dial["dialog"])

        for context_str, reference in test_turns:
            context = agent.ctx_builder.build(
                context_str.split("\n")[-1] if context_str else "Hello",
                n_recent=4, n_relevant=4,
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", RESPONSE_SYSTEM + "\n\n" + context),
                ("human", context_str or "Hello"),
            ])
            try:
                resp = (prompt | llm).invoke({})
                text = resp.content.strip()
                # Strip echoed role prefix if LLM adds it
                for prefix in ("System:", "Assistant:", "AI:"):
                    if text.startswith(prefix):
                        text = text[len(prefix):].strip()
                pairs.append((text, reference))
            except Exception as e:
                pairs.append(("", reference))

        return pairs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_memory_v2(dialogs, model="gpt-4o-mini", n_workers=3):
    preds, refs = [], []
    for dial in tqdm(dialogs, desc="memory_v2"):
        pairs = run_memory_v2_dialogue(dial, model, n_workers)
        for p, r in pairs:
            preds.append(p)
            refs.append(r)
    return preds, refs

# ── Load saved baseline results ──────────────────────────────────────

def load_baseline(name, path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d["predictions"], d["references"]
    except Exception as e:
        print(f"  Warning: could not load {name}: {e}")
        return None, None

# ── Comparison table ─────────────────────────────────────────────────

PAPER = {
    "ChatGPT context-only": {"F1": 19.41, "BLEU1": 21.23, "BLEU2": 12.24},
    "ChatGPT-Rsum":         {"F1": 20.48, "BLEU1": 21.83, "BLEU2": 12.59},
    "ChatGPT-MemoryBank":   {"F1": 20.28, "BLEU1": 21.82, "BLEU2": 12.58},
}

def print_table(results):
    w = 30
    print(f"\n{'Method':<{w}} {'F1':>7} {'BLEU-1':>8} {'BLEU-2':>8}  {'n':>5}")
    print("─" * (w + 35))
    for name, m in results.items():
        paper = PAPER.get(name, {})
        f1_mark  = "✓" if paper and abs(m["F1"]   - paper["F1"])   < 1.5 else ""
        b1_mark  = "✓" if paper and abs(m["BLEU1"] - paper["BLEU1"]) < 1.5 else ""
        print(f"{name:<{w}} {m['F1']:>7.2f} {m['BLEU1']:>8.2f} {m['BLEU2']:>8.2f}  {m['n']:>5}")
    print("─" * (w + 35))
    print("\nPaper claims:")
    for name, p in PAPER.items():
        print(f"  {name:<{w-2}} F1={p['F1']}  BLEU-1={p['BLEU1']}  BLEU-2={p['BLEU2']}")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_dialogs", type=int, default=20)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip re-running baselines, load saved predictions only")
    args = parser.parse_args()

    os.makedirs("results/legacy/msc", exist_ok=True)
    dialogs = load_msc_test(MSC_TEST_FILE, args.max_dialogs)
    print(f"Loaded {len(dialogs)} MSC session-5 dialogues")

    results = {}

    # ── Load saved baseline predictions ──────────────────────────────
    ctx_preds, ctx_refs = load_baseline(
        "context-only",
        "baselines/Rsum/results/context_only_predictions.json"
    )
    if ctx_preds:
        results["ChatGPT context-only"] = metrics(
            ctx_preds[:args.max_dialogs * 6],   # rough cap to same n
            ctx_refs[:args.max_dialogs * 6],
        )

    rsum_preds, rsum_refs = load_baseline(
        "Rsum",
        "baselines/Rsum/results/rsum_predictions.json"
    )
    if rsum_preds:
        results["ChatGPT-Rsum"] = metrics(
            rsum_preds[:args.max_dialogs * 6],
            rsum_refs[:args.max_dialogs * 6],
        )

    # ── Run memory_v2 ─────────────────────────────────────────────────
    print(f"\nRunning memory_v2 on {len(dialogs)} dialogues...")
    mv2_preds, mv2_refs = run_memory_v2(dialogs, model=args.model, n_workers=args.workers)

    with open("results/legacy/msc/memory_v2_predictions.json", "w") as f:
        json.dump({"predictions": mv2_preds, "references": mv2_refs}, f, indent=2)

    results["memory_v2 (ours)"] = metrics(mv2_preds, mv2_refs)

    # ── Print comparison ──────────────────────────────────────────────
    print_table(results)

    with open("results/legacy/msc/msc_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/legacy/msc/msc_comparison.json")
