"""
Shared pieces for every LoCoMo runner, so all systems are parsed and scored identically.

Scoring is the LoCoMo paper's own, copied rather than imported so the runners pull in
no other harness:
  cat 1 multi-hop    split-F1 over comma-separated sub-answers
  cat 2,3,4          token F1 with Porter stemming
  cat 5 adversarial  1.0 only if the model abstains
"""

import json
import os
import re
import string
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from nltk.stem import PorterStemmer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCOMO_PATH = os.path.join(PROJECT_ROOT, "baselines", "locomo", "data", "locomo10.json")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "locomo")

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "commonsense",
                  4: "single-hop", 5: "adversarial"}

# Used by every runner that answers in a single LLM call, so the only thing that
# differs between them is what gets retrieved.
ANSWER_SYSTEM = """\
You answer questions about two people using only the memory context provided. Items
may carry the date of the conversation they came from; resolve relative dates against it.

Give the shortest answer that is complete. Prefer a few words over a full sentence.
No preamble, no extra context, no bullet points unless multiple items are asked for.

If the context does not contain the answer, say only: "I don't know."
"""

_stemmer = PorterStemmer()


# ── Paper-exact scoring ─────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = str(s).replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s, flags=re.IGNORECASE)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.lower().split())


def _token_f1(pred: str, ref: str) -> float:
    p = [_stemmer.stem(w) for w in _normalize(pred).split()]
    r = [_stemmer.stem(w) for w in _normalize(ref).split()]
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0 or not p or not r:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def score_answer(pred: str, answer, category: int) -> float:
    pred, answer = str(pred), str(answer)
    if category in (2, 3, 4):
        return _token_f1(pred, answer)
    if category == 1:
        preds = [p.strip() for p in pred.split(",")]
        return float(np.mean([
            max(_token_f1(p, r.strip()) for p in preds)
            for r in answer.split(",")
        ]))
    if category == 5:
        abstain = ("no information", "not mentioned", "not provided", "don't know",
                   "do not know", "cannot find", "no record")
        return 1.0 if any(ph in pred.lower() for ph in abstain) else 0.0
    return 0.0


# ── Dataset parsing ─────────────────────────────────────────────────

def load_dataset(n_conversations: int) -> list:
    with open(LOCOMO_PATH) as f:
        return json.load(f)[:n_conversations]


def parse_session_date(raw: str) -> str:
    """'1:56 pm on 8 May, 2023' → '2023-05-08'."""
    part = raw.split(" on ")[-1].strip().rstrip(".")
    for fmt in ("%d %B, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(part, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return "2020-01-01"


def get_sessions(conv: dict):
    """Yield (date, turns) for each session, oldest first."""
    c = conv["conversation"]
    keys = sorted(
        (k for k in c if re.fullmatch(r"session_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )
    for key in keys:
        yield parse_session_date(c.get(f"{key}_date_time", "")), c[key]


def session_to_turns(turns: list) -> list:
    """LoCoMo turns → [{speaker, text}], keeping real names and image captions."""
    out = []
    for t in turns:
        if "text" not in t:
            continue
        text = t["text"]
        if t.get("blip_caption"):
            text += f" [shared image: {t['blip_caption']}]"
        out.append({"speaker": t["speaker"], "text": text})
    return out


def select_questions(conv: dict, categories: list, limit=None) -> list:
    questions = [q for q in conv["qa"] if q.get("category") in categories]
    return questions[:limit] if limit else questions


# ── Output ──────────────────────────────────────────────────────────

def text_of(content) -> str:
    """LangChain message content → plain string."""
    if isinstance(content, list):
        content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content).strip()


def make_record(sample_id: str, qa: dict, pred: str, **extra) -> dict:
    truth, category = qa.get("answer", ""), qa["category"]
    return {
        "sample_id": sample_id, "question": qa["question"], "answer": str(truth),
        "response": pred, "category": category,
        "category_name": CATEGORY_NAMES[category],
        "score": score_answer(pred, truth, category),
        **extra,
    }


def append_trace(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def summarize(records: list) -> dict:
    by_category = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r["score"])
    return {
        "n": len(records),
        "overall": float(np.mean([r["score"] for r in records])) if records else 0.0,
        "by_category": {
            CATEGORY_NAMES[c]: {"n": len(v), "score": float(np.mean(v))}
            for c, v in sorted(by_category.items())
        },
    }


def save_results(run_dir: str, payload: dict) -> None:
    with open(os.path.join(run_dir, "locomo_results.json"), "w") as f:
        json.dump(payload, f, indent=2)


def print_report(system: str, records: list) -> None:
    if not records:
        print("\nNothing scored.")
        return
    s = summarize(records)
    print(f"\n{'=' * 64}\n  {system} on LoCoMo — {s['n']} questions")
    print(f"  overall  {s['overall']:.3f}\n")
    for name, v in s["by_category"].items():
        print(f"    {name:<14} n={v['n']:<5} {v['score']:.3f}")
    print("=" * 64)
