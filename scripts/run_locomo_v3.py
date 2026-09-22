"""
Run memory_v3 on LoCoMo.

    venv/bin/python scripts/run_locomo_v3.py --conversations 1     # one, to sanity-check
    venv/bin/python scripts/run_locomo_v3.py                       # all 10

LoCoMo is the generalization check: the synthetic eval set was authored alongside this
system, so a good score there says little about anything else. LoCoMo was not, and it
differs in every way that could be overfitted — two speakers instead of one, events
and facts instead of daily preference cycles, real timestamps spanning months, and a
category of questions whose correct answer is to refuse.

Nothing in the memory system is specialised for it. The same extraction prompt runs
on both datasets — it pulls out preferences (recurring choices). Facts and events are
not extracted into structured rows; they remain in the conversation chunks and
summaries, reachable through semantic search and date lookup.

Scoring is the paper's own, reused from evaluation/evaluate_locomo.py rather than
reimplemented, so the numbers sit alongside the published baselines:
  cat 1 multi-hop    split-F1 over comma-separated sub-answers
  cat 2,3,4          token F1 with Porter stemming
  cat 5 adversarial  1.0 only if the model abstains
"""

import argparse
import json
import os
import re
import shutil
import string
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from nltk.stem import PorterStemmer
from tqdm import tqdm

from memory_v3.agent import AgenticMemoryAgent

LOCOMO_PATH = os.path.join(PROJECT_ROOT, "baselines", "locomo", "data", "locomo10.json")

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "commonsense",
                  4: "single-hop", 5: "adversarial"}

_stemmer = PorterStemmer()


# ── Paper-exact scoring ─────────────────────────────────────────────
# Copied from the LoCoMo reference implementation rather than imported, so this
# runner stays independent of the v2 harness and pulls in no other provider's SDK.
# Keeping it identical is what lets these numbers sit beside the published ones.

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
    if category in (2, 3, 4):                     # temporal, commonsense, single-hop
        return _token_f1(pred, answer)
    if category == 1:                             # multi-hop: one score per sub-answer
        preds = [p.strip() for p in pred.split(",")]
        return float(np.mean([
            max(_token_f1(p, r.strip()) for p in preds)
            for r in answer.split(",")
        ]))
    if category == 5:                             # adversarial: abstaining is correct
        abstain = ("no information", "not mentioned", "not provided", "don't know",
                   "do not know", "cannot find", "no record")
        return 1.0 if any(ph in pred.lower() for ph in abstain) else 0.0
    return 0.0


# ── Dataset parsing ─────────────────────────────────────────────────

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
    """
    LoCoMo turns → pipeline turns, keeping the real names.

    The existing v2 adapter rewrites speakers to User/System, which throws away the
    only thing that says whose life a statement is about. Questions here name the
    person, so the names have to survive ingestion.
    """
    out = []
    for t in turns:
        if "text" not in t:
            continue
        text = t["text"]
        if t.get("blip_caption"):
            text += f" [shared image: {t['blip_caption']}]"
        out.append({"speaker": t["speaker"], "text": text})
    return out


def ingest_conversation(agent: AgenticMemoryAgent, conv: dict) -> dict:
    """Replay every session of one LoCoMo conversation in date order."""
    stats = {"sessions": 0, "added": 0, "chunks": 0, "errors": []}

    for date, turns in get_sessions(conv):
        formatted = session_to_turns(turns)
        if not formatted:
            continue
        report = agent.ingest(
            date, [{"time_of_day": "All Day", "turns": formatted}]
        )
        stats["sessions"] += 1
        stats["added"] += report.added
        stats["chunks"] += report.chunks_indexed
        stats["errors"].extend(report.errors)

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=int, default=10)
    ap.add_argument("--questions", type=int, default=None,
                    help="cap questions per conversation")
    ap.add_argument("--categories", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--run", default="locomo_v3")
    ap.add_argument("--summaries", action="store_true",
                    help="build the summary hierarchy after ingesting each conversation")
    args = ap.parse_args()

    run_dir = os.path.join(PROJECT_ROOT, "results", args.run)
    os.makedirs(run_dir, exist_ok=True)

    with open(LOCOMO_PATH) as f:
        dataset = json.load(f)[: args.conversations]

    records = []
    by_category = defaultdict(list)
    ingest_summary = []

    for i, conv in enumerate(dataset, 1):
        sample_id = conv.get("sample_id", f"conv{i}")
        speakers = f"{conv['conversation']['speaker_a']} & {conv['conversation']['speaker_b']}"
        print(f"\n{'=' * 64}\n[{i}/{len(dataset)}] {sample_id} — {speakers}")

        # Each conversation is its own person-pair, so it gets its own store.
        store_dir = os.path.join(run_dir, "stores", sample_id)
        shutil.rmtree(store_dir, ignore_errors=True)
        os.makedirs(store_dir, exist_ok=True)

        agent = AgenticMemoryAgent(
            base_dir=store_dir,
            model=args.model,
        )

        t0 = time.time()
        stats = ingest_conversation(agent, conv)
        stats.update(
            sample_id=sample_id,
            seconds=round(time.time() - t0, 1),
            entities=agent.store.get_all_entities(),
            speakers=agent.store.get_all_speakers(),
        )
        ingest_summary.append(stats)
        print(f"  ingested {stats['sessions']} sessions → {stats['added']} records, "
              f"{stats['chunks']} chunks, {stats['seconds']}s")
        print(f"  speakers: {stats['speakers']}")
        print(f"  entities: {', '.join(stats['entities'][:12])}")

        if args.summaries:
            agent.build_summaries(force=True)

        questions = [q for q in conv["qa"] if q.get("category") in args.categories]
        if args.questions:
            questions = questions[: args.questions]

        for qa in tqdm(questions, desc=f"  QA {sample_id}", leave=False):
            question = qa["question"]
            truth = qa.get("answer", "")
            category = qa["category"]
            try:
                pred = agent.chat(question)
            except Exception as e:
                pred = f"(error: {e})"

            score = score_answer(pred, truth, category)
            by_category[category].append(score)
            records.append({
                "sample_id": sample_id, "question": question, "answer": str(truth),
                "response": pred, "category": category,
                "category_name": CATEGORY_NAMES[category], "score": score,
            })

        done = [r for r in records if r["sample_id"] == sample_id]
        print(f"  scored {len(done)} questions, mean {np.mean([r['score'] for r in done]):.3f}")

        # Write after each conversation so a later failure costs nothing already done.
        _save(run_dir, args, records, by_category, ingest_summary)

    _report(records, by_category)
    print(f"\nSaved to {os.path.join(run_dir, 'locomo_results.json')}")


def _save(run_dir, args, records, by_category, ingest_summary):
    payload = {
        "model": args.model,
        "summary": {
            "n": len(records),
            "overall": float(np.mean([r["score"] for r in records])) if records else 0.0,
            "by_category": {
                CATEGORY_NAMES[c]: {"n": len(v), "score": float(np.mean(v))}
                for c, v in sorted(by_category.items())
            },
        },
        "ingestion": ingest_summary,
        "records": records,
    }
    with open(os.path.join(run_dir, "locomo_results.json"), "w") as f:
        json.dump(payload, f, indent=2)


def _report(records, by_category):
    if not records:
        print("\nNothing scored.")
        return
    print(f"\n{'=' * 64}")
    print(f"  memory_v3 on LoCoMo — {len(records)} questions")
    print(f"  overall  {np.mean([r['score'] for r in records]):.3f}")
    print()
    for c, scores in sorted(by_category.items()):
        print(f"    {CATEGORY_NAMES[c]:<14} n={len(scores):<5} {np.mean(scores):.3f}")
    print(f"{'=' * 64}")
    print("  Paper baseline (GPT-3.5, full context): overall ~0.27, "
          "single-hop ~0.35, temporal ~0.22, multi-hop ~0.19")


if __name__ == "__main__":
    main()
