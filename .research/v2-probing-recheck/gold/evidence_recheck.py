#!/usr/bin/env python3
"""Current transcript extracts, exact-payload matching, and baseline comparisons."""
from __future__ import annotations

import collections
import datetime as dt
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT / ".research/v2-probing-audit/gold"
namespace = runpy.run_path(str(OLD / "audit_gold.py"), run_name="source_index_import")
namespace["Source"].__init__.__globals__["ROOT"] = ROOT
Source = namespace["Source"]


def read(path):
    return json.loads(path.read_text())


def key(row):
    return row["user_id"], row["type"], row["domain"], row["question"]


def main():
    current = read(OUT / "coverage.json")
    baseline = read(OLD / "coverage.json")
    index = collections.defaultdict(list)
    for row in current:
        index[key(row)].append(row)
    domains = {(r["user_id"], r["id"]): r["domain"] for r in baseline}
    matches = []
    for filename in ("issues.json", "observation_limits.json", "uncertainties.json"):
        for prior in read(OLD / filename):
            prior = {**prior, "domain": domains[prior["user_id"], prior["id"]]}
            candidates = index.get(key(prior), [])
            matches.append({
                "old_id": prior["id"], "old_original_id": prior["original_id"],
                "old_probe_date": prior["probe_date"], "category": prior["category"],
                "old_question": prior["question"],
                "matching_current_question_texts": [{
                    "id": r["id"], "original_id": r["original_id"], "probe_date": r["probe_date"],
                    "question_source": r["source"]["question"],
                    "original_id_equal": r["original_id"] == prior["original_id"],
                    "answer_accept_evidence_equal": (
                        r["observed_answer"] == next(x for x in baseline if x["id"] == prior["id"])["observed_answer"]
                        and r["observed_accept"] == next(x for x in baseline if x["id"] == prior["id"])["observed_accept"]
                        and r["evidence_days"] == next(x for x in baseline if x["id"] == prior["id"])["evidence_days"]),
                    "is_retrospective": r["is_retrospective"],
                    "current_issue_categories": r["issue_categories"],
                    "current_observation_limits": r["observation_limit_categories"],
                    "current_uncertainties": r["uncertainty_categories"],
                } for r in candidates],
                "match_status": "question_present" if candidates else "absent_from_current_sample",
            })
    (OUT / "baseline_finding_matches.json").write_text(json.dumps(matches, indent=2) + "\n")

    transcripts = []
    boundary_rows = [r for r in current if "change_citations_do_not_observe_change" in r["issue_categories"]]
    for uid in ("u2", "u3", "u4", "u5"):
        src = Source(ROOT / f"datasets/v2/{uid}/conversations.json")
        world = read(ROOT / f"datasets/v2/{uid}/world_state.json")
        start = dt.date.fromisoformat(world["start_date"])
        user_turns = [{"date": date, "day": (dt.date.fromisoformat(date) - start).days + 1,
                       "text": turn["text"], "source": src.loc(f"/sessions/{date}/turns/{i}/text")}
                      for date, session in src.data["sessions"].items()
                      for i, turn in enumerate(session["turns"]) if turn["speaker"] == "user"]
        for row in boundary_rows:
            if row["user_id"] != uid:
                continue
            regime = next(r for r in world["regimes"] if r["id"] == row["regime_id"])
            anchor = min(x for x in (regime["cause_day"], regime["start"], regime["first_mention_day"]) if x is not None)
            end = min(row["viewpoint_day"], max(regime["first_mention_day"] + 14, row["first_distinguishing_mention_day"] or 0))
            transcripts.append({"id": row["id"], "domain": row["domain"], "probe_date": row["probe_date"],
                                "source": row["source"]["question"], "regime": regime,
                                "counterfactual_choices": row.get("counterfactual_change_starts"),
                                "prefix_user_turns_near_change": [t for t in user_turns if anchor - 1 <= t["day"] <= end],
                                "first_distinguishing_turns": [t for t in user_turns if t["day"] == row["first_distinguishing_mention_day"]],
                                "first_distinguishing_date_after_probe": bool(row["first_distinguishing_mention_day"] and row["first_distinguishing_mention_day"] > row["viewpoint_day"]),
                                "interpretation_limit": "Counterfactuals reproduce all declared choice values in the tested interval, not every free-text sentence or every simulator randomness constraint. Full displayed first-mention transcripts were manually read; no precise dated switch declaration was present."})
        if uid == "u5":
            for row in current:
                if row["user_id"] == uid and "gold_answer_mismatch_at_probe" in row["issue_categories"]:
                    transcripts.append({"id": row["id"], "check": "stale_current_hobbies",
                                        "reference": row["observed_answer"], "expected": row["expected_answer"],
                                        "probe_date": row["probe_date"], "source": row["source"]["question"],
                                        "turns": [t for t in user_turns if t["date"] in ("2027-01-13", "2027-01-22", "2027-01-25", "2027-04-24")],
                                        "evidence": "January mentions piano; April24 explicitly stops it. January reference omits an active, observed hobby."})
    (OUT / "manual_transcript_checks.json").write_text(json.dumps(transcripts, indent=2, ensure_ascii=False) + "\n")

    retained_pool_defects = []
    for uid in ("u2", "u3", "u4", "u5"):
        pool = Source(ROOT / f"datasets/v2/{uid}/qa_pool.json")
        for i, q in enumerate(pool.data):
            old_matches = [m for m in matches if m["old_question"] == q["question"] and m["old_id"].startswith(uid)]
            if old_matches:
                retained_pool_defects.append({"id": q["id"], "question": q["question"], "answer": q["answer"],
                                              "accept": q["accept"], "evidence_days": q["evidence_days"],
                                              "source": pool.loc(f"/{i}/question"),
                                              "prior_ids": [m["old_id"] for m in old_matches],
                                              "prior_categories": sorted({m["category"] for m in old_matches}),
                                              "current_selected_ids": [r["id"] for r in current if r["user_id"] == uid and r["question"] == q["question"]]})
    (OUT / "prior_findings_current_pool.json").write_text(json.dumps(retained_pool_defects, indent=2) + "\n")
    print(json.dumps({"baseline_finding_records": len(matches), "absent_baseline_records": sum(not r["matching_current_question_texts"] for r in matches),
                      "manual_check_records": len(transcripts), "baseline_defect_texts_retained_in_pool": len(retained_pool_defects)}, indent=2))


if __name__ == "__main__":
    main()
