#!/usr/bin/env python3
"""Offline current-source and coverage checks; only writes beside this script.

The original mechanical source audit is imported with its output redirected here.
Its raw-transcript interpretations are reused only after checking source hashes and
every previously quoted turn; current probe exposure is rebuilt from current data.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / ".research/v2-probing-audit/source"


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def location(path, field, value):
    needle = json.dumps(field) + ": " + json.dumps(value, ensure_ascii=False)
    hits = [n for n, line in enumerate(path.read_text().splitlines(), 1) if needle in line]
    return {"file": str(path.relative_to(ROOT)), "line": hits[0] if hits else None}


def main():
    spec = importlib.util.spec_from_file_location("baseline_source_audit", OLD / "audit_source.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.ROOT, audit.OUT = ROOT, OUT
    audit.main()

    native_spec = importlib.util.spec_from_file_location("baseline_native_checks", OLD / "check_native_world.py")
    native = importlib.util.module_from_spec(native_spec)
    native_spec.loader.exec_module(native)
    native.ROOT, native.OUT = ROOT, OUT
    native.main()
    native_before = read(OLD / "native_world_checks.json")["source_sha256"]
    native_current = read(OUT / "native_world_checks.json")["source_sha256"]
    native_hash_changes = {p: {"baseline": h, "current": native_current.get(p)}
                           for p, h in native_before.items() if h != native_current.get(p)}
    save("native_source_hash_comparison.json", {"checked": len(native_before), "changes": native_hash_changes})

    before = {(r["user"], r["day"]): r for r in jsonl(OLD / "session_coverage.jsonl")}
    current = jsonl(OUT / "session_coverage.jsonl")
    inventory = read(OUT / "inventory.json")
    session_changes = []
    for r in current:
        previous = before.get((r["user"], r["day"]))
        if not previous or previous["session_file_sha256"] != r["session_file_sha256"]:
            session_changes.append({"user": r["user"], "day": r["day"], "date": r["date"],
                                    "current_sha256": r["session_file_sha256"],
                                    "previous_sha256": previous["session_file_sha256"] if previous else None})
    missing = sorted(set(before) - {(r["user"], r["day"]) for r in current})
    baseline_copies = {(r["user"], r["file"]): r for r in read(OLD / "copy_comparison.json")}
    copies = read(OUT / "copy_comparison.json")
    for r in copies:
        previous = baseline_copies[r["user"], r["file"]]
        r["canonical_changed_since_baseline"] = r["canonical_sha256"] != previous["canonical_sha256"]
        r["experiment_changed_since_baseline"] = r["experiment_sha256"] != previous["experiment_sha256"]
    save("copy_comparison.json", copies)

    # Preserve the old interpretations, not the old probe IDs.
    probes = jsonl(OUT / "probe_coverage.jsonl")
    reviews = read(OLD / "flag_reviews.json")["failure_reviews"]
    evidence_failures, current_reviews = [], []
    for r in reviews:
        path = ROOT / "datasets/v2" / r["user"] / "sessions" / (r["date"] + ".json")
        session = read(path)
        if session["failures"][r["failure_index"]] != r["recorded_failure"]:
            evidence_failures.append({"user": r["user"], "date": r["date"], "kind": "recorded_flag_changed"})
        for ev in r["raw_evidence"]:
            actual = session["turns"][ev["turn_number"] - 1]
            if actual != {"speaker": ev["speaker"], "text": ev["text"]}:
                evidence_failures.append({"user": r["user"], "date": r["date"],
                                          "kind": "quoted_turn_changed", "turn_number": ev["turn_number"]})
        citing = [q for q in probes if q["user"] == r["user"] and r["day"] in q["evidence_days"]]
        row = {k: v for k, v in r.items() if k not in ("cited_by_probe_ids", "proved_wrong_gold_probe_ids")}
        row["baseline_cited_by_probe_ids"] = r["cited_by_probe_ids"]
        row["cited_by_current_probe_ids"] = [q["id"] for q in citing]
        row["current_citation_payloads"] = [
            {k: q[k] for k in ("id", "original_id", "source", "type", "domain", "question", "answer", "probe_date", "evidence_days")}
            for q in citing]
        row["status"] = "unchanged_source_contract_finding" if r["classification"].startswith("confirmed_") else "unchanged_prior_flag_interpretation"
        current_reviews.append(row)

    confirmed = [r for r in current_reviews if r["classification"].startswith("confirmed_")]
    optional = [r for r in confirmed if r["classification"] != "confirmed_required_absent_topic_conflict"]
    unmodeled = [r for r in confirmed if r["classification"] == "confirmed_unmodeled_persistent_detail"]
    union = lambda rows: sorted({qid for r in rows for qid in r["cited_by_current_probe_ids"]})
    source_summary = {
        "baseline_source_sha256": sha(OLD / "audit_source.py"),
        "individual_session_files_checked": len(current), "changed_individual_session_files": session_changes,
        "missing_individual_sessions": missing,
        "baseline_reviewed_flag_occurrences": len(reviews), "quoted_evidence_mismatches": evidence_failures,
        "confirmed_source_contract_occurrences": len(confirmed),
        "confirmed_source_contract_session_count": len({(r["user"], r["day"]) for r in confirmed}),
        "classification_counts": dict(collections.Counter(r["classification"] for r in current_reviews)),
        "all_retained_flags_current_citation_exposure_count": len(union(current_reviews)),
        "confirmed_source_contract_current_citation_exposure_count": len(union(confirmed)),
        "confirmed_source_contract_current_citation_exposure_ids": union(confirmed),
        "avoidable_mention_or_unmodeled_detail_current_citation_exposure_count": len(union(optional)),
        "unmodeled_detail_current_citation_exposure_count": len(union(unmodeled)),
        "source_workstream_wrong_gold_proved_count": 0,
        "limit": "Current citation exposure is not a count of wrong questions. Unchanged source hashes allow reusing earlier focused transcript review; unflagged turn semantics are not exhaustively re-audited.",
    }
    masked = read(OLD / "flag_reviews.json")["masked_absent_topic_value_disagreements"]
    for r in masked:
        r["baseline_cited_by_probe_ids"] = r.pop("cited_by_probe_ids")
        r["cited_by_current_probe_ids"] = [q["id"] for q in probes if q["user"] == r["user"] and r["day"] in q["evidence_days"]]
    save("flag_reviews.json", {"summary": source_summary, "failure_reviews": current_reviews,
                              "masked_absent_topic_value_disagreements": masked})

    coverage, provenance = [], []
    for user in (f"u{i}" for i in range(1, 6)):
        base = ROOT / "datasets/v2" / user
        pool = read(base / "qa_pool.json")
        experiment_pool = {q["id"]: q for q in read(ROOT / "experiments/data/synthetic_v2" / user / "qa_pool.json")}
        canonical_pool = {q["id"]: q for q in pool}
        pq = read(base / "probing_questions.json")
        qs = [q for p in pq["probes"] for q in p["questions"]]
        payload_keys = ("type", "capability", "domain", "question", "answer", "answer_type", "accept", "evidence_days", "grading", "tags")
        mismatches, canonical_matches = [], 0
        for q in qs:
            current_pool_q = canonical_pool.get(q["original_id"], {})
            canonical_matches += all(q[k] == current_pool_q.get(k) for k in payload_keys)
            old_pool_q = experiment_pool.get(q["original_id"], {})
            different = [k for k in payload_keys if q[k] != old_pool_q.get(k)]
            if different:
                mismatches.append({"id": q["id"], "original_id": q["original_id"], "differing_fields": different,
                                   "source": location(base / "probing_questions.json", "id", q["id"]),
                                   "current_question": q["question"], "experiment_pool_question": old_pool_q.get("question")})
        provenance.append({"user": user, "probe_instances": len(qs), "canonical_pool_payload_matches": canonical_matches,
                           "experiment_pool_payload_matches": len(qs) - len(mismatches),
                           "experiment_pool_payload_mismatches": mismatches})
        world = read(base / "world_state.json")
        cutoff = max(p["viewpoint_day"] for p in pq["probes"])
        unknown = [q for q in pool if q["type"] == "abstention" and q["answer_type"] == "abstain"]
        selected_unknown = [q for q in qs if q["type"] == "abstention" and q["answer_type"] == "abstain"]
        selected_control = [q for q in qs if q["type"] == "abstention" and q["answer_type"] != "abstain"]
        tail_sessions = [s for d, s in read(base / "conversations.json")["sessions"].items() if s["day"] > cutoff]
        end = dt.date.fromisoformat(world["start_date"]) + dt.timedelta(days=world["num_days"] - 1)
        predictions = [q for q in pool if q["type"] == "prediction"]
        future_targets = [{"id": q["id"], "target_dates": re.findall(r"\d{4}-\d{2}-\d{2}", q["question"]),
                           "source": location(base / "qa_pool.json", "id", q["id"])} for q in predictions]
        assert all(r["target_dates"] and all(dt.date.fromisoformat(d) > end for d in r["target_dates"]) for r in future_targets)
        pattern_currents = [q for q in pool if q["type"] == "pattern_current"]
        assert all(max(q["evidence_days"]) > cutoff for q in pattern_currents)
        coverage.append({
            "user": user, "pool_type_counts": dict(collections.Counter(q["type"] for q in pool)),
            "probe_type_counts": dict(collections.Counter(q["type"] for q in qs)),
            "pool_unknown_abstention_count": len(unknown), "probe_unknown_abstention_count": len(selected_unknown),
            "probe_answerable_abstention_controls_count": len(selected_control),
            "omitted_unknown_pool_items": [{"id": q["id"], "question": q["question"], "evidence_days": q["evidence_days"],
                                           "source": location(base / "qa_pool.json", "id", q["id"])} for q in unknown],
            "selected_abstention_payloads": [{**q, "source": location(base / "probing_questions.json", "id", q["id"])}
                                               for q in qs if q["type"] == "abstention"],
            "num_probes": len(pq["probes"]), "final_probe_date": pq["probes"][-1]["probe_date"],
            "final_probe_viewpoint_day": cutoff, "last_dataset_date": end.isoformat(),
            "unprobed_world_tail_days": world["num_days"] - cutoff,
            "unprobed_tail_session_count": len(tail_sessions),
            "unprobed_tail_session_dates": [s["date"] for s in tail_sessions],
            "selected_evidence_after_final_probe": [q["id"] for q in qs if any(d > cutoff for d in q["evidence_days"])],
            "prediction_pool_count": sum(q["type"] == "prediction" for q in pool),
            "pattern_current_pool_count": sum(q["type"] == "pattern_current" for q in pool),
            "prediction_probe_count": sum(q["type"] == "prediction" for q in qs),
            "pattern_current_probe_count": sum(q["type"] == "pattern_current" for q in qs),
            "omitted_prediction_pool_targets": future_targets,
            "omitted_pattern_current_pool_items": [{"id": q["id"], "evidence_days": q["evidence_days"],
                                                     "source": location(base / "qa_pool.json", "id", q["id"])} for q in pattern_currents],
        })
    totals = {
        "unknown_abstention_pool_count": sum(r["pool_unknown_abstention_count"] for r in coverage),
        "unknown_abstention_probe_count": sum(r["probe_unknown_abstention_count"] for r in coverage),
        "answerable_abstention_probe_control_count": sum(r["probe_answerable_abstention_controls_count"] for r in coverage),
        "prediction_pool_count": sum(r["prediction_pool_count"] for r in coverage),
        "pattern_current_pool_count": sum(r["pattern_current_pool_count"] for r in coverage),
        "prediction_probe_count": sum(r["prediction_probe_count"] for r in coverage),
        "pattern_current_probe_count": sum(r["pattern_current_probe_count"] for r in coverage),
        "unprobed_tail_session_count": sum(r["unprobed_tail_session_count"] for r in coverage),
    }
    save("coverage.json", {"totals": totals, "users": coverage})
    save("pool_provenance.json", {"users": provenance, "totals": {
        "canonical_pool_payload_matches": sum(r["canonical_pool_payload_matches"] for r in provenance),
        "experiment_pool_payload_matches": sum(r["experiment_pool_payload_matches"] for r in provenance),
        "experiment_pool_payload_mismatches": sum(len(r["experiment_pool_payload_mismatches"]) for r in provenance)}})

    from setup_2.loader import load
    timelines = load(ROOT / "datasets/v2")
    setup_rows = []
    for timeline in timelines:
        source_questions = {q["id"]: q for p in read(ROOT / "datasets/v2" / timeline.id / "probing_questions.json")["probes"] for q in p["questions"]}
        sessions = [s for c in timeline.checkpoints for s in c.sessions]
        exact_questions = all(q.question == source_questions[q.id]["question"] and q.asked_on == source_questions[q.id]["probe_date"]
                              and q.answer == source_questions[q.id]["answer"] and q.accept == source_questions[q.id]["accept"]
                              for c in timeline.checkpoints for q in c.questions)
        cutoffs_valid = all(s.date[:7] == c.month and s.date <= c.cutoff for c in timeline.checkpoints for s in c.sessions)
        qcutoffs_valid = all(q.asked_on <= c.cutoff for c in timeline.checkpoints for q in c.questions)
        setup_rows.append({"user": timeline.id, "checkpoints": len(timeline.checkpoints), "sessions": len(sessions),
                           "session_dates_unique": len({s.date for s in sessions}) == len(sessions),
                           "question_payloads_and_original_dates_preserved": exact_questions,
                           "monthly_session_cutoffs_valid": cutoffs_valid, "question_introduction_cutoffs_valid": qcutoffs_valid,
                           "final_checkpoint_cutoff": timeline.checkpoints[-1].cutoff,
                           "final_partial_month_sessions": len(timeline.checkpoints[-1].sessions),
                           "final_checkpoint_questions": len(timeline.checkpoints[-1].questions),
                           "final_checkpoint_new_questions": len(timeline.checkpoints[-1].new_question_ids)})
        assert exact_questions and cutoffs_valid and qcutoffs_valid
    save("setup_2_load_check.json", {"users": setup_rows, "live_evaluation_run": False,
                                   "scope": "Actual offline loader only; runner code confirms monthly answer/grade barriers. No model instantiated."})
    summary = {"source_integrity": inventory["totals"], "session_comparison": {
        "checked": len(current), "changed": len(session_changes), "missing": len(missing)},
        "copy_comparison": {"compared": len(copies), "byte_equal": sum(r["byte_equal"] for r in copies),
                            "json_equal": sum(r["json_equal"] for r in copies),
                            "changed_canonical_files": [r["canonical"] for r in copies if r["canonical_changed_since_baseline"]]},
        "native_world": read(OUT / "native_world_checks.json")["totals"],
        "native_source_hash_comparison": {"checked": len(native_before), "changes": native_hash_changes},
        "source_contract": source_summary, "coverage": totals}
    save("summary.json", summary)
    print(json.dumps(summary, indent=2))
    assert not session_changes and not missing and not evidence_failures
    assert all(not r["session_projection_world_mismatches"] and not r["top_level_aggregate_world_mismatches"]
               and not r["session_shape_failure_count"] and not r["fidelity_nonusage_mismatches"]
               for r in inventory["users"])
    assert sum(not r["json_equal"] for r in copies) == inventory["totals"]["copy_mismatches"]


if __name__ == "__main__":
    main()
