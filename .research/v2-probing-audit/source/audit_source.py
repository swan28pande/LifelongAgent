#!/usr/bin/env python3
"""Read-only source/integration inventory; writes only beside this audit script.

Run from any directory with the project Python: python <this-file>.
No LLM calls, log regeneration, paid APIs, or network access.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def read(path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON object key {key!r} in {path}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=reject_duplicates)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def jsonl(name, values):
    (OUT / name).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))


def lines_for(path, pattern):
    return [i for i, line in enumerate(path.read_text().splitlines(), 1) if re.search(pattern, line)]


def loc(path, pattern):
    hits = lines_for(path, pattern)
    return {"file": str(path.relative_to(ROOT)), "line": hits[0] if hits else None}


def text_loc(path, text):
    needle = '"text": ' + json.dumps(text, ensure_ascii=False)
    hits = [i for i, line in enumerate(path.read_text().splitlines(), 1) if needle in line]
    return {"file": str(path.relative_to(ROOT)), "line": hits[0] if hits else None}


def main():
    inventory, all_sessions, all_probes, flagged, copies, candidate_turns = [], [], [], [], [], []
    for user in (f"u{i}" for i in range(1, 6)):
        base = ROOT / "datasets/v2" / user
        conv = read(base / "conversations.json")
        world = read(base / "world_state.json")
        fid = read(base / "fidelity.json")
        pq = read(base / "probing_questions.json")
        probes = [q for p in pq["probes"] for q in p["questions"]]
        sessions = {p.stem: (p, read(p)) for p in sorted((base / "sessions").glob("*.json"))}
        expected = {d["date"]: d for d in world["days"] if d["has_session"]}
        mismatches = []
        top_level_mismatches = {k: {"aggregate": conv[k], "world": world[k]} for k in ("user_id", "name") if conv[k] != world[k]}
        for date in set(conv["sessions"]) | set(sessions) | set(expected):
            if date not in conv["sessions"] or date not in sessions or date not in expected:
                mismatches.append({"date": date, "aggregate": date in conv["sessions"],
                                   "individual": date in sessions, "world_has_session": date in expected})
                continue
            path, session = sessions[date]
            projection = {k: session[k] for k in ("day", "date", "weekday", "turns")}
            if conv["sessions"][date] != projection:
                mismatches.append({"date": date, "kind": "aggregate_individual_mismatch"})
            day = expected[date]
            for key in ("day", "date", "weekday"):
                if session[key] != day[key]:
                    mismatches.append({"date": date, "kind": "world_metadata_mismatch", "key": key})
            calculated = dt.date.fromisoformat(world["start_date"]) + dt.timedelta(days=session["day"] - 1)
            if calculated.isoformat() != date or calculated.strftime("%A") != session["weekday"]:
                mismatches.append({"date": date, "kind": "calendar_metadata_mismatch"})
            if session["user_id"] != user:
                mismatches.append({"date": date, "kind": "user_id_mismatch"})
            turns = session["turns"]
            shape = []
            if not 12 <= len(turns) <= 20:
                shape.append("turn_count_outside_12_20")
            if any(t.get("speaker") not in ("user", "assistant") or not t.get("text") for t in turns):
                shape.append("invalid_turn")
            if any(a["speaker"] == b["speaker"] for a, b in zip(turns, turns[1:])):
                shape.append("not_alternating")
            if session["passed_validation"] != (not session["failures"]):
                shape.append("validation_boolean_inconsistent")
            citations = [q["id"] for q in probes if session["day"] in q["evidence_days"]]
            all_sessions.append({"user": user, "day": session["day"], "date": date,
                "session_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source": loc(path, r'^\s*"day":'), "num_turns": len(turns),
                "passed_validation": session["passed_validation"], "failures": session["failures"],
                "first_try_failures": session["first_try_failures"], "attempts": session["attempts"],
                "shape_failures": shape, "cited_by_probe_ids": citations})
            if not session["passed_validation"]:
                flagged.append({"user": user, "day": session["day"], "date": date,
                    "source": loc(path, r'^\s*"passed_validation":'),
                    "failure_source": loc(path, r'^\s*"failures":'),
                    "failures": session["failures"], "first_try_failures": session["first_try_failures"],
                    "extracted": session["extracted"], "attempts": session["attempts"],
                    "world_day_source": loc(base / "world_state.json", rf'^\s*"day": {session["day"]},?$'),
                    "world_preferences": day["preferences"],
                    "required_statements": sum((day.get(k, []) for k in (
                        "facts_to_state_today", "event_updates_today", "other_people_today", "distractors_today")), []),
                    "cause_instructions": day["cause_instructions"], "known_facts": day["known_facts"],
                    "cited_by_probe_ids": citations,
                    "turns": [{**t, "source": text_loc(path, t["text"])} for t in turns]})
            for i, t in enumerate(turns):
                # A candidate only: odd endings and rule language require semantic review.
                reasons = []
                if t["text"].rstrip().endswith((",", "-", ":", ";")):
                    reasons.append("suspicious_terminal_punctuation")
                if re.search(r"\b(every (?:week|day|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
                             r"alternate|rotation|rotate|schedule|routine|always|usually)\b", t["text"], re.I):
                    reasons.append("routine_language")
                if reasons:
                    candidate_turns.append({"user": user, "day": session["day"], "date": date,
                        "turn_index": i, "speaker": t["speaker"], "text": t["text"],
                        "source": text_loc(path, t["text"]), "candidate_reasons": reasons})
        recorded = [s for _, s in sessions.values()]
        derived_fid = {"sessions": len(recorded),
            "passed_first_try": sum(s["passed_validation"] and s["attempts"] == 1 for s in recorded),
            "passed_after_retries": sum(s["passed_validation"] for s in recorded),
            "flagged": sorted(s["date"] for s in recorded if not s["passed_validation"]),
            "first_try_failure_types": dict(collections.Counter(f.split(":")[0] for s in recorded for f in s["first_try_failures"])),
            "remaining_failure_types": dict(collections.Counter(f.split(":")[0] for s in recorded for f in s["failures"])),
            "mean_attempts": round(sum(s["attempts"] for s in recorded) / len(recorded), 2),
            "mean_turns": round(sum(len(s["turns"]) for s in recorded) / len(recorded), 1)}
        fid_diff = {k: {"saved": fid[k], "derived": v} for k, v in derived_fid.items() if fid[k] != v}
        log_path = base / "llm_log.jsonl"
        log_text = log_path.read_text() if log_path.exists() else None
        log_is_pointer = bool(log_text and log_text.startswith("version https://git-lfs.github.com/spec/v1"))
        log_usage = collections.Counter()
        log_records = []
        if log_text and not log_is_pointer:
            for line in log_text.splitlines():
                rec = json.loads(line)
                log_records.append(rec)
                key = rec["model"]
                log_usage[key + ".calls"] += 1
                for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
                    log_usage[key + "." + field] += rec[field]
                log_usage[key + ".errors"] += bool(rec["error"])
        flags_by_day = {s["day"]: s for _, s in sessions.values() if not s["passed_validation"]}
        for q in probes:
            flagged_days = [d for d in q["evidence_days"] if d in flags_by_day]
            unseen_days = [d for d in q["evidence_days"] if not world["days"][d - 1]["has_session"]]
            all_probes.append({"user": user, "id": q["id"], "original_id": q["original_id"],
                "source": loc(base / "probing_questions.json", '"id": "' + re.escape(q["id"]) + '"'),
                "type": q["type"], "domain": q["domain"], "question": q["question"], "answer": q["answer"],
                "probe_date": q["probe_date"], "viewpoint_day": q["viewpoint_day"],
                "evidence_days": q["evidence_days"], "flagged_evidence_days": flagged_days,
                "flagged_failures": {d: flags_by_day[d]["failures"] for d in flagged_days},
                "unsampled_evidence_days": unseen_days,
                "post_cutoff_sessions_if_whole_history_loaded": sum(s["day"] > q["viewpoint_day"] for _, s in sessions.values())})
        inventory.append({"user": user, "aggregate_sessions": len(conv["sessions"]),
            "individual_sessions": len(sessions), "world_num_days": len(world["days"]),
            "world_has_session_true": len(expected), "intentional_unsampled_days": len(world["days"]) - len(expected),
            "session_projection_world_mismatches": sorted(mismatches, key=lambda r: r["date"]),
            "top_level_aggregate_world_mismatches": top_level_mismatches,
            "aggregate_dates_chronological": list(conv["sessions"]) == sorted(conv["sessions"]),
            "session_shape_failure_count": sum(bool(s["shape_failures"]) for s in all_sessions if s["user"] == user),
            "num_probes": len(pq["probes"]), "probing_questions": len(probes),
            "fidelity_derived": derived_fid, "fidelity_nonusage_mismatches": fid_diff,
            "llm_log_is_lfs_pointer": log_is_pointer,
            "llm_log_exists": log_path.exists(),
            "llm_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest() if log_path.exists() else None,
            "llm_log_available_usage_records": len(log_records) if log_text and not log_is_pointer else None,
            "llm_log_record_fields": sorted({k for r in log_records for k in r}),
            "llm_log_usage_equal_to_fidelity": dict(log_usage) == fid["usage"] if log_records else None})
        for filename in ("conversations.json", "probing_questions.json", "world_state.json", "qa_pairs.json", "qa_pool.json", "fidelity.json", "stats.json"):
            a, b = base / filename, ROOT / "experiments/data/synthetic_v2" / user / filename
            copies.append({"user": user, "file": filename, "canonical": str(a.relative_to(ROOT)),
                "experiment": str(b.relative_to(ROOT)), "canonical_sha256": hashlib.sha256(a.read_bytes()).hexdigest(),
                "experiment_sha256": hashlib.sha256(b.read_bytes()).hexdigest(),
                "byte_equal": a.read_bytes() == b.read_bytes(), "json_equal": read(a) == read(b)})
    save("inventory.json", {"users": inventory, "totals": {
        "sessions": len(all_sessions), "probes": len(all_probes), "flagged_sessions": len(flagged),
        "remaining_failure_occurrences": sum(len(s["failures"]) for s in flagged),
        "remaining_failure_types": dict(collections.Counter(f.split(":")[0] for s in flagged for f in s["failures"])),
        "questions_citing_flagged_sessions": sum(bool(q["flagged_evidence_days"]) for q in all_probes),
        "questions_citing_unsampled_evidence_days": sum(bool(q["unsampled_evidence_days"]) for q in all_probes),
        "copy_mismatches": sum(not c["json_equal"] for c in copies),
        "shape_failure_sessions": sum(bool(s["shape_failures"]) for s in all_sessions)}})
    save("copy_comparison.json", copies)
    save("flagged_sessions.json", flagged)
    jsonl("session_coverage.jsonl", all_sessions)
    jsonl("probe_coverage.jsonl", all_probes)
    jsonl("transcript_candidates.jsonl", candidate_turns)
    print(json.dumps(read(OUT / "inventory.json")["totals"], indent=2))


if __name__ == "__main__":
    main()
