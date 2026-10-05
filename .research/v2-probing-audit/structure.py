"""Read-only structural and explicit-calendar audit of canonical v2 probes."""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
os.environ["TMPDIR"] = str(PROJECT / "tmp")
sys.path.insert(0, str(PROJECT))
from generator_v2.probing import build_probing_questions  # noqa: E402


def month_ends(start: dt.date, days: int) -> list[dt.date]:
    end = start + dt.timedelta(days=days - 1)
    current = start
    points = []
    while current <= end:
        following = (current.replace(year=current.year + 1, month=1, day=1)
                     if current.month == 12 else current.replace(month=current.month + 1, day=1))
        boundary = following - dt.timedelta(days=1)
        if boundary <= end:
            points.append(boundary)
        current = following
    return points


def date_targets(question: str) -> list[tuple[str, dt.date]]:
    targets = [(value, dt.date.fromisoformat(value))
               for value in re.findall(r"\b20\d\d-\d\d-\d\d\b", question)]
    for value in re.findall(r"\b[A-Za-z]+\s+20\d\d\b", question):
        try:
            targets.append((value, dt.datetime.strptime(value, "%B %Y").date()))
        except ValueError:
            pass
    return targets


def locations(text: str) -> dict[str, dict[str, int]]:
    found = {}
    current = None
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r'\s*"id":\s*"([^"]+)"', line)
        if match:
            current = match.group(1)
            found[current] = {"id": number}
        elif current:
            key = re.match(r'\s*"([^"]+)":', line)
            if key and key.group(1) in {"question", "answer", "accept", "evidence_days", "viewpoint_day"}:
                found[current][key.group(1)] = number
    return found


def main() -> None:
    checks, issues, calendar, manifest, users, selection = [], [], [], {}, {}, []
    identifiers = set()
    required = {"id", "original_id", "type", "capability", "domain", "question", "answer",
                "answer_type", "accept", "evidence_days", "grading", "viewpoint_day",
                "probe_date", "probe_month", "is_retrospective", "tags"}
    for uid in ("u1", "u2", "u3", "u4", "u5"):
        root = PROJECT / "datasets/v2" / uid
        path = root / "probing_questions.json"
        raw = path.read_text()
        data = json.loads(raw)
        world = json.loads((root / "world_state.json").read_text())
        pool = json.loads((root / "qa_pool.json").read_text())
        pool_by_id = {q["id"]: q for q in pool}
        legacy = json.loads((root / "qa_pairs.json").read_text())
        lines = locations(raw)
        start = dt.date.fromisoformat(world["start_date"])
        end = start + dt.timedelta(days=world["num_days"] - 1)
        flat = [q for p in data["probes"] for q in p["questions"]]
        by_id = {q["id"]: q for q in flat}

        def check(name: str, passed: bool, q: dict | None = None, detail: object = None) -> None:
            row = {"user": uid, "check": name, "passed": passed,
                   "question_id": q["id"] if q else None, "detail": detail}
            checks.append(row)
            if not passed:
                issues.append(row)

        for name in ("probing_questions.json", "qa_pairs.json", "qa_pool.json",
                     "world_state.json", "conversations.json", "fidelity.json"):
            source = root / name
            manifest[str(source.relative_to(PROJECT))] = {
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": source.stat().st_size,
            }
        check("identity", data["user_id"] == world["user_id"] == uid
              and data["name"] == world["name"])
        check("timeline_metadata", data["start_date"] == world["start_date"]
              and data["num_days"] == world["num_days"])
        check("question_count", len(flat) == data["total_questions"] == 200)
        check("probe_count", len(data["probes"]) == data["num_probes"] == 23)
        actual = [dt.date.fromisoformat(p["probe_date"]) for p in data["probes"]]
        check("month_end_schedule", actual == month_ends(start, world["num_days"]))
        check("generator_reproducible", data == build_probing_questions(uid))
        for p in data["probes"]:
            cutoff = dt.date.fromisoformat(p["probe_date"])
            check("probe_day_matches_date", p["viewpoint_day"] == (cutoff - start).days + 1)
            check("probe_month_matches_date", p["probe_month"] == cutoff.strftime("%Y-%m"))
            check("probe_size", p["num_questions"] == len(p["questions"]))
            check("probe_retro_count", p["num_retrospective"] == sum(q["is_retrospective"] for q in p["questions"]))
            for q in p["questions"]:
                check("required_fields", required <= q.keys(), q)
                check("id_format", bool(re.fullmatch(uid + r"_[pr]\d{6}_\d{4}", q["id"])), q)
                check("retrospective_boolean", isinstance(q["is_retrospective"], bool), q)
                check("unique_id", q["id"] not in identifiers, q)
                identifiers.add(q["id"])
                check("parent_viewpoint", q["viewpoint_day"] == p["viewpoint_day"]
                      and q["probe_date"] == p["probe_date"] and q["probe_month"] == p["probe_month"], q)
                check("nonempty_question_answer", isinstance(q["question"], str) and bool(q["question"].strip())
                      and isinstance(q["answer"], str) and bool(q["answer"].strip()), q)
                check("valid_answer_type", q["answer_type"] in {"value", "date", "yes_no", "free_text", "abstain"}, q)
                check("valid_grading", q["grading"] in {"exact", "date_exact", "llm_judge"}, q)
                check("accepted_values", isinstance(q["accept"], list) and bool(q["accept"])
                      and all(isinstance(v, str) and bool(v.strip()) for v in q["accept"]), q)
                if q["answer_type"] == "yes_no":
                    check("yes_no_accept", all(v.lower().strip() in {"yes", "no"} for v in q["accept"]), q)
                if q["grading"] == "date_exact":
                    valid_dates = True
                    for value in q["accept"]:
                        try:
                            valid_dates &= dt.date.fromisoformat(value) <= cutoff
                        except ValueError:
                            valid_dates = False
                    check("date_accept_format_and_cutoff", valid_dates, q)
                check("evidence_range", isinstance(q["evidence_days"], list) and all(
                    isinstance(day, int) and 1 <= day <= q["viewpoint_day"] for day in q["evidence_days"]), q)
                check("pool_source_exists", q["original_id"] in pool_by_id, q)
                source = pool_by_id.get(q["original_id"])
                if source:
                    fields = ("question", "answer", "accept", "evidence_days", "type", "capability", "domain", "grading", "answer_type", "tags")
                    check("pool_payload_preserved", all(q[k] == source[k] for k in fields), q)
                if q["is_retrospective"]:
                    origin = by_id.get(q.get("retrospective_of"))
                    check("retrospective_origin_exists", bool(origin), q)
                    if origin:
                        offset = (cutoff.year - int(origin["probe_month"][:4])) * 12 + cutoff.month - int(origin["probe_month"][5:])
                        check("retrospective_offset", 2 <= offset <= 6, q, offset)
                        check("retrospective_same_source", not origin["is_retrospective"]
                              and q["original_id"] == origin["original_id"]
                              and q["original_probe_month"] == origin["probe_month"], q)
                        check("retrospective_payload", all(q[k] == origin[k] for k in
                              ("question", "answer", "accept", "evidence_days", "type", "domain", "grading", "answer_type")), q)
                future = [(label, target) for label, target in date_targets(q["question"]) if target > cutoff]
                if future:
                    calendar.append({"user": uid, "id": q["id"], "type": q["type"],
                                     "question": q["question"], "probe_date": p["probe_date"],
                                     "target_dates": [label for label, _ in future],
                                     "category": "explicit_future_target", "confidence": "confirmed",
                                     "file": str(path.relative_to(PROJECT)), "line": lines[q["id"]]["question"],
                                     "observed": p["probe_date"],
                                     "expected": "A historical target must not be later than the probe cutoff."})
        source_types = collections.Counter(q["type"] for q in pool)
        types = collections.Counter(q["type"] for q in flat)
        abstention = [q for q in pool if q["type"] == "abstention"]
        selection.append({"user": uid, "pool_types": dict(source_types), "probe_types": dict(types),
                          "pool_abstention_unanswerable": sum(q["answer_type"] == "abstain" for q in abstention),
                          "probe_abstention_unanswerable": sum(q["type"] == "abstention" and q["answer_type"] == "abstain" for q in flat),
                          "excluded_empty_evidence_pool": sum(not q["evidence_days"] for q in pool),
                          "unused_tail_days": (end - actual[-1]).days,
                          "timeline_end": end.isoformat(), "last_probe": actual[-1].isoformat()})
        users[uid] = {"questions": len(flat), "primary": sum(not q["is_retrospective"] for q in flat),
                      "retrospective": sum(q["is_retrospective"] for q in flat), "probes": len(actual),
                      "first_probe": actual[0].isoformat(), "last_probe": actual[-1].isoformat(),
                      "legacy_questions": len(legacy), "calendar_errors": sum(q["user"] == uid for q in calendar)}
    for name in ("generator_v2/probing.py", "generator_v2/qa.py", "generator_v2/rules.py",
                 "generator_v2/schema.py", "generator_v2/simulator.py", "generator_v2/validator.py",
                 "generator_v2/DECISIONS.md", "experiments/benchmarks/v2.py", "experiments/core/runner.py",
                 "experiments/core/grading.py"):
        source = PROJECT / name
        manifest[name] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size}
    outputs = {"manifest.json": manifest, "structure_checks.json": checks, "structure_issues.json": issues,
               "calendar_issues.json": calendar, "selection.json": selection,
               "structure_summary.json": {"users": users, "checks": len(checks), "failed_checks": len(issues),
                                          "calendar_issues": len(calendar), "unique_question_ids": len(identifiers)}}
    for name, value in outputs.items():
        (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(outputs["structure_summary.json"], indent=2))


if __name__ == "__main__":
    main()
