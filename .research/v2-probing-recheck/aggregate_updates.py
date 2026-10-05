"""Consolidate final offline evidence without editing any project inputs."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(name):
    return json.loads((OUT / name).read_text())


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main():
    current = {q["id"]: q for q in read("current_questions.json")}
    findings = []

    def add(row, kind, artifact, expanded=False):
        ident = row.get("id", row.get("question_id"))
        q = current[ident]
        findings.append({"id": ident, "user": q["user_id"], "type": q["type"],
                         "question": q["question"], "probe_date": q["probe_date"],
                         "answer": q["answer"], "file": q["file"], "line": q["line"],
                         "category": row["category"], "class": kind,
                         "scope": "expanded_change_boundary_review" if expanded else "baseline_comparable_checks",
                         "evidence_artifact": artifact, "detail": row})

    for name in ("gold/issues.json", "gold/supplemental_issues.json"):
        for row in read(name):
            add(row, "confirmed_reference_error" if row["kind"] == "reference_error" else "citation_limit", name)
    for row in read("gold/observation_limits.json"):
        add(row, "observation_limit", "gold/observation_limits.json")
    for row in read("gold/uncertainties.json"):
        add(row, "uncertainty", "gold/uncertainties.json")
    for row in read("recall/issues.json"):
        add(row, "observation_limit", "recall/issues.json")

    narrow_dates = {r["id"] for r in read("gold/observation_limits.json")}
    for row in read("gold/broader_boundary_reviews.json"):
        if row["status"] != "credible_bounded_observability_limit" or row["id"] in narrow_dates:
            continue
        add(dict(row, category="non_unique_change_date_rejected_by_grader"), "observation_limit",
            "gold/broader_boundary_reviews.json", expanded=True)
    write("findings.json", findings)

    gold = {r["id"]: r for r in read("gold/coverage.json")}
    recall = {r["question_id"]: r for r in map(json.loads, (OUT / "recall/coverage.jsonl").read_text().splitlines())}
    assert not (gold.keys() & recall.keys())
    assert gold.keys() | recall.keys() == current.keys()
    by_id = collections.defaultdict(list)
    for finding in findings:
        by_id[finding["id"]].append(finding)
    coverage = []
    for ident, q in current.items():
        truth_matches = (recall[ident]["checks"]["gold_equals_world_truth"] if ident in recall
                         else gold[ident]["gold_matches_semantic_truth"])
        coverage.append({"id": ident, "original_id": q["original_id"], "user": q["user_id"],
                         "type": q["type"], "question": q["question"], "probe_date": q["probe_date"],
                         "reference_matches_synthetic_truth": truth_matches,
                         "classes": sorted({r["class"] for r in by_id[ident]}),
                         "categories": sorted({r["category"] for r in by_id[ident]}),
                         "file": q["file"], "line": q["line"]})
    (OUT / "question_coverage.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in coverage))

    narrow = [r for r in findings if r["scope"] == "baseline_comparable_checks"]
    class_ids = {kind: sorted({r["id"] for r in findings if r["class"] == kind})
                 for kind in {r["class"] for r in findings}}
    errors = set(class_ids["confirmed_reference_error"])
    limits = set(class_ids["citation_limit"]) | set(class_ids["observation_limit"])
    summary = {"current_questions": len(current), "recall": len(recall), "non_recall": len(gold),
               "structural_checks": read("structure_summary.json")["checks"],
               "structural_failures": read("structure_summary.json")["failed_checks"],
               "explicit_future_targets": len(read("calendar_issues.json")),
               "future_dates_in_any_question_reference_or_acceptance": len(read("all_fields_future_dates.json")),
               "reference_matches_synthetic_truth": sum(r["reference_matches_synthetic_truth"] for r in coverage),
               "unique_ids_by_class": class_ids,
               "confirmed_reference_error_instances": len(errors),
               "distinct_observation_or_citation_limit_instances": len(limits - errors),
               "duration_completion_uncertainties": len(class_ids["uncertainty"]),
               "baseline_comparable_narrow_finding_union": len({r["id"] for r in narrow}),
               "additional_expanded_boundary_instances": len({r["id"] for r in findings if r["scope"] != "baseline_comparable_checks"}),
               "expanded_finding_union": len({r["id"] for r in findings}),
               "distinct_instances_by_category": {c: len({r["id"] for r in findings if r["category"] == c})
                                                   for c in sorted({r["category"] for r in findings})},
               "source": read("source/summary.json"),
               "selection_match_summary": read("comparison_summary.json"),
               "monthly_loader": {k: v for k, v in read("loader_schedule.json").items() if k != "users"},
               "limits": ["Expanded date-boundary checks are broader than the original audit; counts are not a pure before/after defect comparison.",
                          "Observation/citation limits and source exposures are not incorrect synthetic reference answers.",
                          "This is offline verification, not an agent evaluation or fresh paid LLM validation."]}
    write("summary.json", summary)

    def table(ids):
        rows = ["| Current question | Original probe | Finding |", "| --- | --- | --- |"]
        for ident in sorted(ids):
            q = current[ident]
            categories = sorted({r["category"] for r in by_id[ident]})
            link = f"[{ident}]({ROOT / q['file']}:{q['line']})"
            rows.append(f"| {link} | {q['probe_date']} | {', '.join(categories)} |")
        return "\n".join(rows)

    narrow_ids = {r["id"] for r in narrow}
    expanded_ids = {r["id"] for r in findings} - narrow_ids
    text = ["# Current question findings", "",
            "All locations refer to the updated canonical probing files. Repeated evaluation attempts are not counted separately here; dataset retrospective records retain their own IDs.", "",
            "## Incorrect references", "", table(errors), "",
            "Both references omit the still-active piano hobby at January 31, 2027. Their answer and acceptance errors are counted once per question.", "",
            "## Specific observation and citation limits", "", table(narrow_ids & limits), "",
            "Six change-date rows also have citation limits; that overlap is counted once. Priya's diet reference is correct and supported elsewhere in the original prefix; its listed citation is wrong. Recall limits agree with hidden truth but isolate missing/ambiguous observations.", "",
            "## Duration uncertainties", "", table(set(class_ids["uncertainty"])), "",
            "The completed framing is not established by the available closure observations. These are uncertainties, not confirmed wrong totals.", "",
            "## Additional expanded date-boundary findings", "", table(expanded_ids), "",
            "These extend the original first-choice check. The 20 credible bounded date findings include six rows already listed above and the 14 additional rows here. Three contested candidates are excluded. Detailed alternative dates, source excerpts, constraints, and interpretation limits are in gold/broader_boundary_reviews.json.", ""]
    (OUT / "question_findings.md").write_text("\n".join(text))

    stability = [{"file": name, "unchanged_during_recheck": hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == entry["sha256"]}
                 for name, entry in read("manifest.json").items()]
    assert all(r["unchanged_during_recheck"] for r in stability)
    write("input_stability.json", stability)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"source", "unique_ids_by_class", "selection_match_summary", "limits"}}, indent=2))


if __name__ == "__main__":
    main()
