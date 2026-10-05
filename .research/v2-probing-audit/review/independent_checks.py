"""Read-only independent checks; writes only to this review directory."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
AUDIT = ROOT / ".research/v2-probing-audit"
OUT = AUDIT / "review"


def load(path):
    return json.loads(Path(path).read_text())


def raw_eval(rule, day, date, anchor):
    """Independent dictionary-based evaluator for the rule types in these inputs."""
    offset = day - anchor
    assert offset >= 0
    kind = rule["type"]
    if kind == "constant":
        return rule["value"]
    if kind == "weekly_alternate":
        return rule["values"][(offset // 7) % len(rule["values"])]
    if kind == "n_day_cycle":
        pos = offset % sum(s["days"] for s in rule["segments"])
        for seg in rule["segments"]:
            if pos < seg["days"]:
                return seg["value"]
            pos -= seg["days"]
    if kind == "day_of_week":
        return rule["map"][date.strftime("%a").lower()]
    if kind == "nested":
        block = rule["blocks"][(offset // rule["block_days"]) % len(rule["blocks"])]
        return raw_eval(block, day, date, anchor)
    if kind == "conditional":
        if rule["condition"] == "is_workday":
            key = "true" if date.weekday() < 5 else "false"
        elif rule["condition"] == "season":
            key = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}[date.month]
        else:
            raise AssertionError("Unexpected condition: " + rule["condition"])
        case = rule["cases"].get(key, rule["cases"].get("default"))
        return case if isinstance(case, str) else raw_eval(case, day, date, anchor)
    raise AssertionError(kind)


def main():
    questions, worlds, conversations = {}, {}, {}
    all_inputs = {}
    for uid in ("u1", "u2", "u3", "u4", "u5"):
        base = ROOT / "datasets/v2" / uid
        probing = load(base / "probing_questions.json")
        worlds[uid] = load(base / "world_state.json")
        conversations[uid] = load(base / "conversations.json")["sessions"]
        pool = {q["id"]: q for q in load(base / "qa_pool.json")}
        for probe in probing["probes"]:
            for question in probe["questions"]:
                q = {**question, "user_id": uid}
                assert q["id"] not in questions
                questions[q["id"]] = q
                original = pool[q["original_id"]]
                assert all(q[k] == original[k] for k in ("type", "capability", "domain", "question", "answer", "answer_type", "accept", "evidence_days", "grading", "tags"))
                assert q["viewpoint_day"] == probe["viewpoint_day"]
                assert q["probe_date"] == probe["probe_date"]
                assert max(q["evidence_days"]) <= q["viewpoint_day"]
        for filename in ("conversations.json", "probing_questions.json", "world_state.json", "qa_pairs.json", "qa_pool.json", "fidelity.json", "stats.json"):
            left = base / filename
            right = ROOT / "experiments/data/synthetic_v2" / uid / filename
            all_inputs[str(left.relative_to(ROOT))] = hashlib.sha256(left.read_bytes()).hexdigest()
            assert left.read_bytes() == right.read_bytes()

    recall_rows = [json.loads(line) for line in (AUDIT / "recall/coverage.jsonl").read_text().splitlines()]
    gold_rows = load(AUDIT / "gold/coverage.json")
    recall_ids = [r["question_id"] for r in recall_rows]
    gold_ids = [r["id"] for r in gold_rows]
    assert len(recall_ids) == len(set(recall_ids)) == 489
    assert len(gold_ids) == len(set(gold_ids)) == 511
    assert not set(recall_ids) & set(gold_ids)
    assert set(recall_ids) | set(gold_ids) == set(questions)
    assert all(questions[qid]["type"] == "recall" for qid in recall_ids)
    assert all(questions[qid]["type"] != "recall" for qid in gold_ids)

    manifest = load(AUDIT / "manifest.json")
    source_hash_mismatches = [p for p, record in manifest.items()
                              if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != record["sha256"]]
    assert not source_hash_mismatches

    cells, base_mismatches, recall_mismatches = 0, [], []
    for uid, world in worlds.items():
        regimes = {r["id"]: r for r in world["regimes"]}
        for day in world["days"]:
            for domain, pref in day["preferences"].items():
                r = regimes[pref["regime_id"]]
                value = raw_eval(r["rule"], day["day"], dt.date.fromisoformat(day["date"]), r["anchor"])
                cells += 1
                if value != pref["base_value"]:
                    base_mismatches.append([uid, day["day"], domain, value, pref["base_value"]])
        for q in questions.values():
            if q["user_id"] != uid or q["type"] != "recall":
                continue
            targets = re.findall(r"\d{4}-\d{2}-\d{2}", q["question"])
            assert len(targets) == 1
            daynum = (dt.date.fromisoformat(targets[0]) - dt.date.fromisoformat(world["start_date"])).days + 1
            truth = world["days"][daynum - 1]["preferences"][q["domain"]]["value"]
            if q["answer"] != truth or q["accept"] != [truth]:
                recall_mismatches.append(q["id"])
    assert not base_mismatches
    assert not recall_mismatches

    fact_mismatches, stale_review = [], None
    for q in questions.values():
        if q["type"] not in ("fact_current", "fact_at_time"):
            continue
        world = worlds[q["user_id"]]
        target = q["viewpoint_day"] if q["type"] == "fact_current" else (
            dt.date.fromisoformat(re.search(r"\d{4}-\d{2}-\d{2}", q["question"]).group()) - dt.date.fromisoformat(world["start_date"])).days + 1
        vals = world["days"][target - 1]["active_facts"][q["domain"]]
        expected = ", ".join(vals) if vals else "none"
        if q["answer"] != expected:
            fact_mismatches.append({"id": q["id"], "reference": q["answer"], "expected": expected})
        if q["id"] == "u5_p202701_0076":
            stale_review = {"id": q["id"], "active": vals,
                            "known": world["days"][target - 1]["known_facts"]["hobbies"],
                            "hobby_statements": [s for s in world["statements"] if s.get("entity") == "hobbies"]}
    assert [x["id"] for x in fact_mismatches] == ["u5_p202701_0076"]

    temporal_rows = []
    for qid in ("u4_p202706_0100", "u5_p202608_0032"):
        q = questions[qid]
        numbers = re.findall(r"\d{4}-\d{2}-\d{2}", q["answer"])
        start, end = map(dt.date.fromisoformat, numbers)
        probe = dt.date.fromisoformat(q["probe_date"])
        completed, elapsed = (end - start).days + 1, (probe - start).days + 1
        assert end > probe
        temporal_rows.append({"id": qid, "start": start.isoformat(), "end": end.isoformat(),
                              "completed": completed, "elapsed": elapsed,
                              "elapsed_relative_error": abs(completed - elapsed) / completed,
                              "within_judge_approx_ten_percent": abs(completed - elapsed) / completed < 0.1})

    counterfactuals = []
    specs = [("u3_p202711_0137", 603, [603, 604], None),
             ("u5_p202610_0047", 201, [200, 201, 202, 203], None),
             ("u4_p202710_0129", None, [603, 604, 605], None),
             ("u5_p202702_0080", None, list(range(345, 351)), None),
             ("u5_r202708_0197", None, list(range(345, 351)), None),
             ("u5_p202609_0043", None, [190, 197], {"type": "weekly_alternate", "values": ["poetry collections", "library thrillers"]}),
             ("u5_r202611_0189", None, [190, 197], {"type": "weekly_alternate", "values": ["poetry collections", "library thrillers"]})]
    for qid, target, starts, alt_rule in specs:
        q = questions[qid]
        world = worlds[q["user_id"]]
        if target is None:
            row = next(x for x in gold_rows if x["id"] == qid)
            regime_id = row["regime_id"]
            reg = next(r for r in world["regimes"] if r["id"] == regime_id)
        else:
            reg = next(r for r in world["regimes"] if r["id"] == world["days"][target - 1]["preferences"][q["domain"]]["regime_id"])
        domain_regs = [r for r in world["regimes"] if r["domain"] == q["domain"]]
        prev = domain_regs[domain_regs.index(reg) - 1]
        observed = [d for d in world["days"] if prev["start"] <= d["day"] <= min(reg["end"], q["viewpoint_day"])
                    and d["preferences"][q["domain"]]["mentioned"]]
        for start in starts:
            candidate_rule = alt_rule if alt_rule and start != reg["start"] else reg["rule"]
            mismatches = []
            for d in observed:
                pred = raw_eval(prev["rule"], d["day"], dt.date.fromisoformat(d["date"]), prev["anchor"]) if d["day"] < start else raw_eval(candidate_rule, d["day"], dt.date.fromisoformat(d["date"]), start)
                if pred != d["preferences"][q["domain"]]["base_value"]:
                    mismatches.append(d["day"])
            target_answer = None
            if target:
                date = dt.date.fromisoformat(world["days"][target - 1]["date"])
                target_answer = raw_eval(prev["rule"], target, date, prev["anchor"]) if target < start else raw_eval(candidate_rule, target, date, start)
            counterfactuals.append({"id": qid, "candidate_start": start,
                                    "candidate_date": world["days"][start - 1]["date"],
                                    "observations_checked": len(observed), "mismatch_days": mismatches,
                                    "target_answer": target_answer,
                                    "date_in_current_accept": world["days"][start - 1]["date"] in q["accept"],
                                    "changed_rule_order": candidate_rule != reg["rule"],
                                    "candidate_lag": start - reg["cause_day"] if reg["cause_day"] else None})

    q = questions["u5_p202605_0011"]
    world = worlds["u5"]
    reg = next(r for r in world["regimes"] if r["id"] == "exercise.r0")
    variant = copy.deepcopy(reg["rule"])
    variant["blocks"][1]["map"]["sat"] = "long run"
    observations = [d for d in world["days"][:min(reg["end"], q["viewpoint_day"])] if d["preferences"]["exercise"]["mentioned"]]
    phase_counterfactual = {
        "id": q["id"], "gold": q["answer"], "alternative": "long run",
        "observations_checked": len(observations),
        "mismatch_days": [d["day"] for d in observations if raw_eval(variant, d["day"], dt.date.fromisoformat(d["date"]), reg["anchor"]) != d["preferences"]["exercise"]["base_value"]],
        "affected_phase_days": [d["day"] for d in world["days"][:reg["end"]] if d["day"] % 14 == 0],
        "scope": "Modeled exercise observations, challenged separately with transcript attribution review."
    }
    assert not phase_counterfactual["mismatch_days"]

    pet_reviews = []
    pet_pattern = re.compile(r"\b(?:pets?|dogs?|cats?|puppy|puppies|kittens?|adopt(?:ed|ing)?|olive|milo|dash|animal(?:s)?|furry|canine|feline|companion)\b", re.I)
    for qid in ("u2_p202603_0001", "u2_p202608_0038", "u2_r202702_0188", "u3_p202604_0007"):
        q = questions[qid]
        hits = []
        for date, session in conversations[q["user_id"]].items():
            if date > q["probe_date"]:
                continue
            for turn_num, turn in enumerate(session["turns"], 1):
                if pet_pattern.search(turn["text"]):
                    hits.append({"date": date, "turn": turn_num, **turn})
        pet_reviews.append({"id": qid, "gold": q["answer"], "evidence": q["evidence_days"],
                            "user_hit_count": sum(h["speaker"] == "user" for h in hits),
                            "all_hits": hits,
                            "all_pet_statements": [s for s in worlds[q["user_id"]]["statements"] if s.get("entity") == "pets"],
                            "limitation": "Keyword checks plus review of hits and day-1/adoption text; no closed-world guarantee is in the assistant task."})

    flags = load(AUDIT / "source/flag_reviews.json")
    reviewed = flags["failure_reviews"]
    confirmed = [r for r in reviewed if r["classification"].startswith("confirmed_")]
    eligible = [r for r in confirmed if r["classification"] != "confirmed_required_absent_topic_conflict"]
    raw_probe_ids = {qid for r in reviewed for qid in r["cited_by_probe_ids"]}
    confirmed_probe_ids = {qid for r in confirmed for qid in r["cited_by_probe_ids"]}
    flagged_summary = {"failure_classification_counts": dict(Counter(r["classification"] for r in reviewed)),
                       "reviewed_failures": len(reviewed), "reviewed_sessions": len({(r["user"], r["day"]) for r in reviewed}),
                       "raw_flag_probe_exposure": len(raw_probe_ids),
                       "confirmed_source_contract_sessions": len({(r["user"], r["day"]) for r in confirmed}),
                       "confirmed_source_contract_probe_exposure": len(confirmed_probe_ids),
                       "avoidable_or_unmodeled_sessions": len({(r["user"], r["day"]) for r in eligible}),
                       "avoidable_or_unmodeled_probe_exposure": len({qid for r in eligible for qid in r["cited_by_probe_ids"]})}
    assert flagged_summary["raw_flag_probe_exposure"] == 54
    assert flagged_summary["confirmed_source_contract_sessions"] == 19
    assert flagged_summary["confirmed_source_contract_probe_exposure"] == 33

    calendar = load(AUDIT / "calendar_issues.json")
    gold_issues = load(AUDIT / "gold/issues.json")
    recall_issues = load(AUDIT / "recall/issues.json")
    categories = {
        "explicit_future_target": {r["id"] for r in calendar},
        "wrong_current_gold": {"u5_p202701_0076"},
        "future_gold_endpoint": {r["id"] for r in gold_issues if r["category"] == "future_completed_duration_in_gold"},
        "recall_answerability": {r["question_id"] for r in recall_issues if r["category"] in ("unobserved_shift_boundary", "unobserved_answer_phase")},
        "unsupported_empty_pet": {r["id"] for r in pet_reviews},
        "underinclusive_change_dates": {"u4_p202710_0129", "u5_p202609_0043", "u5_r202611_0189", "u5_p202702_0080", "u5_r202708_0197"},
        "citation_only": {r["id"] for r in gold_issues if r["category"] == "confounder_change_not_cited"},
        "closure_uncertainty": {r["id"] for r in load(AUDIT / "gold/uncertainties.json")},
    }
    pairs = []
    names = list(categories)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = categories[left] & categories[right]
            if overlap:
                pairs.append({"left": left, "right": right, "ids": sorted(overlap)})
    subset_gold = {r["id"] for r in gold_issues if r["category"] == "explicit_future_target"}
    subset_recall = {r["question_id"] for r in recall_issues if r["category"] == "future_target_in_recall"}
    assert subset_gold | subset_recall == categories["explicit_future_target"]
    assert not subset_gold & subset_recall

    final_findings = load(AUDIT / "findings.json")
    final_summary = load(AUDIT / "summary.json")
    final_coverage = [json.loads(line) for line in (AUDIT / "question_coverage.jsonl").read_text().splitlines()]
    final_coverage_key = "question_id" if "question_id" in final_coverage[0] else "id"
    assert len(final_coverage) == len({r[final_coverage_key] for r in final_coverage}) == 1000
    assert {r[final_coverage_key] for r in final_coverage} == set(questions)
    final_class_ids = {kind: {r["question_id"] for r in final_findings if r["class"] == kind}
                       for kind in ("confirmed_error", "observation_limit", "citation_limit", "uncertainty")}
    assert final_class_ids["confirmed_error"] == categories["explicit_future_target"] | categories["wrong_current_gold"] | categories["future_gold_endpoint"]
    assert final_class_ids["observation_limit"] == categories["recall_answerability"] | categories["unsupported_empty_pet"] | categories["underinclusive_change_dates"]
    assert final_class_ids["citation_limit"] == categories["citation_only"] | categories["underinclusive_change_dates"]
    assert final_class_ids["uncertainty"] == categories["closure_uncertainty"]
    assert set().union(*final_class_ids.values()) == set().union(*categories.values())
    assert {k: len(v) for k, v in final_class_ids.items()} == final_summary["distinct_questions_by_class"]
    assert len(set().union(*final_class_ids.values())) == final_summary["all_narrow_question_finding_union"] == 71
    final_counts = {"distinct_per_class": {k: len(v) for k, v in final_class_ids.items()},
                    "observation_citation_overlap": sorted(final_class_ids["observation_limit"] & final_class_ids["citation_limit"]),
                    "additional_observation_or_citation_union": len(final_class_ids["observation_limit"] | final_class_ids["citation_limit"]),
                    "narrow_union": 71, "aggregate_coverage_ids_verified": 1000}

    from experiments.benchmarks import v2
    from experiments.core.grading import strict_check
    from experiments.core.types import Question
    loaded = v2.load()
    date_grading = []
    for r in counterfactuals:
        if r["target_answer"] is None and not r["mismatch_days"]:
            q = questions[r["id"]]
            obj = Question(id=q["id"], question=q["question"], answer=q["answer"], accept=q["accept"],
                           answer_type=q["answer_type"], grading=q["grading"], category=q["type"])
            date_grading.append({"id": r["id"], "candidate_date": r["candidate_date"],
                                 "strict_result": strict_check(obj, r["candidate_date"]),
                                 "observations_compatible": True})

    result = {
        "scope": "Independent read-only review of canonical input facts and workstream artifacts; no network or LLM calls.",
        "coverage": {"canonical_ids": len(questions), "recall_ids": len(recall_ids), "other_ids": len(gold_ids),
                     "missing_ids": [], "duplicate_ids": [], "cross_workstream_overlap": [],
                     "per_user_counts": dict(Counter(q["user_id"] for q in questions.values())),
                     "per_type_counts": dict(Counter(q["type"] for q in questions.values()))},
        "manifest_hash_mismatches": source_hash_mismatches,
        "compared_identical_json_files": len(all_inputs),
        "independent_base_cells_checked": cells,
        "base_mismatches": base_mismatches,
        "recall_gold_mismatches": recall_mismatches,
        "fact_reference_mismatches": fact_mismatches,
        "stale_hobby_review": stale_review,
        "future_gold_durations": temporal_rows,
        "counterfactuals": counterfactuals,
        "date_grading": date_grading,
        "unobserved_phase_counterfactual": phase_counterfactual,
        "empty_pet_reviews": pet_reviews,
        "saved_flag_review_counts": flagged_summary,
        "loader": {"instances": len(loaded), "questions": sum(len(i.questions) for i in loaded),
                   "monthly_ids_loaded": sorted({q.id for i in loaded for q in i.questions} & set(questions))},
        "llm_log_formats": {uid: "lfs_pointer" if (ROOT / "datasets/v2" / uid / "llm_log.jsonl").read_text().startswith("version https://git-lfs.github.com/spec/v1") else "actual_jsonl" for uid in worlds},
        "final_aggregate_reconciliation": final_counts,
        "candidate_consolidated_counts": {"per_category": {k: len(v) for k, v in categories.items()},
                                           "union": len(set().union(*categories.values())), "overlaps": pairs,
                                           "explicit_future_partition": {"recall": len(subset_recall), "other": len(subset_gold)},
                                           "note": "Categories are separate; no exhaustive natural-language correctness claim."},
        "limits": ["Dictionary rule checks validate modeled truth, not observed entailment.",
                   "Counterfactual numeric checks cover modeled base observations and preserve stored one-off exceptions; transcript attribution/timing must be reviewed separately.",
                   "Provenance equality does not prove truth at a changed viewpoint.",
                   "Flag citation exposure does not establish a wrong question.",
                   "The normal runner does not currently load monthly questions; code and offline reproduction show a potential monthly leakage path, not an actual completed leaky monthly run."]
    }
    (OUT / "independent_checks.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("coverage", "manifest_hash_mismatches", "compared_identical_json_files", "independent_base_cells_checked", "fact_reference_mismatches", "candidate_consolidated_counts", "saved_flag_review_counts", "loader", "llm_log_formats")}, indent=2))


if __name__ == "__main__":
    main()
