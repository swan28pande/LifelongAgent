"""Independent, local read-only cross-check; outputs only verification.json here."""

from __future__ import annotations

import calendar
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / ".research/v2-probing-audit"
CURRENT = ROOT / ".research/v2-probing-recheck"
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def rule_value(rule, day, date, anchor):
    """Small independent interpreter for only the counterexamples' rule types."""
    kind = rule["type"]
    if kind == "constant":
        return rule["value"]
    if kind == "day_of_week":
        return rule["map"][date.strftime("%a").lower()]
    if kind == "weekly_alternate":
        return rule["values"][((day - anchor) // 7) % len(rule["values"])]
    if kind == "n_day_cycle":
        offset = (day - anchor) % sum(s["days"] for s in rule["segments"])
        for segment in rule["segments"]:
            if offset < segment["days"]:
                return segment["value"]
            offset -= segment["days"]
    if kind == "nested":
        child = rule["blocks"][((day - anchor) // rule["block_days"]) % len(rule["blocks"])]
        return rule_value(child, day, date, anchor)
    if kind == "conditional":
        assert rule["condition"] == "is_workday", rule
        result = rule["cases"]["true" if date.weekday() < 5 else "false"]
        return rule_value(result, day, date, anchor) if isinstance(result, dict) else result
    raise ValueError(rule)


def main():
    result = {"scope": "Fresh raw-input checks plus independent counterexample arithmetic; saved structural-ledger count is independently recounted, not a second semantic proof.", "failures": []}
    fail = result["failures"]
    users = [f"u{i}" for i in range(1, 6)]
    probes = {u: read(f"datasets/v2/{u}/probing_questions.json") for u in users}
    worlds = {u: read(f"datasets/v2/{u}/world_state.json") for u in users}
    convs = {u: read(f"datasets/v2/{u}/conversations.json") for u in users}
    pools = {u: {q["id"]: q for q in read(f"datasets/v2/{u}/qa_pool.json")} for u in users}
    rows = [(u, p, q) for u in users for p in probes[u]["probes"] for q in p["questions"]]
    by_id = {q["id"]: (u, p, q) for u, p, q in rows}
    structural = read(".research/v2-probing-recheck/structure_checks.json")
    result["structural_ledger"] = {"checks": len(structural), "failed": sum(not r["passed"] for r in structural), "current_questions": len(rows), "unique_ids": len(by_id)}
    if len(rows) != 1000 or len(by_id) != 1000 or len(structural) != 13484 or result["structural_ledger"]["failed"]:
        fail.append("structural count mismatch")
    result["types"] = dict(collections.Counter(q["type"] for _, _, q in rows))
    raw_invariant_errors, future = [], []
    for u, p, q in rows:
        start = dt.date.fromisoformat(probes[u]["start_date"])
        date = dt.date.fromisoformat(p["probe_date"])
        day = (date - start).days + 1
        sampled = {(dt.date.fromisoformat(d) - start).days + 1 for d in convs[u]["sessions"]}
        if date.day != calendar.monthrange(date.year, date.month)[1] or q["viewpoint_day"] != day or q["probe_date"] != p["probe_date"]:
            raw_invariant_errors.append(q["id"])
        if any(e > day or e not in sampled for e in q["evidence_days"]):
            raw_invariant_errors.append(q["id"])
        for field in ("question", "answer", "accept"):
            text = json.dumps(q[field]) if field == "accept" else q[field]
            targets = [(s, dt.date.fromisoformat(s)) for s in re.findall(r"\b20\d\d-\d\d-\d\d\b", text)]
            for s in re.findall(r"\b[A-Za-z]+\s+20\d\d\b", text):
                try:
                    d = dt.datetime.strptime(s, "%B %Y").date()
                except ValueError:
                    continue
                targets.append((s, d.replace(day=calendar.monthrange(d.year, d.month)[1])))
            future.extend({"id": q["id"], "field": field, "target": s} for s, d in targets if d > date)
    result["fresh_raw_invariants"] = {"errors": raw_invariant_errors, "future_dates_three_fields": future}
    if raw_invariant_errors or future:
        fail.append("raw metadata/cutoff error")

    from generator_v2.probing import build_probing_questions
    result["builder_reproduction"] = {u: build_probing_questions(u) == probes[u] for u in users}
    if not all(result["builder_reproduction"].values()):
        fail.append("builder mismatch")
    payload_fields = ("type", "capability", "domain", "question", "answer", "answer_type", "accept", "evidence_days", "grading", "tags")
    canonical_mismatch, stale_mismatch = [], []
    stale = {u: {q["id"]: q for q in read(f"experiments/data/synthetic_v2/{u}/qa_pool.json")} for u in users}
    for u, _, q in rows:
        original = pools[u].get(q["original_id"], {})
        old = stale[u].get(q["original_id"], {})
        if any(q.get(k) != original.get(k) for k in payload_fields):
            canonical_mismatch.append(q["id"])
        if any(q.get(k) != old.get(k) for k in payload_fields):
            stale_mismatch.append(q["id"])
    result["pool_payloads"] = {"canonical_matches": len(rows)-len(canonical_mismatch), "canonical_mismatches": canonical_mismatch, "stale_experiment_mismatch_count": len(stale_mismatch), "stale_experiment_mismatch_by_user": dict(collections.Counter(by_id[q][0] for q in stale_mismatch))}
    if canonical_mismatch or len(stale_mismatch) != 786:
        fail.append("pool payload mismatch count")
    pairs = read(".research/v2-probing-recheck/source/copy_comparison.json")
    actual_pairs = [{"canonical": r["canonical"], "experiment": r["experiment"], "byte_equal": (ROOT/r["canonical"]).read_bytes() == (ROOT/r["experiment"]).read_bytes()} for r in pairs]
    result["copy_pairs"] = {"checked": len(actual_pairs), "equal": sum(r["byte_equal"] for r in actual_pairs), "mismatches": [r for r in actual_pairs if not r["byte_equal"]]}

    source_rows = [json.loads(s) for s in (BASE/"source/session_coverage.jsonl").read_text().splitlines()]
    changed, projections = [], []
    for r in source_rows:
        filename = r["source"]["file"]
        if sha(filename) != r["session_file_sha256"]:
            changed.append(filename)
        s = read(filename)
        agg = convs[r["user"]]["sessions"][r["date"]]
        if any(s[k] != v for k, v in agg.items()):
            projections.append(filename)
    result["source_identity"] = {"checked": len(source_rows), "changed": changed, "aggregate_projection_mismatches": projections}
    if changed or projections or len(source_rows) != 2738:
        fail.append("source identity")
    flag_reviews = read(".research/v2-probing-recheck/source/flag_reviews.json")["failure_reviews"]
    confirmed = [r for r in flag_reviews if r["classification"].startswith("confirmed_")]
    sessions = {(r["user"], r["day"]) for r in confirmed}
    exposures = {q["id"] for u, _, q in rows if any((u, d) in sessions for d in q["evidence_days"])}
    quotes_missing = []
    for r in flag_reviews:
        for e in r.get("raw_evidence", []):
            s = read(e["source"]["file"])
            if not any(t["text"] == e["text"] for t in s["turns"]):
                quotes_missing.append(e["source"])
    result["retained_source_contract"] = {"confirmed_occurrences": len(confirmed), "confirmed_sessions": len(sessions), "current_citation_exposure_count": len(exposures), "current_citation_exposure_ids": sorted(exposures), "quotation_mismatches": quotes_missing, "wrong_gold_inference": "No wrong selected answer inferred from citation exposure."}
    if len(confirmed) != 19 or len(exposures) != 38 or quotes_missing:
        fail.append("retained source count/quote mismatch")

    unknown_pool = [q for u in users for q in pools[u].values() if q["type"] == "abstention" and q["answer_type"] == "abstain"]
    unknown_selected = [q["id"] for _, _, q in rows if q["type"] == "abstention" and q["answer_type"] == "abstain"]
    result["coverage"] = {"unknown_pool": len(unknown_pool), "unknown_selected": unknown_selected, "answerable_abstention_controls": sum(q["type"] == "abstention" and q["answer_type"] != "abstain" for _, _, q in rows), "prediction_selected": sum(q["type"] == "prediction" for _, _, q in rows), "current_pattern_selected": sum(q["type"] == "pattern_current" for _, _, q in rows)}
    if len(unknown_pool) != 30 or unknown_selected:
        fail.append("unknown coverage count")

    from setup_2.loader import load
    timelines = load()
    schedule = []
    for t in timelines:
        seen = [s.date for c in t.checkpoints for s in c.sessions]
        late = [s.date for c in t.checkpoints for s in c.sessions if s.date > c.cutoff or s.date[:7] != c.month]
        schedule.append({"user": t.id, "checkpoints": len(t.checkpoints), "sessions": len(seen), "all_sessions_once": len(seen) == len(set(seen)) == len(convs[t.id]["sessions"]), "late_sessions": late, "final_cutoff": t.checkpoints[-1].cutoff, "final_questions": len(t.checkpoints[-1].questions), "final_new_questions": len(t.checkpoints[-1].new_question_ids), "input_hash_files": sorted(t.input_hashes)})
    result["setup_2_schedule"] = {"users": schedule, "total_sessions": sum(r["sessions"] for r in schedule), "repeated_answer_instances": sum(len(c.questions) for t in timelines for c in t.checkpoints), "network_or_model_run": False}
    if any(r["checkpoints"] != 24 or not r["all_sessions_once"] or r["late_sessions"] for r in schedule):
        fail.append("setup schedule")

    hobby_ids = ["u5_p202701_0069", "u5_p202701_0073"]
    hobby = []
    for qid in hobby_ids:
        u, p, q = by_id[qid]
        truth = worlds[u]["days"][q["viewpoint_day"]-1]["active_facts"]["hobbies"]
        hobby.append({"id": qid, "question": q["question"], "answer": q["answer"], "accepted": q["accept"], "expected": sorted(truth), "omitted": sorted(set(truth)-set(s.strip() for s in q["answer"].split(",")))})
    result["wrong_current_hobbies"] = hobby
    if any(r["omitted"] != ["piano"] for r in hobby):
        fail.append("hobby witness")
    removal = next(s for s in worlds["u5"]["statements"] if s["id"] == "hobby.piano_stop")
    result["piano_removal"] = removal
    result["january_piano_turns"] = [{"date": d, "text": t["text"]} for d, s in convs["u5"]["sessions"].items() if d[:7] == "2027-01" for t in s["turns"] if t["speaker"] == "user" and "piano" in t["text"].lower()]

    recall_truth_mismatches = []
    for u, _, q in rows:
        if q["type"] != "recall":
            continue
        target = dt.date.fromisoformat(re.search(r"20\d\d-\d\d-\d\d", q["question"]).group())
        day = (target - dt.date.fromisoformat(worlds[u]["start_date"])).days + 1
        truth = worlds[u]["days"][day-1]["preferences"][q["domain"]]["value"]
        if q["answer"] != truth or q["accept"] != [truth]:
            recall_truth_mismatches.append(q["id"])
    result["recall_raw_hidden_truth"] = {"checked": sum(q["type"] == "recall" for _, _, q in rows), "mismatches": recall_truth_mismatches}
    u, _, phase_q = by_id["u5_p202605_0008"]
    phase_days = [d["day"] for d in worlds[u]["days"] if d["day"] <= 97 and d["preferences"]["exercise"]["base_value"] == "bike ride"]
    phase_mentions = [d for d in phase_days if worlds[u]["days"][d-1]["preferences"]["exercise"]["mentioned"]]
    result["retained_unseen_phase"] = {"question_id": phase_q["id"], "bike_phase_days_whole_regime": phase_days, "bike_phase_observations": phase_mentions, "evidence_days": phase_q["evidence_days"], "cited_values": [worlds[u]["days"][d-1]["preferences"]["exercise"]["value"] for d in phase_q["evidence_days"]]}

    daniel_old = next(r for r in worlds["u4"]["regimes"] if r["id"] == "music.r3")
    daniel_new = next(r for r in worlds["u4"]["regimes"] if r["id"] == "music.r4")
    daniel_q = by_id["u4_p202712_0152"][2]
    daniel_alternatives = []
    observations = [d for d in worlds["u4"]["days"] if daniel_old["start"] <= d["day"] <= daniel_q["viewpoint_day"] and d["preferences"]["music"]["mentioned"]]
    for start in range(658, 662):
        mismatches = []
        for d in observations:
            r = daniel_old if d["day"] < start else daniel_new
            anchor = r["anchor"] if r == daniel_old else start
            if rule_value(r["rule"], d["day"], dt.date.fromisoformat(d["date"]), anchor) != d["preferences"]["music"]["base_value"]:
                mismatches.append(d["day"])
        target_regime = daniel_old if 659 < start else daniel_new
        anchor = target_regime["anchor"] if target_regime == daniel_old else start
        daniel_alternatives.append({"start_day": start, "start_date": worlds["u4"]["days"][start-1]["date"], "target_value": rule_value(target_regime["rule"], 659, dt.date(2027,12,19), anchor), "mismatch_days": mismatches})
    result["daniel_boundary"] = {"id": daniel_q["id"], "gold": daniel_q["answer"], "observed_domain_choices": len(observations), "counterfactuals": daniel_alternatives, "target_session_omits_modeled_music": not worlds["u4"]["days"][658]["preferences"]["music"]["mentioned"], "cause": daniel_new["cause_day"]}

    fact_q = by_id["u5_p202706_0105"][2]
    support = [t["text"] for t in convs["u5"]["sessions"]["2027-06-19"]["turns"] if t["speaker"] == "user" and "vegetarian" in t["text"]]
    result["diet_citation_witness"] = {"id": fact_q["id"], "reference": fact_q["answer"], "evidence": fact_q["evidence_days"], "evidence_day_1_truth": worlds["u5"]["days"][0]["active_facts"]["diet"], "target_truth": worlds["u5"]["days"][471]["active_facts"]["diet"], "supporting_date": "2027-06-19", "dated_switch_turns": support, "full_prefix_has_support": bool(support) and "2027-06-19" <= fact_q["probe_date"]}

    import yaml
    ladder = yaml.safe_load((ROOT/"generator_v2/personas/difficulty_ladder.yaml").read_text())["users"]
    candidates = read(".research/v2-probing-recheck/gold/broader_boundary_candidates.json")
    reviews = {r["id"]: r for r in read(".research/v2-probing-recheck/gold/broader_boundary_reviews.json")}
    checked = []
    for row in candidates:
        world = worlds[row["user_id"]]
        regime = next(r for r in world["regimes"] if r["id"] == row["regime_id"])
        previous = max((r for r in world["regimes"] if r["domain"] == row["domain"] and r["end"] < regime["start"]), key=lambda r:r["end"])
        cutoff_day = by_id[row["id"]][2]["viewpoint_day"]
        obs = [d for d in world["days"] if previous["start"] <= d["day"] <= min(cutoff_day,regime["end"]) and d["preferences"][row["domain"]]["mentioned"]]
        alternatives = []
        for alt in row["rejected_compatible_alternatives"]:
            newrule = dict(regime["rule"])
            if alt["variant"] == "rotated_weekly_value_order":
                newrule["values"] = newrule["values"][1:] + newrule["values"][:1]
            mismatches = []
            for d in obs:
                rule = previous["rule"] if d["day"] < alt["day"] else newrule
                anchor = previous["anchor"] if d["day"] < alt["day"] else alt["anchor"]
                if rule_value(rule, d["day"], dt.date.fromisoformat(d["date"]), anchor) != d["preferences"][row["domain"]]["base_value"]:
                    mismatches.append(d["day"])
            lag_valid = True
            if regime["cause_day"] is not None:
                low, high = ladder[row["user_id"]]["lag_days"]
                lag_valid = low <= alt["day"]-regime["cause_day"] <= high
            alternatives.append({"date":alt["date"], "value_mismatch_days":mismatches, "before_or_at_probe":alt["day"]<=cutoff_day, "lag_valid":lag_valid, "outside_accept":alt["date"] not in row["accepted"], "reversion_anchor_valid":regime["kind"] != "reversion" or alt["anchor"] == regime["anchor"]})
        checked.append({"id":row["id"], "all_declared_old_new_choices_checked":len(obs), "manual_status":reviews[row["id"]]["status"], "alternatives":alternatives})
    result["broader_counterexample_arithmetic"] = {"candidate_instances":len(checked), "status_counts":dict(collections.Counter(r["manual_status"] for r in checked)), "checks":checked, "manual_near_boundary_text_is_separate":True}
    if any(a["value_mismatch_days"] or not a["before_or_at_probe"] or not a["lag_valid"] or not a["outside_accept"] or not a["reversion_anchor_valid"] for r in checked for a in r["alternatives"]):
        fail.append("broader counterexample arithmetic/constraint failure")
    narrow = {r["id"] for r in read(".research/v2-probing-recheck/gold/observation_limits.json")}
    credible = {r["id"] for r in checked if r["manual_status"] == "credible_bounded_observability_limit"}
    result["date_limit_overlap"] = {"narrow_count":len(narrow), "credible_broader_count":len(credible), "narrow_inside_broader":sorted(narrow & credible), "narrow_outside_broader":sorted(narrow-credible), "union_count":len(narrow|credible)}
    narrow_cutoff_errors = []
    for r in read(".research/v2-probing-recheck/gold/observation_limits.json"):
        for a in r["expected"]["conversation_compatible_rejected_starts"]:
            if a["start_day"] > r["viewpoint_day"]:
                narrow_cutoff_errors.append({"id":r["id"], "alternative":a})
    root_workout_witness = next(r["detail"]["expected"]["conversation_compatible_rejected_starts"] for r in read(".research/v2-probing-recheck/findings.json") if r["id"] == "u5_p202701_0068" and r["category"] == "non_unique_change_date_rejected_by_grader")
    result["corrected_narrow_witness_cutoffs"] = {"post_probe_alternatives":narrow_cutoff_errors, "root_january_workout_witnesses":root_workout_witness}
    if narrow_cutoff_errors or [r["start_date"] for r in root_workout_witness] != ["2027-01-28"]:
        fail.append("corrected narrow witness cutoff")
    excerpt_errors, excerpts_checked = [], 0
    for row in candidates:
        for turn in row["prefix_excerpt_turns"]:
            src = turn["source"]
            value = read(src["file"])
            for component in src["pointer"].strip("/").split("/"):
                component = component.replace("~1", "/").replace("~0", "~")
                value = value[int(component)] if isinstance(value, list) else value[component]
            excerpts_checked += 1
            if value != turn["text"] or turn["date"] > row["probe_date"]:
                excerpt_errors.append({"id":row["id"], "source":src})
    result["broader_actual_prefix_excerpts"] = {"checked":excerpts_checked, "source_pointer_or_cutoff_errors":excerpt_errors}
    if excerpt_errors:
        fail.append("broader raw-prefix source/cutoff error")

    root_coverage = [json.loads(s) for s in (CURRENT/"question_coverage.jsonl").read_text().splitlines()]
    ledger_error = []
    for r in root_coverage:
        if r["id"] not in by_id or any(r[k] != by_id[r["id"]][2][k] for k in ("question","original_id","type","probe_date")):
            ledger_error.append(r["id"])
    classes = collections.defaultdict(set)
    for r in root_coverage:
        for c in r["classes"]:
            classes[c].add(r["id"])
    union = set().union(*classes.values())
    current_refs = set(hobby_ids)
    cite_ids = {r["id"] for r in read(".research/v2-probing-recheck/gold/issues.json") if r["kind"] == "evidence_gap"}
    cite_ids |= {r["id"] for r in read(".research/v2-probing-recheck/gold/supplemental_issues.json")}
    recall_ids = {r["question_id"] for r in read(".research/v2-probing-recheck/recall/issues.json")}
    uncertain_ids = {r["id"] for r in read(".research/v2-probing-recheck/gold/uncertainties.json")}
    narrow_union = current_refs | cite_ids | narrow | recall_ids | uncertain_ids
    expanded_union = narrow_union | credible
    result["consolidated_counts"] = {"coverage_rows":len(root_coverage), "unique_ids":len({r["id"] for r in root_coverage}), "payload_mismatches":ledger_error, "classes":{k:len(v) for k,v in classes.items()}, "reference_error_ids_match":classes["confirmed_reference_error"] == current_refs, "citation_ids_match":classes["citation_limit"] == cite_ids, "observation_ids_match":classes["observation_limit"] == credible | recall_ids, "uncertainty_ids_match":classes["uncertainty"] == uncertain_ids, "reference_truth_matches":sum(r["reference_matches_synthetic_truth"] for r in root_coverage), "baseline_style_union":len(narrow_union), "additional_broader":len(credible-narrow_union), "expanded_union":len(expanded_union), "ledger_union_matches":union == expanded_union}
    if ledger_error or len(root_coverage)!=1000 or len(narrow_union)!=15 or len(expanded_union)!=29 or union!=expanded_union:
        fail.append("consolidated ledger mismatch")
    manifest = read(".research/v2-probing-recheck/manifest.json")
    changed_inputs = [filename for filename, record in manifest.items() if sha(filename) != record["sha256"]]
    result["current_input_stability"] = {"checked":len(manifest), "changed":changed_inputs}
    if changed_inputs:
        fail.append("current input changed")
    previous_qs = {}
    baseline_manifest = read(".research/v2-probing-audit/manifest.json")
    baseline_blobs_match = {}
    for u in users:
        filename = f"datasets/v2/{u}/probing_questions.json"
        blob = subprocess.run(["git","show",f"98a2674:{filename}"],cwd=ROOT,capture_output=True,check=True).stdout
        baseline_blobs_match[u] = hashlib.sha256(blob).hexdigest() == baseline_manifest[filename]["sha256"]
        previous_qs.update({q["id"]:(u,q) for p in json.loads(blob)["probes"] for q in p["questions"]})
    finding_matches = read(".research/v2-probing-recheck/prior_finding_matches.json")
    prior_match_errors, statuses = [], collections.Counter()
    for r in finding_matches:
        u, old_q = previous_qs[r["baseline_id"]]
        def same(q):
            return all(q[k] == old_q[k] for k in ("question","type","domain"))
        selected = [q for uid,_,q in rows if uid == u and same(q)]
        pool_matches = [q for q in pools[u].values() if same(q)]
        status = "still_selected" if selected else "pool_only" if pool_matches else "absent_from_current_pool"
        statuses[status] += 1
        if status != r["selection_status"] or {q["id"] for q in selected} != {q["id"] for q in r["current_matches"]}:
            prior_match_errors.append(r["baseline_id"])
    result["prior_semantic_match_check"] = {"baseline_blobs_match_audit_hashes":baseline_blobs_match, "checked_findings":len(finding_matches), "statuses":dict(statuses), "mismatches":prior_match_errors, "matching_key":"user plus question text/type/domain; compare answer/evidence after matching; IDs are not semantic keys"}
    if prior_match_errors or not all(baseline_blobs_match.values()):
        fail.append("prior matching disagreement")
    (OUT/"verification.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k in ("failures","structural_ledger","builder_reproduction","pool_payloads","copy_pairs","source_identity","coverage","date_limit_overlap")}, indent=2))


if __name__ == "__main__":
    main()
