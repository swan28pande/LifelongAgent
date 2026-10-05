#!/usr/bin/env python3
"""Separate evidence metadata findings and manually reviewed boundary candidates."""
from __future__ import annotations

import collections
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ns = runpy.run_path(str(ROOT / ".research/v2-probing-audit/gold/audit_gold.py"), run_name="metadata_import")
ns["Source"].__init__.__globals__["ROOT"] = ROOT
Source = ns["Source"]


# These annotations follow direct reads of the saved full first/cause/nearby
# user turns. They do not assert exhaustive entailment of every later sentence.
BOUNDARY_NOTES = {
    ("u3", "lunch.r1"): ("credible_bounded_observability_limit", "2026-05-30", "May29 starts baking and still packs leftovers. May31 gives no lunch choice. June1 reports sourdough sandwich, without announcing a dated routine switch. May30/31 alternatives keep all declared lunch choices and permitted cause lags."),
    ("u3", "commute.r3"): ("credible_bounded_observability_limit", "2027-11-30", "Dec1 reports spraining ankle yesterday and taking a Lyft today. There is no Nov30 commute report or dated announcement of starting rideshare. Same-day injury-to-change lag0 is permitted by U3; Nov30 remains a compatible boundary."),
    ("u4", "workout.r3"): ("credible_bounded_observability_limit", "2027-01-25", "Jan24 sells bicycle and gives up cycling; Jan26 and Jan28 report bouldering, Jan31 yoga. These are shared old/new weekday choices. No dated new workout announcement appears. Earlier or Jan29 boundary respects cause lags and choices."),
    ("u4", "breakfast.r3"): ("credible_bounded_observability_limit", "2027-06-02", "June1 reports smoothie bowl; June2 has no session; June3 reports bagel with salmon. June3 does not announce starting a new routine. Starting June2 preserves all declared breakfast choices."),
    ("u4", "music.r2.revert"): ("contested_condition_link", None, "July2 explicitly reports testing negative that morning and complete recovery, then return to design work and alternative music. Earlier music boundaries preserve choices, but moving an illness-linked reversion before stated recovery relies on separating listening choice from the specified temporary condition. Keep outside counted credible limits."),
    ("u4", "music.r3"): ("credible_bounded_observability_limit", "2027-10-24", "Oct26 mentions jazz vinyl without dated switch announcement; Oct27 first reports ambient electronic. Oct24/25 respect job-change lag and all declared music observations. This is the prior narrow case."),
    ("u5", "bedtime.r1"): ("credible_bounded_observability_limit", "2026-06-07", "June6 reports 1am weekend bedtime; June7 has no bedtime report; June8 plans 10:30pm without declaring first use. June7 fits dog-adoption lag and all declared bedtime choices."),
    ("u5", "music.r1"): ("credible_bounded_observability_limit", "2026-06-25", "June22/24 report lo-fi, June26 film scores; no dated switch announcement. Rotating weekly order with a June25/26 start preserves all declared choices, including the later continuing cycle."),
    ("u5", "dinner.r1"): ("credible_bounded_observability_limit", "2026-07-29", "July28 starts dating Arjun but eats lentil soup separately. August2 reports leftovers and August5 stir-fry, shared old/new choices. July29 onward candidates after relationship change preserve every declared dinner value; no dated dinner-routine announcement."),
    ("u5", "music.r2"): ("credible_bounded_observability_limit", "2026-09-13", "Sept12 dates concert yesterday but still listens to lo-fi; Sept19 first reports synthwave. No Sept13-18 music choice is stated or exact switch date announced. Sept13/14/15 respect concert cause lag and full declared choices."),
    ("u5", "bedtime.r2"): ("contested_decision_wording", None, "Nov17 says they chatted about sleep goals today and 'I've decided' on eleven. A Nov14 start matches modeled bedtime values but this decision wording can be read as pinning adoption to Nov17, which is already accepted. Do not count rejected Nov14 as a credible full-text counterexample."),
    ("u5", "exercise.r4"): ("credible_bounded_observability_limit", "2027-01-28", "Jan27/31 report morning dog walks shared by old/new rules; taking up trail running as a hobby is dated separately from the weekday workout schedule. Jan28 is pre-probe, satisfies lag4 and all declared choices. Future February alternatives in the baseline narrow algorithm are explicitly not needed for this finding."),
    ("u5", "music.r3"): ("credible_bounded_observability_limit", "2027-02-09", "Feb8 reports synthwave without announcing switch; Feb14 reports running playlists. Feb9-13 preserve all choices and allowed marathon-related cause lags. Prior narrow primary/retrospective case."),
    ("u5", "commute.r3"): ("credible_bounded_observability_limit", "2027-03-26", "March25/31 report light rail, shared old/new weekday choices. March28 unusual Sunday office trip mentions planned bike but usual no-commute exception is unchanged in both worlds. April6 first work-at-home choice does not announce a start date. March26 onward shifts preserve declared choices; no dated schedule transition is given."),
    ("u5", "reading.r4"): ("credible_bounded_observability_limit", "2027-05-27", "May26 reads book club paperback; May27-29 have no reading choice; May30 listens to narrated novel without first-use declaration. May27/28 remain compatible boundaries."),
    ("u5", "commute.r4"): ("credible_bounded_observability_limit", "2027-08-06", "Aug2 job starts, Aug4 train, Aug5 work at home, Aug9 bike. Aug6 has no commute report. Aug6 satisfies cause lag4 and choices, without conflict with any near-boundary dated transition statement."),
    ("u5", "bedtime.r3"): ("credible_bounded_observability_limit", "2027-08-03", "Aug2 reports eleven, Aug3-4 no bedtime choice, Aug9 reports ten. The new management job is dated Aug2; Aug3/4 bedtime changes respect cause lags and all declared choices. Aug9 does not date first adoption."),
    ("u5", "exercise.r5"): ("credible_bounded_observability_limit", "2027-09-07", "Sept5 dates hiking-group membership yesterday, Sept6 reports trail run, Sept11 first group hike without first-hike announcement. Sept7-10 starts respect cause lag and all declared choices; no earlier hike/routine-change date is stated."),
}


def main():
    coverage = json.loads((OUT / "coverage.json").read_text())
    findings, fallback_checks = [], []
    for uid in ("u1", "u2", "u3", "u4", "u5"):
        world = Source(ROOT / f"datasets/v2/{uid}/world_state.json")
        conversations = Source(ROOT / f"datasets/v2/{uid}/conversations.json")
        for row in coverage:
            if row["user_id"] != uid or row["type"] != "fact_at_time":
                continue
            for value in row["expected_values"]:
                sources = [(i, s) for i, s in enumerate(world.data["statements"]) if s["entity"] == row["domain"] and s["value"] == value and s["kind"] in ("background_fact", "fact_change") and s["effective_day"] <= row["target_day"]]
                cited = [s for _, s in sources if s["stated_day"] in row["evidence_days"]]
                check = {"id": row["id"], "value": value, "evidence_days": row["evidence_days"], "cited_statement_support": bool(cited),
                         "actual_statement_available_by_probe": any(s["stated_day"] <= row["viewpoint_day"] for _, s in sources),
                         "actual_statement_sources": [{"id": s["id"], "effective_day": s["effective_day"], "stated_day": s["stated_day"], "source": world.loc(f"/statements/{i}")} for i, s in sources]}
                fallback_checks.append(check)
                if row["evidence_days"] != [1] or cited:
                    continue
                # Current-only case: correct answer is expressly supported by
                # retrospective June19 source, not by its cited March1 fallback.
                supporting_turns = [{"date": date, "text": t["text"], "source": conversations.loc(f"/sessions/{date}/turns/{i}/text")}
                                    for date, session in conversations.data["sessions"].items()
                                    for i, t in enumerate(session["turns"]) if t["speaker"] == "user" and date == "2027-06-19" and "June 13th" in t["text"]]
                findings.append({"id": row["id"], "user_id": uid, "original_id": row["original_id"], "type": row["type"],
                                 "category": "fact_fallback_citation_does_not_support_value", "kind": "evidence_gap",
                                 "question": row["question"], "source": row["source"], "probe_date": row["probe_date"],
                                 "observed": {"answer": row["observed_answer"], "evidence_days": row["evidence_days"]},
                                 "expected": {"supporting_statement_day": 476, "supporting_turns": supporting_turns},
                                 "synthetic_reference_correct": True, "full_prefix_answerable": True,
                                 "rationale": "Filtering statement evidence to the historical target removes the delayed retrospective declaration. The [1] fallback cites pescatarian, not vegetarian. June19 is available at the June30 probe and explicitly dates the vegetarian switch to June13."})
    (OUT / "fact_evidence_support_checks.json").write_text(json.dumps(fallback_checks, indent=2) + "\n")
    (OUT / "supplemental_issues.json").write_text(json.dumps(findings, indent=2, ensure_ascii=False) + "\n")

    candidates = json.loads((OUT / "broader_boundary_candidates.json").read_text())
    reviews = []
    for row in candidates:
        status, witness, rationale = BOUNDARY_NOTES[row["user_id"], row["regime_id"]]
        assert witness is None or witness in {a["date"] for a in row["rejected_compatible_alternatives"]}
        reviews.append({"id": row["id"], "question_source": row["question_source"], "status": status,
                        "example_rejected_boundary": witness, "accepted": row["accepted"], "probe_date": row["probe_date"],
                        "rationale": rationale, "primary_evidence": "broader_boundary_candidates.json contains full source-indexed user-turn extracts and complete declared-choice comparison counts.",
                        "scope_limit": "Manual review of near-boundary full text plus declared-choice counterexample; no exhaustive natural-language entailment proof or paid judge result."})
    (OUT / "broader_boundary_reviews.json").write_text(json.dumps(reviews, indent=2) + "\n")
    summary = json.loads((OUT / "broader_boundary_summary.json").read_text())
    summary.update(manual_review_status="complete_bounded_review", review_counts=dict(collections.Counter(r["status"] for r in reviews)),
                   credible_bounded_limit_ids=[r["id"] for r in reviews if r["status"] == "credible_bounded_observability_limit"],
                   excluded_contested_ids=[r["id"] for r in reviews if r["status"] != "credible_bounded_observability_limit"],
                   scope="Reviewed witnesses predate or equal the original probe. This broader check is separate from the baseline narrow-first-value criteria and should not be presented as 20 incorrect synthetic gold answers.")
    (OUT / "broader_boundary_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"fact_values_checked": len(fallback_checks), "fact_values_with_actual_source_after_probe": sum(not c["actual_statement_available_by_probe"] for c in fallback_checks),
                      "fallback_issues": len(findings), "broader_boundary_review": summary["review_counts"]}, indent=2))


if __name__ == "__main__":
    main()
