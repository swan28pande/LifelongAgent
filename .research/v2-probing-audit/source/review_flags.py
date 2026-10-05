#!/usr/bin/env python3
"""Materialize explicitly reviewed transcript judgments, separate from recorded flags.

Annotations below are human/agent interpretation of verbatim local evidence. They
are not a new LLM extraction or proof that cited probing questions have wrong golds.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]

# (user, date, failure ordinal) -> classification, verbatim turn numbers, reasoning.
REVIEW = {
    ("u2", "2026-06-29", 0): ("refuted_by_world_and_prior_transcript", [8],
        "Joining the club is existing world statement hobby.tennis, effective/stated day 120 (Sunday June 28); the prior day's user also explicitly says 'I joined a tennis club today!'. 'This weekend' on Monday June 29 is consistent. The validator's known input retains only hobbies: tennis, losing fact-change narrative history."),
    ("u2", "2026-09-01", 0): ("confirmed_avoidable_absent_topic_mention", [5, 6], "Workout is marked unmentioned; assistant asks about tennis and user discusses recent court time. Historical/negative discussion still violates the absent-topic rule."),
    ("u2", "2026-09-11", 0): ("confirmed_avoidable_absent_topic_mention", [5, 6], "Workout is marked unmentioned; assistant and user discuss possible tennis this weekend. A future plan still mentions the topic."),
    ("u2", "2026-11-01", 0): ("confirmed_avoidable_absent_topic_mention", [3, 4], "Workout is marked unmentioned; assistant asks about tennis and user says no tennis today."),
    ("u2", "2027-07-25", 0): ("confirmed_avoidable_absent_topic_mention", [5, 6], "Workout is marked unmentioned; tennis last Thursday is nevertheless discussed."),
    ("u2", "2027-09-06", 0): ("confirmed_avoidable_absent_topic_mention", [3, 4], "Workout is marked unmentioned; tennis on the weekend is nevertheless discussed."),
    ("u2", "2027-11-15", 0): ("confirmed_avoidable_absent_topic_mention", [3, 4], "Workout is marked unmentioned; Saturday court activity is nevertheless discussed."),
    ("u2", "2027-12-01", 0): ("confirmed_avoidable_absent_topic_mention", [5, 6], "Workout is marked unmentioned; yesterday's tennis is nevertheless discussed."),
    ("u2", "2027-12-11", 0): ("confirmed_avoidable_absent_topic_mention", [7, 8], "Workout is marked unmentioned; user explicitly reports today's singles match. The value tennis is correct, but this is an extra observation outside planned mention sampling."),
    ("u2", "2027-12-31", 0): ("confirmed_avoidable_absent_topic_mention", [7, 8], "Workout is marked unmentioned; assistant and user discuss tennis a couple of days ago."),
    ("u3", "2026-07-19", 0): ("refuted_by_verbatim_correct_relative_date", [2, 4], "The user says 'I came down with a really nasty case of the flu yesterday.' July 19 minus one day is July 18, effective day 140. Event and cause link are present."),
    ("u3", "2026-08-28", 0): ("refuted_by_required_world_statement", [4], "Required distractor0 literally says 'her roommate bought a new espresso machine'; having a roommate is entailed by that statement, not newly invented. Exact purchase date is underspecified but not the saved invention allegation."),
    ("u3", "2027-05-25", 0): ("content_present_date_underspecified", [6, 8], "The user explicitly states a new sandwich shop just opened near the clinic. The precise effective date May 24 is not pinned down by 'just'; no different day is clearly stated. Missing event content is not established."),
    ("u3", "2027-06-04", 0): ("confirmed_unmodeled_persistent_detail", [10], "The user refers to 'our future trip'. World statements include an online Italian course, but no trip. A future trip beyond today violates the writer's no-new-trips/plans instruction; it does not contradict the day's correct commute answer."),
    ("u3", "2027-06-26", 0): ("confirmed_unmodeled_persistent_detail", [12], "'Instead of focusing on Phoenix' adds prior wedding-venue planning history absent from the world. The earlier May 25 transcript also mentions venue browsing/Phoenix, so this is an unmodeled embellishment rather than an internal conversation contradiction. The Tucson move and its Wednesday date are correctly stated."),
    ("u3", "2027-09-13", 0): ("confirmed_required_absent_topic_conflict", [7, 8], "The world requires distractor2, a podcast host's book tour, while podcast.mentioned is false. Literal required content intrinsically raises the supposedly absent topic. The observed topic mention is real; assigning the entire failure to discretionary writer leakage ignores contradictory instructions."),
    ("u3", "2027-09-13", 1): ("content_present_date_underspecified", [7, 8], "The assistant states the book-tour news; the user reacts as if newly learning it. Event content is present. 'I saw some news this morning' specifies when the assistant saw news, not unambiguously when it was announced. No definite wrong event date is established. Distractors are a separate writer block from REQUIRED UPDATES (conversation.py:65,101); the distractor rules allow natural small talk and do not explicitly require a user statement, so speaker attribution alone is not a confirmed contract defect."),
    ("u3", "2027-12-01", 0): ("refuted_by_verbatim_correct_relative_date", [2], "The user says 'I sprained my ankle yesterday while I was out on a hike.' December 1 minus one day is November 30, effective day 640. The ankle content and date are present."),
    ("u3", "2027-12-01", 1): ("confirmed_unmodeled_persistent_detail", [10], "Dash being excited on walks and needing leash-pulling training adds behavior/backstory not present in the world, which states only adoption of a rescue greyhound named Dash. This violates no-new-details-about-known-pets; the sprain and rideshare still appear."),
    ("u4", "2026-03-02", 0): ("confirmed_required_absent_topic_conflict", [8, 9, 10], "Required hobbies.1 explicitly introduces cycling, while workout.mentioned is false. The forced hobby overlaps the absent workout topic; historical cycling is observable but the generator's literal obligations conflict."),
    ("u4", "2026-04-13", 0): ("confirmed_avoidable_absent_topic_mention", [5, 6], "Workout is marked unmentioned; assistant and user discuss yesterday's long bike ride."),
    ("u4", "2026-04-28", 0): ("confirmed_avoidable_absent_topic_mention", [5, 6], "Workout is marked unmentioned; assistant asks about bike activity and user discusses recent cycling."),
    ("u4", "2026-06-19", 0): ("confirmed_avoidable_absent_topic_mention", [4, 5], "Workout is marked unmentioned; user reports today's bike ride. The saved extractor selects cycling, while the world value is rest day. The pref_leak branch masks that value disagreement because its expected value is not mentioned. A reader could distinguish a leisure ride from the modeled workout, so the strongest independent finding is a real extra cycling observation plus a recorded-extraction/world discrepancy; no probing question directly cites this day."),
    ("u4", "2026-07-29", 0): ("refuted_by_verbatim_correct_relative_date", [4, 6], "The user says 'last night' they attended an instructional session preparing traditional culinary dishes from Thailand. July 29 minus one day is July 28, effective day 150. Paraphrase states the Thai cooking class."),
    ("u4", "2026-09-26", 0): ("confirmed_required_absent_topic_conflict", [1, 2, 7], "Required wrist.0 says fractured his wrist climbing, while workout.mentioned is false. Stating the mandated cause overlaps the absent topic. The assistant also independently asks about cycling in turn 1. The fracture content is present and dated today."),
    ("u4", "2026-11-16", 0): ("confirmed_unmodeled_persistent_detail", [6], "'We'll probably do a proper celebration next weekend' adds a beyond-today plan absent from world statements. This violates the strict writer instruction; it is a plausible embellishment and does not make the promotion false."),
    ("u4", "2026-12-26", 0): ("refuted_by_verbatim_correct_relative_date", [6, 8, 10], "The user says 'Yesterday' they inherited grandfather's vinyl compilation, clarifies it is a massive stack of albums, and says they sorted it yesterday evening. December 26 minus one day is December 25, effective day 300."),
    ("u4", "2027-08-24", 0): ("refuted_by_verbatim_correct_relative_date", [4], "The user says 'Two days ago, we officially kicked off a massive design competition deadline'. August 24 minus two days is August 22, effective day 540. The entire required event and date are present."),
}


def source_loc(path, needle):
    hits = [i for i, line in enumerate(path.read_text().splitlines(), 1) if needle in line]
    return {"file": str(path.relative_to(ROOT)), "line": hits[0] if hits else None}


def main():
    flagged = json.loads((OUT / "flagged_sessions.json").read_text())
    probes = [json.loads(l) for l in (OUT / "probe_coverage.jsonl").read_text().splitlines()]
    rows, per_session, independently_confirmed, masked_discrepancies = [], [], [], []
    for s in sorted(flagged, key=lambda s: (s["user"], s["day"])):
        classes = []
        for i, failure in enumerate(s["failures"]):
            key = (s["user"], s["date"], i)
            classification, turns, reasoning = REVIEW[key]
            classes.append(classification)
            row = {"user": s["user"], "day": s["day"], "date": s["date"], "failure_index": i,
                "recorded_failure": failure, "recorded_failure_type": failure.split(":")[0],
                "previously_reported": True, "failure_source": s["failure_source"], "classification": classification,
                "reasoning": reasoning, "raw_evidence": [{"turn_number": j, **s["turns"][j - 1]} for j in turns],
                "world_day_source": s["world_day_source"], "world_preferences": s["world_preferences"], "required_statements": s["required_statements"],
                "cited_by_probe_ids": s["cited_by_probe_ids"], "proved_wrong_gold_probe_ids": [],
                "limit": "A session citation does not imply the question depends on the defective detail; affected IDs here are citation exposure only."}
            if s["user"] == "u2" and s["day"] == 121:
                w = ROOT / "datasets/v2/u2/world_state.json"
                p = ROOT / "datasets/v2/u2/sessions/2026-06-28.json"
                row["counter_evidence"] = [source_loc(w, '"id": "hobby.tennis"'),
                    source_loc(p, 'I joined a tennis club today!')]
            rows.append(row)
            if classification.startswith("confirmed_"):
                independently_confirmed.append(row)
        for domain, selected in s["extracted"]["preferences"].items():
            pref = s["world_preferences"][domain]
            if not pref["mentioned"] and selected not in ("not mentioned", "other") and selected != pref["value"]:
                masked_discrepancies.append({"user": s["user"], "day": s["day"], "date": s["date"], "domain": domain,
                    "stored_world_value": pref["value"], "saved_extracted_value": selected,
                    "world_day_source": s["world_day_source"], "recorded_failures": s["failures"],
                    "raw_evidence": s["turns"][3], "cited_by_probe_ids": s["cited_by_probe_ids"],
                    "limit": "The text describes a real bicycle ride; distinguishing leisure cycling from the modeled workout is a semantic limitation. This does not prove a wrong probe gold."})
        per_session.append({"user": s["user"], "day": s["day"], "date": s["date"],
            "source": s["source"], "reported_failures": s["failures"], "review_classifications": classes,
            "cited_by_probe_ids": s["cited_by_probe_ids"]})
    count = collections.Counter(r["classification"] for r in rows)
    confirmed_probe_ids = sorted({p for r in independently_confirmed for p in r["cited_by_probe_ids"]})
    forced = [r for r in rows if r["classification"] == "confirmed_required_absent_topic_conflict"]
    unmodeled = [r for r in rows if r["classification"] == "confirmed_unmodeled_persistent_detail"]
    clean_optional = [r for r in independently_confirmed if r not in forced]
    result = {"summary": {"reported_flagged_sessions": len(flagged), "reported_failure_occurrences": len(rows),
        "failure_classifications": dict(count), "reported_flag_citation_exposure_probe_ids": sorted(q["id"] for q in probes if q["flagged_evidence_days"]),
        "reported_flag_citation_exposure_count": sum(bool(q["flagged_evidence_days"]) for q in probes),
        "confirmed_topic_mention_occurrences": 15, "forced_required_absent_conflicts": len(forced),
        "confirmed_unmodeled_detail_occurrences": len(unmodeled),
        "reported_statement_missing_with_content_completely_absent": 0,
        "reported_missing_statement_with_correct_explicit_relative_date": 5,
        "confirmed_source_contract_session_count": len({(r["user"], r["day"]) for r in independently_confirmed}),
        "confirmed_source_contract_citation_exposure_probe_count": len(confirmed_probe_ids),
        "confirmed_source_contract_citation_exposure_probe_ids": confirmed_probe_ids,
        "avoidable_mention_or_unmodeled_detail_session_count": len({(r["user"], r["day"]) for r in clean_optional}),
        "avoidable_mention_or_unmodeled_detail_citation_exposure_count": len({p for r in clean_optional for p in r["cited_by_probe_ids"]}),
        "unmodeled_detail_citation_exposure_probe_count": len({p for r in unmodeled for p in r["cited_by_probe_ids"]}),
        "masked_absent_topic_extracted_value_disagreements": len(masked_discrepancies),
        "source_workstream_wrong_gold_proved_count": 0},
        "failure_reviews": rows, "per_session_probe_sets": per_session,
        "masked_absent_topic_value_disagreements": masked_discrepancies}
    (OUT / "flag_reviews.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    issues = [{"id": f"SOURCE-FLAG-{i:03}", "scope": "conversation_source_contract", **r} for i, r in enumerate(independently_confirmed, 1)]
    issues.extend([
        {"id": "INTEGRATION-001", "scope": "monthly_evaluation_integration", "kind": "monthly_probe_files_not_loaded",
         "source": "experiments/benchmarks/v2.py:16", "expected": "Load monthly probing files to evaluate the monthly benchmark.",
         "observed": "Actual loader reads qa_pairs.json: 250 end-of-history curated questions, zero of 1,000 monthly IDs.",
         "record_ids": sorted(q["id"] for q in probes), "question_error_count": None},
        {"id": "INTEGRATION-002", "scope": "monthly_evaluation_integration", "kind": "no_cutoff_in_runner",
         "source": "experiments/core/runner.py:73", "observed": "Actual runner ingests all sessions before answering. Manual supply of March 31 monthly probes observes 699 later u1 sessions through February 28, 2028.",
         "reproduced_ids": ["u1_p202603_0001", "u1_p202603_0002"], "question_error_count": None,
         "limit": "Normal loader currently ignores monthly questions; this is an integration blocker and a demonstrated hypothetical leakage path, not evidence a completed monthly result run occurred."}])
    (OUT / "issues.json").write_text(json.dumps(issues, indent=2, ensure_ascii=False) + "\n")
    table = ["# Flagged session references", "",
        "Every row is a previously reported retained session flag. Probe IDs are citation exposures, not confirmed wrong answers. Review interpretations are kept separate in `flag_reviews.json`.", "",
        "| User/day | Retained session validation | Review classifications | Citing monthly probe IDs |",
        "|---|---|---|---|"]
    for s in per_session:
        path, line = s["source"]["file"], s["source"]["line"]
        flags = "; ".join(f.split(":")[0] for f in s["reported_failures"])
        classes = "; ".join(s["review_classifications"])
        ids = ", ".join("`" + q + "`" for q in s["cited_by_probe_ids"]) or "None"
        table.append(f'| {s["user"]}/{s["day"]} | [{s["date"]}]({ROOT / path}:{line}) — {flags} | {classes} | {ids} |')
    (OUT / "flagged-session-index.md").write_text("\n".join(table) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
