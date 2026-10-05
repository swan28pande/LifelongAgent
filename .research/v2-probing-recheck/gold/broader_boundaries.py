#!/usr/bin/env python3
"""Bounded choice-preserving date alternatives beyond the prior narrow audit.

These are diagnostic candidates until the free-text excerpts are manually
reviewed. A reversion retains its restored anchor when its boundary moves.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ns = runpy.run_path(str(ROOT / ".research/v2-probing-audit/gold/audit_gold.py"), run_name="boundaries_import")
ns["Source"].__init__.__globals__["ROOT"] = ROOT
rules, World, Source = ns["rules"], ns["WorldState"], ns["Source"]


def main():
    coverage = json.loads((OUT / "coverage.json").read_text())
    ladder = ns["load_ladder"](ROOT / "generator_v2/personas/difficulty_ladder.yaml")
    records = []
    reviewed = 0
    for uid in ("u1", "u2", "u3", "u4", "u5"):
        world = World.model_validate_json((ROOT / f"datasets/v2/{uid}/world_state.json").read_text())
        spec = ns["load_spec"](ROOT / f"generator_v2/personas/{uid}.yaml", ladder)
        conversations = Source(ROOT / f"datasets/v2/{uid}/conversations.json")
        for row in coverage:
            if row["user_id"] != uid or row["type"] != "change_detection" or row["visibility"] == "explicit":
                continue
            reviewed += 1
            regime = next(r for r in world.regimes if r.id == row["regime_id"])
            previous = next(r for r in world.regimes if r.id == row["previous_regime_id"])
            lower = max(previous.start + 14, regime.start - 14)
            upper = min(regime.end - 13, regime.start + 14, row["viewpoint_day"])
            if regime.cause_day is not None:
                lower = max(lower, regime.cause_day + spec.difficulty.lag_days.min)
                upper = min(upper, regime.cause_day + spec.difficulty.lag_days.max)
            observations = [d for d in world.days[lower - 1:min(row["viewpoint_day"], regime.end)] if d.preferences[row["domain"]].mentioned]
            alternatives = []
            for candidate in range(lower, upper + 1):
                variants = [("same_rule", regime.rule)]
                if regime.rule.type == "weekly_alternate":
                    variants.append(("rotated_weekly_value_order", regime.rule.model_copy(update={"values": regime.rule.values[1:] + regime.rule.values[:1]})))
                for variant, rule in variants:
                    valid = True
                    for day in observations:
                        context = rules.context_for(day.date, (day.active_facts.get("city") or [None])[0])
                        predicted = (rules.evaluate(previous.rule, day.day, day.date, previous.anchor, context) if day.day < candidate else
                                     rules.evaluate(rule, day.day, day.date, regime.anchor if regime.kind == "reversion" else candidate, context))
                        if predicted != day.preferences[row["domain"]].base_value:
                            valid = False
                            break
                    date = (world.start_date + dt.timedelta(days=candidate - 1)).isoformat()
                    if valid and date not in row["observed_accept"]:
                        alternatives.append({"day": candidate, "date": date, "variant": variant,
                                             "anchor": regime.anchor if regime.kind == "reversion" else candidate,
                                             "cause_lag": candidate - regime.cause_day if regime.cause_day is not None else None})
            if not alternatives:
                continue
            turns = []
            for date, session in conversations.data["sessions"].items():
                day = (dt.date.fromisoformat(date) - world.start_date).days + 1
                if lower - 2 <= day <= min(row["viewpoint_day"], max(regime.first_mention_day + 7, regime.start + 7)):
                    turns.extend({"day": day, "date": date, "text": t["text"], "source": conversations.loc(f"/sessions/{date}/turns/{i}/text")}
                                 for i, t in enumerate(session["turns"]) if t["speaker"] == "user")
            records.append({"id": row["id"], "user_id": uid, "regime_id": regime.id, "domain": row["domain"],
                            "question": row["question"], "question_source": row["source"]["question"],
                            "probe_date": row["probe_date"], "visibility": regime.visibility, "regime_kind": regime.kind,
                            "cause_day": regime.cause_day, "actual_start": regime.start, "first_mention": regime.first_mention_day,
                            "accepted": row["observed_accept"], "rejected_compatible_alternatives": alternatives,
                            "observed_choice_count": len(observations), "prefix_excerpt_turns": turns,
                            "scope": "All declared domain choices within the changed old/new interval through this probe, including stated exception base values; later regimes fixed; alternative dates no later than the probe and persona cause lags respected. Raw-text review is additional, not automatically established by value consistency."})
    summary = {"current_change_questions": sum(r["type"] == "change_detection" for r in coverage),
               "nonexplicit_change_questions_checked": reviewed, "candidate_question_count": len(records),
               "unique_regimes_with_candidates": len({(r["user_id"], r["regime_id"]) for r in records}),
               "ids": [r["id"] for r in records], "manual_review_status": "pending"}
    (OUT / "broader_boundary_candidates.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    (OUT / "broader_boundary_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
