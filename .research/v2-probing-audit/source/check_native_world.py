#!/usr/bin/env python3
"""Native read-only world checks and replay of saved (not newly extracted) claims."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))

# Validator and conversation import LLM only for annotations/functions not exercised.
# Refuse all constructions so this script cannot accidentally create a cloud model.
class DisabledLLM:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("LLM disabled by read-only audit")

fake_llm = types.ModuleType("generator_v2.llm")
fake_llm.LLM = DisabledLLM
sys.modules["generator_v2.llm"] = fake_llm

from generator_v2 import checks, config, conversation, rules, simulator, validator
from generator_v2.schema import WorldState, load_ladder, load_spec


def main():
    ladder = load_ladder(config.LADDER_PATH)
    users, values, replay = [], [], []
    for user in (f"u{i}" for i in range(1, 6)):
        base = ROOT / "datasets/v2" / user
        spec = load_spec(config.PERSONA_DIR / f"{user}.yaml", ladder)
        saved = json.loads((base / "world_state.json").read_text())
        world = WorldState.model_validate(saved)
        regenerated = simulator.simulate(spec).model_dump(mode="json")
        base_mismatches, exception_mismatches, cells, exceptions = [], [], 0, 0
        regimes = {r.id: r for r in world.regimes}
        prefs = {p.domain: p for p in spec.preferences}
        for day in world.days:
            cities = day.active_facts.get("city", [])
            ctx = rules.context_for(day.date, cities[0] if cities else None)
            for domain, pref in day.preferences.items():
                cells += 1
                reg = regimes[pref.regime_id]
                expected = rules.evaluate(reg.rule, day.day, day.date, reg.anchor, ctx)
                problems = []
                if pref.base_value != expected:
                    problems.append("base_value_differs_from_resolved_rule")
                if not reg.start <= day.day <= reg.end or reg.domain != domain:
                    problems.append("regime_membership_mismatch")
                if not pref.is_exception and pref.value != expected:
                    problems.append("regular_value_differs_from_resolved_rule")
                if problems:
                    base_mismatches.append({"day": day.day, "domain": domain, "actual": pref.model_dump(),
                                            "expected_base": expected, "problems": problems})
                if pref.is_exception:
                    exceptions += 1
                    explicit = next((e for e in prefs[domain].exceptions if e.day == day.day), None)
                    allowed = ((explicit.value, explicit.reason),) if explicit else tuple((p.value, p.reason) for p in prefs[domain].exception_pool)
                    if (pref.value, pref.exception_reason) not in allowed:
                        exception_mismatches.append({"day": day.day, "domain": domain, "value": pref.value,
                                                     "reason": pref.exception_reason, "allowed": allowed})
                values.append({"user": user, "day": day.day, "date": day.date.isoformat(), "domain": domain,
                    "regime_id": reg.id, "anchor": reg.anchor, "resolved_rule_type": reg.rule.type,
                    "evaluated_base_value": expected, "stored_base_value": pref.base_value,
                    "stored_value": pref.value, "is_exception": pref.is_exception,
                    "exception_reason": pref.exception_reason, "base_matches": pref.base_value == expected,
                    "regular_value_matches": pref.is_exception or pref.value == expected})
            if not day.has_session:
                continue
            session = json.loads((base / "sessions" / f"{day.date.isoformat()}.json").read_text())
            check = validator.build_check(spec, world, day)
            rechecked = validator.diff(check, session["extracted"])
            replay.append({"user": user, "day": day.day, "date": day.date.isoformat(),
                "num_required_statements": len(check.statements), "required_statement_ids": sorted(check.statements),
                "saved_failures": session["failures"], "replayed_failures": rechecked,
                "equal_to_saved": rechecked == session["failures"],
                "native_shape_errors": conversation.shape_errors(session["turns"]),
                "known_input": check.known if session["failures"] else None,
                "causes_input": check.causes if session["failures"] else None})
        measured = checks.measure_knobs(spec, world)
        knob_rows = checks.compare_knobs(spec, measured)
        users.append({"user": user, "persona_and_world_schema_validated": True,
            "spec_errors": checks.check_spec(spec), "world_errors": checks.check_world(spec, world),
            "knob_errors": checks.knob_errors(knob_rows), "knob_rows": [dataclasses.asdict(r) for r in knob_rows],
            "deterministically_regenerated_world_json_equal": saved == regenerated,
            "regenerated_world_top_level_mismatches": [k for k in saved if saved[k] != regenerated.get(k)],
            "preference_cells_checked": cells, "exception_cells_checked": exceptions,
            "base_regime_mismatches": base_mismatches, "exception_value_reason_mismatches": exception_mismatches})
    source_paths = [ROOT / p for p in ("generator_v2/schema.py", "generator_v2/checks.py", "generator_v2/rules.py",
        "generator_v2/simulator.py", "generator_v2/validator.py", "generator_v2/conversation.py", "generator_v2/config.py",
        "generator_v2/DECISIONS.md", "generator_v2/prompts/conversation_system.txt", "generator_v2/prompts/validator_system.txt",
        "generator_v2/personas/difficulty_ladder.yaml", "experiments/benchmarks/v2.py", "experiments/core/runner.py",
        "experiments/core/grading.py", "experiments/core/types.py", "experiments/core/method.py", "experiments/config.py")]
    source_paths += sorted((ROOT / "generator_v2/personas").glob("u[1-5].yaml"))
    result = {"users": users, "totals": {"days_schema_validated": 3650,
        "preference_cells_checked": len(values), "exception_cells_checked": sum(u["exception_cells_checked"] for u in users),
        "native_spec_errors": sum(len(u["spec_errors"]) for u in users),
        "native_world_errors": sum(len(u["world_errors"]) for u in users),
        "native_knob_errors": sum(len(u["knob_errors"]) for u in users),
        "base_regime_mismatches": sum(len(u["base_regime_mismatches"]) for u in users),
        "exception_value_reason_mismatches": sum(len(u["exception_value_reason_mismatches"]) for u in users),
        "world_regeneration_equal_users": sum(u["deterministically_regenerated_world_json_equal"] for u in users),
        "recorded_validator_replay_sessions": len(replay), "recorded_validator_replay_mismatches": sum(not r["equal_to_saved"] for r in replay),
        "required_statement_occurrences_checked": sum(r["num_required_statements"] for r in replay),
        "native_shape_failure_sessions": sum(bool(r["native_shape_errors"]) for r in replay)},
        "limits": "Replayed diff uses existing extracted claims; it checks deterministic consistency, not a new semantic transcript judgment.",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    (OUT / "native_world_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "world_preference_coverage.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values))
    (OUT / "validator_replay.jsonl").write_text("".join(json.dumps(r) + "\n" for r in replay))
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
