#!/usr/bin/env python3
"""Verify final audit artifacts against current source inputs; no source mutations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def read(name):
    return json.loads((OUT / name).read_text())


def read_jsonl(name):
    return [json.loads(l) for l in (OUT / name).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inv, native, integration, review = (read(p) for p in ("inventory.json", "native_world_checks.json", "integration.json", "flag_reviews.json"))
    ss, qs, values, replay = (read_jsonl(p) for p in ("session_coverage.jsonl", "probe_coverage.jsonl", "world_preference_coverage.jsonl", "validator_replay.jsonl"))
    assert len(ss) == 2738 and len(qs) == 1000 and len(values) == 15330 and len(replay) == 2738
    assert sum(s["num_turns"] for s in ss) == 34659
    assert all(not u["session_projection_world_mismatches"] and not u["top_level_aggregate_world_mismatches"] and not u["fidelity_nonusage_mismatches"] for u in inv["users"])
    for s in ss:
        assert sha(ROOT / s["source"]["file"]) == s["session_file_sha256"]
    copies = read("copy_comparison.json")
    assert len(copies) == 35 and all(c["byte_equal"] and c["json_equal"] for c in copies)
    for c in copies:
        assert sha(ROOT / c["canonical"]) == c["canonical_sha256"]
        assert sha(ROOT / c["experiment"]) == c["experiment_sha256"]
    assert all(sha(ROOT / p) == h for p, h in native["source_sha256"].items())
    assert all(r["equal_to_saved"] and not r["native_shape_errors"] for r in replay)
    for key in ("native_spec_errors", "native_world_errors", "native_knob_errors", "base_regime_mismatches", "exception_value_reason_mismatches", "recorded_validator_replay_mismatches"):
        assert native["totals"][key] == 0
    assert native["totals"]["world_regeneration_equal_users"] == 5
    assert len(review["failure_reviews"]) == 28 and len(review["per_session_probe_sets"]) == 26
    flagged_ids = {q["id"] for q in qs if q["flagged_evidence_days"]}
    assert len(flagged_ids) == 54
    assert flagged_ids == {p for s in review["per_session_probe_sets"] for p in s["cited_by_probe_ids"]}
    assert not any(r["proved_wrong_gold_probe_ids"] for r in review["failure_reviews"])
    assert review["summary"]["source_workstream_wrong_gold_proved_count"] == 0
    assert review["summary"]["masked_absent_topic_extracted_value_disagreements"] == 1
    assert review["summary"]["failure_classifications"]["content_present_date_underspecified"] == 2
    assert "new_required_speaker_violation_count" not in review["summary"]
    assert not any(i.get("kind") == "required_update_first_disclosed_by_assistant" for i in read("issues.json"))
    assert integration["loader"]["loaded_questions"] == 250 and not integration["loader"]["monthly_ids_loaded"]
    assert not integration["loader"]["curated_metadata_mismatches"]
    assert all(r["future_sessions_seen"] == 699 for r in integration["runner_monthly_cutoff_reproduction"]["observations"])
    u5 = next(u for u in inv["users"] if u["user"] == "u5")
    assert u5["llm_log_exists"] and not u5["llm_log_is_lfs_pointer"]
    assert u5["llm_log_available_usage_records"] == 1735 and u5["llm_log_usage_equal_to_fidelity"]
    assert all(u["llm_log_is_lfs_pointer"] for u in inv["users"] if u["user"] != "u5")
    output = {"status": "passed", "canonical_individual_session_hashes_rechecked": len(ss),
        "canonical_experiment_pair_hashes_rechecked": len(copies), "source_code_hashes_rechecked": len(native["source_sha256"]),
        "monthly_probe_coverage": len(qs), "world_preference_cell_coverage": len(values),
        "saved_validation_replay_coverage": len(replay), "raw_flag_reviews": len(review["failure_reviews"]),
        "flag_citation_exposures": len(flagged_ids), "proven_wrong_gold_from_this_workstream": 0,
        "lfs_pointer_users": [u["user"] for u in inv["users"] if u["llm_log_is_lfs_pointer"]],
        "u5_usage_log_reconciled": True, "retracted_speaker_allegation_absent": True,
        "audit_artifact_sha256": {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "verification.json"}}
    (OUT / "verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "audit_artifact_sha256"}, indent=2))


if __name__ == "__main__":
    main()
