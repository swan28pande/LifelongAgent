#!/usr/bin/env python3
"""Reproduce narrow transcript scans; manual judgments remain explicitly separate."""
from __future__ import annotations
import collections
import hashlib
import json
import re
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_gold import ROOT, OUT, Source

PET = re.compile(r"\b(pets?|cats?|dogs?|animals?)\b", re.I)
NEGATIVE_OR_FIRST = re.compile(r"\b(no|never|without|first|don.t|didn.t|haven.t|hadn.t|not)\b.{0,80}\b(pets?|cats?|dogs?|animals?)\b|\b(pets?|cats?|dogs?|animals?)\b.{0,80}\b(no|never|without|first|not)\b", re.I)


def main():
    coverage = json.loads((OUT / "coverage.json").read_text())
    sources = {}
    records = []
    for uid in ("u2", "u3", "u4", "u5"):
        path = ROOT / f"datasets/v2/{uid}/conversations.json"
        source = Source(path)
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        user_turns = []
        for date, session in source.data["sessions"].items():
            for index, turn in enumerate(session["turns"]):
                if turn["speaker"] == "user":
                    user_turns.append({"date": date, "text": turn["text"], "source": source.loc(f"/sessions/{date}/turns/{index}/text")})
        for row in coverage:
            if row["user_id"] != uid or row["type"] != "fact_at_time" or row["observed_answer"] != "none":
                continue
            prefix = [turn for turn in user_turns if turn["date"] <= row["probe_date"]]
            mentions = [turn for turn in prefix if PET.search(turn["text"])]
            candidates = [turn for turn in mentions if NEGATIVE_OR_FIRST.search(turn["text"])]
            records.append({"id": row["id"], "check": "empty_fact_transcript_scan", "cutoff": row["probe_date"],
                            "user_turns_in_prefix": len(prefix), "pet_vocabulary_user_turns": len(mentions),
                            "negative_or_first_pet_regex_candidates": candidates,
                            "pet_mentions_on_adoption_date": [turn for turn in mentions if turn["date"] == "2027-01-24"],
                            "limitation": "Lexical screening plus manual review of recorded fact statements; not an exhaustive semantic proof of absence. Empty-set truth is separately established by world state."})
        selected_dates = {"u4": ["2027-10-26", "2027-10-27"],
                          "u5": ["2026-09-08", "2026-09-12", "2026-09-13", "2027-01-13", "2027-01-22", "2027-01-25", "2027-02-08", "2027-02-14", "2027-04-24"]}.get(uid, [])
        for date in selected_dates:
            records.append({"user_id": uid, "date": date, "check": "selected_user_turns_for_manual_semantic_review",
                            "turns": [turn for turn in user_turns if turn["date"] == date]})
    (OUT / "manual_transcript_checks.json").write_text(json.dumps({"input_sha256": sources, "records": records}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"empty_fact_scans": sum(r["check"] == "empty_fact_transcript_scan" for r in records),
                      "selected_sessions": sum(r["check"] == "selected_user_turns_for_manual_semantic_review" for r in records)}, indent=2))


if __name__ == "__main__":
    main()
