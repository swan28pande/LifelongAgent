"""Offline comparison with the preserved original v2 audit; no dataset writes."""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / ".research/v2-probing-audit"


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def dates(text):
    for value in re.findall(r"\b20\d\d-\d\d-\d\d\b", text):
        yield value, dt.date.fromisoformat(value)
    for value in re.findall(r"\b[A-Za-z]+\s+20\d\d\b", text):
        try:
            target = dt.datetime.strptime(value, "%B %Y").date()
        except ValueError:
            continue
        yield value, target


def main():
    baseline = json.loads((BASE / "manifest.json").read_text())
    comparison = []
    for name, before in baseline.items():
        path = ROOT / name
        after = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        comparison.append({"file": name, "before": before["sha256"], "after": after,
                           "status": "unchanged" if after == before["sha256"] else "changed" if after else "missing"})
    dump("input_comparison.json", comparison)

    # Reuse the baseline structural implementation with a distinct output directory.
    module = runpy.run_path(str(BASE / "structure.py"), run_name="updated_structure")
    module["main"].__globals__["OUT"] = OUT
    module["main"]()

    previous, current, pools = {}, [], {}
    for uid in ("u1", "u2", "u3", "u4", "u5"):
        filename = f"datasets/v2/{uid}/probing_questions.json"
        raw = subprocess.run(["git", "show", f"98a2674:{filename}"], cwd=ROOT,
                             check=True, capture_output=True).stdout
        if hashlib.sha256(raw).hexdigest() != baseline[filename]["sha256"]:
            raise ValueError(f"Git baseline does not match the recorded audit: {filename}")
        data = json.loads(raw)
        previous.update({q["id"]: dict(q, user_id=uid) for p in data["probes"] for q in p["questions"]})
        base = ROOT / f"datasets/v2/{uid}"
        now = json.loads((base / "probing_questions.json").read_text())
        positions = module["locations"]((base / "probing_questions.json").read_text())
        current.extend(dict(q, user_id=uid, file=filename, line=positions[q["id"]]["question"])
                       for p in now["probes"] for q in p["questions"])
        pools[uid] = json.loads((base / "qa_pool.json").read_text())

    rows = [json.loads(line) for line in (BASE / "question_coverage.jsonl").read_text().splitlines()]
    matched = []
    for row in rows:
        if not row["finding_categories"]:
            continue
        old = previous[row["id"]]

        def same(q):
            return q["question"] == old["question"] and q["type"] == old["type"] and q["domain"] == old["domain"]

        selected = [q for q in current if q["user_id"] == old["user_id"] and same(q)]
        pool_matches = [q for q in pools[old["user_id"]] if same(q)]
        matched.append({"baseline_id": old["id"], "categories": row["finding_categories"],
                        "baseline": old, "current_matches": selected,
                        "current_pool_matches": pool_matches,
                        "selection_status": "still_selected" if selected else "pool_only" if pool_matches else "absent_from_current_pool"})
    dump("prior_finding_matches.json", matched)
    dump("current_questions.json", current)

    future = []
    for q in current:
        cutoff = dt.date.fromisoformat(q["probe_date"])
        for field in ("question", "answer", "accept"):
            text = json.dumps(q[field]) if field == "accept" else q[field]
            for literal, target in dates(text):
                if target > cutoff:
                    future.append({"id": q["id"], "field": field, "literal": literal,
                                   "probe_date": q["probe_date"], "file": q["file"], "line": q["line"]})
    dump("all_fields_future_dates.json", future)
    summary = {"current_questions": len(current), "current_types": dict(collections.Counter(q["type"] for q in current)),
               "baseline_narrow_finding_instances": len(matched),
               "exact_question_matching_status": dict(collections.Counter(q["selection_status"] for q in matched)),
               "future_dates_in_question_answer_or_accept": len(future)}
    dump("comparison_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
