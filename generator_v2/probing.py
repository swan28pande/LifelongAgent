"""Generate timestamped probing questions for sequential evaluation.

For each user, assigns existing QA pool questions to monthly probe points based on when
the evidence becomes available. At each probe (end of month N), questions can reference
evidence from month 1 through month N — testing memory incrementally, not only at the end.

A question is assigned to the month containing its latest evidence day. Some questions
are promoted to later probes as retrospective tests (asking about month 3 during month 6).

Output: experiments/data/synthetic_v2/<user>/probing_questions.json

Usage:
    python -m generator_v2.probing                # all users
    python -m generator_v2.probing --user u1      # single user
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from . import config


def _month_boundaries(start: dt.date, num_days: int) -> list[dict]:
    """Return probe points: last day of each month within the timeline."""
    probes = []
    current = start
    while True:
        if current.month == 12:
            month_end = dt.date(current.year + 1, 1, 1) - dt.timedelta(days=1)
        else:
            month_end = dt.date(current.year, current.month + 1, 1) - dt.timedelta(days=1)
        day_num = (month_end - start).days + 1
        if day_num > num_days:
            break
        probes.append({
            "day": day_num,
            "date": month_end.isoformat(),
            "month": month_end.strftime("%Y-%m"),
        })
        current = month_end + dt.timedelta(days=1)
    return probes


def _day_to_month(day: int, start: dt.date) -> str:
    d = start + dt.timedelta(days=day - 1)
    return d.strftime("%Y-%m")


def build_probing_questions(
    user_id: str, retro_fraction: float = 0.3, max_questions: int = 200,
) -> dict:
    """Build monthly probing questions for one user.

    Takes the full QA pool (qa_pool.json), assigns each to the earliest valid
    monthly probe, samples down to max_questions (distributed proportionally
    across months with at least 1 per month that has any), and creates
    retrospective copies of some questions at later probes.
    """
    base = config.OUTPUT_DIR / user_id
    qa_pool = json.loads((base / "qa_pool.json").read_text())
    ws = json.loads((base / "world_state.json").read_text())
    start = dt.date.fromisoformat(ws["start_date"])
    num_days = ws["num_days"]
    name = ws["name"]

    probes = _month_boundaries(start, num_days)
    month_to_probe = {p["month"]: p for p in probes}
    probe_months = [p["month"] for p in probes]

    rng = random.Random(f"{ws.get('seed', user_id)}:probing")

    # Assign each question to the earliest probe after all evidence is available.
    # For fact_at_time questions, also consider the date embedded in the question
    # text — the system can't be asked about a date it hasn't reached yet.
    by_month: dict[str, list[dict]] = defaultdict(list)
    for q in qa_pool:
        evidence = q.get("evidence_days", [])
        if not evidence:
            continue
        latest_evidence = max(evidence)
        if latest_evidence > num_days:
            continue
        effective_day = latest_evidence
        if q["type"] == "fact_at_time":
            m = re.search(r"\d{4}-\d{2}-\d{2}", q["question"])
            if m:
                qdate = dt.date.fromisoformat(m.group())
                qday = (qdate - start).days + 1
                effective_day = max(effective_day, qday)
                if effective_day > num_days:
                    continue
        assigned_month = _day_to_month(effective_day, start)
        if assigned_month not in month_to_probe:
            continue
        by_month[assigned_month].append(q)

    # Downsample to fit primary + retrospective within max_questions.
    primary_budget = max_questions - round(max_questions * retro_fraction / (1 + retro_fraction))
    total_available = sum(len(v) for v in by_month.values())
    # Over-allocate primary slightly since some retro slots can't be filled
    # (candidates near timeline end can't be placed 2+ months later)
    primary_budget = min(primary_budget + 4, total_available)
    if total_available > primary_budget:
        non_empty = [m for m in probe_months if by_month.get(m)]
        # Allocate proportionally, guarantee at least 1 per non-empty month
        month_budgets = {}
        for m in non_empty:
            month_budgets[m] = max(1, round(primary_budget * len(by_month[m]) / total_available))
        # Adjust to hit exact budget
        allocated = sum(month_budgets.values())
        diff = allocated - primary_budget
        adjustable = sorted(non_empty, key=lambda m: len(by_month[m]), reverse=(diff > 0))
        for m in adjustable:
            if diff == 0:
                break
            if diff > 0 and month_budgets[m] > 1:
                month_budgets[m] -= 1
                diff -= 1
            elif diff < 0 and month_budgets[m] < len(by_month[m]):
                month_budgets[m] += 1
                diff += 1
        for m in non_empty:
            pool = by_month[m]
            if len(pool) > month_budgets[m]:
                by_month[m] = rng.sample(pool, month_budgets[m])

    # Build probes with assigned questions
    result_probes = []
    all_assigned = []
    qid_counter = 0

    for probe in probes:
        month = probe["month"]
        questions = by_month.get(month, [])

        probe_qs = []
        for q in questions:
            qid_counter += 1
            pq = {
                "id": f"{user_id}_p{month.replace('-', '')}_{qid_counter:04d}",
                "original_id": q.get("id", ""),
                "type": q["type"],
                "capability": q.get("capability", ""),
                "domain": q.get("domain", ""),
                "question": q["question"],
                "answer": q["answer"],
                "answer_type": q["answer_type"],
                "accept": q["accept"],
                "evidence_days": q["evidence_days"],
                "grading": q["grading"],
                "viewpoint_day": probe["day"],
                "probe_date": probe["date"],
                "probe_month": month,
                "is_retrospective": False,
                "tags": q.get("tags", []),
            }
            probe_qs.append(pq)
            all_assigned.append((month, pq))

        result_probes.append({
            "probe_date": probe["date"],
            "probe_month": month,
            "viewpoint_day": probe["day"],
            "questions": probe_qs,
        })

    # Add retrospective questions: pick questions from earlier months
    # and re-ask them at a later probe to test long-term retention
    probe_idx = {p["probe_month"]: i for i, p in enumerate(result_probes)}
    retro_candidates = [(m, q) for m, q in all_assigned
                        if probe_idx.get(m, len(probes)) < len(probes) - 1]

    primary_total = sum(len(p["questions"]) for p in result_probes)
    retro_budget = max_questions - primary_total
    n_retro = min(int(len(retro_candidates) * retro_fraction), retro_budget)
    if retro_candidates and n_retro > 0:
        # Over-sample candidates to account for those that can't be placed
        rng.shuffle(retro_candidates)
        added = 0
        for orig_month, orig_q in retro_candidates:
            if added >= n_retro:
                break
            orig_idx = probe_idx[orig_month]
            if len(probes) - orig_idx - 1 < 2:
                continue
            offset = rng.randint(2, min(6, len(probes) - orig_idx - 1))
            target_idx = orig_idx + offset
            target_probe = result_probes[target_idx]

            qid_counter += 1
            retro_q = {
                **orig_q,
                "id": f"{user_id}_r{target_probe['probe_month'].replace('-', '')}_{qid_counter:04d}",
                "viewpoint_day": target_probe["viewpoint_day"],
                "probe_date": target_probe["probe_date"],
                "probe_month": target_probe["probe_month"],
                "is_retrospective": True,
                "retrospective_of": orig_q["id"],
                "original_probe_month": orig_month,
            }
            target_probe["questions"].append(retro_q)
            added += 1

    # Compute counts
    total = 0
    for p in result_probes:
        p["num_questions"] = len(p["questions"])
        p["num_retrospective"] = sum(1 for q in p["questions"] if q.get("is_retrospective"))
        total += p["num_questions"]

    return {
        "user_id": user_id,
        "name": name,
        "start_date": ws["start_date"],
        "num_days": num_days,
        "num_probes": len(result_probes),
        "total_questions": total,
        "retro_fraction": retro_fraction,
        "probes": result_probes,
    }


def generate_all(users: list[str] | None = None) -> None:
    base = config.OUTPUT_DIR
    if users is None:
        users = sorted(
            p.stem for p in config.PERSONA_DIR.glob("u*.yaml")
            if ".draft" not in p.name and (base / p.stem / "qa_pool.json").exists()
        )

    for uid in users:
        print(f"[{uid}] generating probing questions...", flush=True)
        result = build_probing_questions(uid)
        out_path = base / uid / "probing_questions.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        print(f"  {result['num_probes']} probes, {result['total_questions']} questions "
              f"({sum(p['num_retrospective'] for p in result['probes'])} retrospective)")
        for p in result["probes"]:
            retro = f" (+{p['num_retrospective']} retro)" if p["num_retrospective"] else ""
            print(f"    {p['probe_month']}: {p['num_questions']} questions{retro}")
        print(f"  -> {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", nargs="+", help="specific users (default: all)")
    args = ap.parse_args()
    generate_all(args.user)
