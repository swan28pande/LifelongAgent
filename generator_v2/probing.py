"""Historical monthly probes, grounded in information available at each probe.

Run ``python -m generator_v2.probing --user u1`` or omit --user for all users.
New retrospective records have their own probe date. Evaluation repetitions keep
the original record unchanged; the monthly runner owns that repetition protocol.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
from collections import Counter, defaultdict

from . import config
from .outputs import sync_outputs
from .qa import QABuilder
from .schema import PersonaSpec, QAItem, WorldState, load_ladder, load_spec

MONTHS = "January February March April May June July August September October November December".split()


def _month_boundaries(start: dt.date, num_days: int) -> list[dict]:
    """Full months and the final partial month, without dropping timeline days."""
    probes = []
    end = start + dt.timedelta(days=num_days - 1)
    current = start
    while current <= end:
        next_month = dt.date(current.year + current.month // 12, current.month % 12 + 1, 1)
        last = min(end, next_month - dt.timedelta(days=1))
        probes.append({"day": (last - start).days + 1,
                       "date": last.isoformat(), "month": last.strftime("%Y-%m")})
        current = next_month
    return probes


def _available_day(item: QAItem, start: dt.date) -> int:
    """Wait for evidence and all historical dates/periods in the reference."""
    latest = max(item.evidence_days, default=1)
    text = " ".join([item.question, item.answer, *item.accept])
    for value in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text):
        latest = max(latest, (dt.date.fromisoformat(value) - start).days + 1)
    for month, year in re.findall(r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b", text):
        m, y = MONTHS.index(month) + 1, int(year)
        end = dt.date(y + m // 12, m % 12 + 1, 1) - dt.timedelta(days=1)
        latest = max(latest, (end - start).days + 1)
    return latest


def _record(user_id: str, item: QAItem, probe: dict, number: int, original: dict | None = None) -> dict:
    result = item.model_dump(exclude={"difficulty_user", "recency_distance_days"})
    retrospective = original is not None
    result.update(id=f"{user_id}_{'r' if retrospective else 'p'}{probe['month'].replace('-', '')}_{number:04d}",
                  original_id=item.id, probe_date=probe["date"], probe_month=probe["month"],
                  is_retrospective=retrospective)
    if original:
        result.update(retrospective_of=original["id"], original_probe_month=original["probe_month"])
    return result


def build_probing_questions(user_id: str, retro_fraction: float = 0.3, max_questions: int = 200) -> dict:
    base = config.OUTPUT_DIR / user_id
    pool = [QAItem.model_validate(q) for q in json.loads((base / "qa_pool.json").read_text())]
    world = WorldState.model_validate_json((base / "world_state.json").read_text())
    spec = load_spec(config.PERSONA_DIR / f"{user_id}.yaml", load_ladder(config.LADDER_PATH))
    return build_probes(spec, world, pool, retro_fraction, max_questions)


def build_probes(spec: PersonaSpec, world: WorldState, pool: list[QAItem],
                 retro_fraction: float = 0.3, max_questions: int = 200) -> dict:
    if not 0 <= retro_fraction <= 1 or max_questions < 1:
        raise ValueError("retro_fraction must be between 0 and 1 and max_questions must be positive")
    user_id = world.user_id
    builder = QABuilder(spec, world)
    probes = _month_boundaries(world.start_date, world.num_days)
    rng = random.Random(f"{world.seed}:probing")
    by_probe: dict[int, list[QAItem]] = defaultdict(list)

    for item in pool:
        # Forecasts are intentionally excluded from this historical evaluation.
        if item.type == "prediction":
            continue
        current = item.type in ("fact_current", "pattern_current") or (
            item.type == "abstention" and "control" in item.tags)
        if not item.evidence_days and not (item.type == "abstention" and item.answer_type == "abstain"):
            continue
        available = _available_day(item, world.start_date)
        if current and item.type != "pattern_current":
            entity = item.domain if item.type == "fact_current" else next(t for t in item.tags if t in builder.facts)
            values = world.days[-1].active_facts[entity]
            # Preserve the previous introduction schedule based on disclosed values,
            # then recompute the complete answer (including removals) at that probe.
            available = max((n for value in values for n in builder._fact_evidence(entity, value)), default=1)
        for i, probe in enumerate(probes):
            if probe["day"] < available:
                continue
            grounded = builder.at_viewpoint(item, probe["day"])
            if grounded is not None:
                by_probe[i].append(grounded)
                break

    candidates = [(i, q) for i, items in sorted(by_probe.items()) for q in items]
    budget = min(len(candidates), max_questions,
                 round(max_questions / (1 + retro_fraction)))
    selected: list[tuple[int, QAItem]] = []
    chosen: set[tuple[int, str]] = set()

    def pick(i: int, item: QAItem) -> None:
        if len(selected) < budget and (i, item.id) not in chosen:
            selected.append((i, item))
            chosen.add((i, item.id))

    # Empty evidence is valid for genuinely unknown facts. Keep answerable controls
    # too, so abstention cannot be a constant answer.
    categories = {(q.type, q.answer_type == "abstain" if q.type == "abstention" else None)
                  for _, q in candidates}
    for category in sorted(categories):
        options = [(i, q) for i, q in candidates
                   if (q.type, q.answer_type == "abstain" if q.type == "abstention" else None) == category]
        i, item = rng.choice(options)
        pick(i, item)
    for i, item in candidates:
        if item.type == "abstention":
            pick(i, item)
    for i in sorted(by_probe):
        if not any(j == i for j, _ in selected):
            pick(i, rng.choice(by_probe[i]))
    # Allocate proportionally as before, keeping month and family coverage explicit.
    while len(selected) < budget:
        open_ = [i for i in sorted(by_probe) if any((i, q.id) not in chosen for q in by_probe[i])]
        if not open_:
            break
        i = min(open_, key=lambda n: (sum(j == n for j, _ in selected) / len(by_probe[n]), n))
        remaining = [q for q in by_probe[i] if (i, q.id) not in chosen]
        seen = Counter((q.type, q.domain) for j, q in selected if j == i)
        rng.shuffle(remaining)
        pick(i, min(remaining, key=lambda q: seen[q.type, q.domain]))

    result = [{"probe_date": p["date"], "probe_month": p["month"],
               "viewpoint_day": p["day"], "questions": []} for p in probes]
    originals = []
    number = 0
    for i, item in sorted(selected, key=lambda row: (row[0], row[1].id)):
        number += 1
        record = _record(user_id, item, probes[i], number)
        result[i]["questions"].append(record)
        originals.append((i, item, record))

    rng.shuffle(originals)
    for i, item, original in originals:
        if number >= max_questions:
            break
        targets = list(range(i + 2, min(i + 7, len(probes))))
        rng.shuffle(targets)
        for j in targets:
            if (j, item.id) in chosen:
                continue
            grounded = builder.at_viewpoint(item, probes[j]["day"])
            if grounded is None:
                continue
            number += 1
            result[j]["questions"].append(_record(user_id, grounded, probes[j], number, original))
            chosen.add((j, item.id))
            break

    # Late timelines or small pools may not have enough retrospective slots.
    for i, item in candidates:
        if number >= max_questions:
            break
        if (i, item.id) not in chosen:
            number += 1
            result[i]["questions"].append(_record(user_id, item, probes[i], number))
            chosen.add((i, item.id))
    for probe in result:
        probe["num_questions"] = len(probe["questions"])
        probe["num_retrospective"] = sum(q["is_retrospective"] for q in probe["questions"])
    return {"user_id": user_id, "name": world.name, "start_date": world.start_date.isoformat(),
            "num_days": world.num_days, "num_probes": len(probes), "total_questions": number,
            "retro_fraction": retro_fraction, "question_policy": "historical",
            "probes": result}


def generate_all(users: list[str] | None = None) -> None:
    if users is None:
        users = sorted(p.stem for p in config.PERSONA_DIR.glob("u*.yaml") if ".draft" not in p.name)
    for uid in users:
        result = build_probing_questions(uid)
        (config.OUTPUT_DIR / uid / "probing_questions.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False))
        sync_outputs(uid)
        print(f"[{uid}] {result['num_probes']} probes, {result['total_questions']} questions")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", nargs="+")
    generate_all(parser.parse_args().user)
