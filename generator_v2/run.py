"""CLI entry point.

    python -m generator_v2.run simulate --user all      # world state + checks + QA + stats, no LLM
    python -m generator_v2.run qa --user u5             # QA only (re-simulates; deterministic)
    python -m generator_v2.run generate --user u1 --days 1-183   # conversations (Vertex AI)
    python -m generator_v2.run fidelity --user u1       # fidelity.json + merged conversations.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter
from pathlib import Path

from . import checks, config, qa, simulator
from .schema import PersonaSpec, WorldState, load_ladder, load_spec


def persona_paths(user: str) -> list[Path]:
    if user != "all":
        return [config.PERSONA_DIR / f"{user}.yaml"]
    return sorted(p for p in config.PERSONA_DIR.glob("u*.yaml") if ".draft" not in p.name)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


# ── Printouts ───────────────────────────────────────────────────────

def knobs_table(rows: list[checks.KnobRow]) -> str:
    w = max(len(r.knob) for r in rows)
    lines = [f"  {'knob':{w}}  {'ladder':<22} {'measured':<22} ok"]
    for r in rows:
        target = r.target if len(r.target) <= 22 else r.target[:19] + "..."
        lines.append(f"  {r.knob:{w}}  {target:<22} {r.measured[:40]:<22} {'✓' if r.ok else '✗'}")
    return "\n".join(lines)


def domain_timeline(spec: PersonaSpec, world: WorldState) -> str:
    out = []
    d = spec.date_of
    for p in spec.preferences:
        out.append(f"\n  [{p.domain}]  ({p.noun})")
        for r in world.regimes_for(p.domain):
            span = f"d{r.start:>3}-{r.end:<3} {d(r.start).isoformat()}→{d(r.end).isoformat()}"
            line = f"    {r.kind:<9} {span}  {r.label}"
            if r.kind != "initial":
                if r.visibility == "uncaused":
                    why = "uncaused"
                elif r.kind == "reversion":
                    why = f"{r.visibility}: {r.cause_text}" if r.cause_text else "uncaused reversion"
                else:
                    why = f"{r.visibility}, lag {r.lag_days}: {r.cause_text} (d{r.cause_day})"
                line += f"\n{'':14}cause → {why}"
            line += f"\n{'':14}observed {len(r.mention_days)}×, first d{r.first_mention_day}"
            out.append(line)
        excs = [(day.day, day.preferences[p.domain]) for day in world.days if day.preferences[p.domain].is_exception]
        if excs:
            sample = ", ".join(f"d{n}: {x.value} ({x.exception_reason})" for n, x in excs[:3])
            out.append(f"    exceptions: {len(excs)} - e.g. {sample}")
        for dis in (x for x in spec.distractors if x.domain == p.domain):
            flag = " [CONFOUNDER]" if dis.confounder else ""
            out.append(f"    distractor d{dis.day}: {dis.text}{flag}")
    return "\n".join(out)


def fact_timeline(spec: PersonaSpec, world: WorldState) -> str:
    out = []
    builder = qa.QABuilder(spec, world)
    stated = {s.id: s for s in world.statements}
    for f in spec.facts:
        parts = [f"{v} (d{s}-{'end' if e == spec.num_days else e})" for v, s, e in builder._intervals(f.entity)]
        sep = " → " if f.cardinality == "single" else "; "
        out.append(f"  {f.entity:<17} [{f.cardinality}] " + (sep.join(parts) if parts else "(none)"))
        for c in f.changes:
            st = stated[c.id]
            if c.day > 1 and st.retrospective:
                out.append(f"{'':21}{c.id}: effective d{c.day}, stated d{st.stated_day} (retrospective)")
    for e in spec.events:
        states = " → ".join(f"{s.status} d{s.day}" for s in e.states)
        out.append(f"  event {e.chain_id:<15} {states}")
    for p in spec.other_people:
        for fact in p.facts:
            out.append(f"  other: {p.person} ({p.relation}) d{fact.day}: {fact.text}")
    return "\n".join(out)


def qa_summary(items, n_samples: int, seed: int) -> str:
    by_type = Counter(q.type for q in items)
    lines = [f"  total {len(items)}"]
    lines += [f"  {t:<20} {by_type[t]}" for t in qa.TYPE_ORDER if by_type[t]]
    dist = qa.answer_distribution(items)
    worst = sorted(((c.most_common(1)[0][1] / sum(c.values()), f"{t}/{d}", sum(c.values()))
                    for (t, d), c in dist.items() if sum(c.values()) >= config.GUARD_MIN_GROUP_SIZE),
                   reverse=True)[:5]
    lines.append("  highest top-answer shares: " + ", ".join(f"{g} {s:.0%} (n={n})" for s, g, n in worst))
    lines.append(f"\n  {n_samples} random QA items:")
    for q in random.Random(seed).sample(items, min(n_samples, len(items))):
        tags = f" [{', '.join(q.tags)}]" if q.tags else ""
        lines.append(f"  • ({q.type}/{q.domain}){tags}\n    Q: {q.question}\n    A: {q.answer}")
    return "\n".join(lines)


# ── Commands ────────────────────────────────────────────────────────

def run_user(path: Path, ladder, write_world: bool, n_samples: int) -> bool:
    spec = load_spec(path, ladder)
    report: list[str] = []
    say = report.append
    say(f"\n{'═' * 78}\n{spec.user_id.upper()} - {spec.name} (seed {spec.seed}, {spec.num_days} days "
        f"from {spec.start_date})\n{'═' * 78}")
    errors = checks.check_spec(spec)
    world = simulator.simulate(spec) if not errors else None
    rows = []
    if world:
        errors += checks.check_world(spec, world)
        m = checks.measure_knobs(spec, world)
        rows = checks.compare_knobs(spec, m)
        errors += checks.knob_errors(rows)
    pool = qa.build_qa(spec, world) if world else []
    items = qa.curate(pool, config.QA_CURATED, spec.seed)
    errors += qa.guard_errors(pool) + qa.guard_errors(items)
    if rows:
        say("\nDifficulty knobs (measured vs ladder):\n" + knobs_table(rows))
    if world:
        say("\nPreference timeline:" + domain_timeline(spec, world))
        say("\nFacts and events:\n" + fact_timeline(spec, world))
        say("\nQA:\n" + qa_summary(items, n_samples, spec.seed))
    if errors:
        say(f"\n✗ {len(errors)} error(s) - nothing written:")
        report.extend(f"  - {e}" for e in errors)
        print("\n".join(report))
        return False

    out = config.OUTPUT_DIR / spec.user_id
    if write_world:
        write_json(out / "world_state.json", world.model_dump(mode="json"))
    write_json(out / "qa_pairs.json", [q.model_dump() for q in items])
    write_json(out / "qa_pool.json", [q.model_dump() for q in pool])
    session_days = [d for d in world.days if d.has_session]
    write_json(out / "stats.json", {
        "user_id": spec.user_id,
        "knobs": {r.knob: {"ladder": r.target, "measured": r.measured, "ok": r.ok} for r in rows},
        "sessions": len(session_days),
        "mentions_total": sum(p.mentioned for d in world.days for p in d.preferences.values()),
        "mention_reasons": dict(Counter(p.mention_reason for d in world.days for p in d.preferences.values()
                                        if p.mention_reason)),
        "retrospective_statements": sum(s.retrospective for s in world.statements),
        "qa": qa.qa_stats(items),
        "qa_pool": {"total": len(pool), "by_type": dict(Counter(q.type for q in pool))},
    })
    say(f"\n✓ all checks passed - wrote {out.relative_to(config.REPO_DIR)}/")
    (out / "simulate_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return True


def write_fidelity(spec: PersonaSpec) -> dict:
    from . import conversation
    out = config.OUTPUT_DIR / spec.user_id
    fid = conversation.fidelity(spec.user_id)
    write_json(out / "fidelity.json", fid)
    write_json(out / "conversations.json", conversation.merge(spec))
    print(json.dumps({k: v for k, v in fid.items() if k != "flagged"}, indent=2))
    if fid["flagged"]:
        print(f"flagged sessions ({len(fid['flagged'])}): {', '.join(fid['flagged'][:20])}")
    return fid


def run_generate(args, ladder) -> int:
    from . import conversation
    spec = load_spec(config.PERSONA_DIR / f"{args.user}.yaml", ladder)
    errors = checks.check_spec(spec)
    world = simulator.simulate(spec)
    errors += checks.check_world(spec, world)
    if errors:
        print("refusing to generate — the world state has errors:\n" + "\n".join(errors))
        return 1
    days = conversation.parse_days(args.days, spec.num_days)
    asyncio.run(conversation.generate(spec, world, days, args.model, args.validator_model, args.concurrency))
    write_fidelity(spec)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generator_v2.run")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("simulate", "qa"):
        p = sub.add_parser(name)
        p.add_argument("--user", default="all")
        p.add_argument("--samples", type=int, default=20)
    g = sub.add_parser("generate")
    g.add_argument("--user", required=True)
    g.add_argument("--days", help="e.g. 1-183 (default: all)")
    g.add_argument("--model", default=config.CONVERSATION_MODEL)
    g.add_argument("--validator-model", default=config.VALIDATOR_MODEL)
    g.add_argument("--concurrency", type=int, default=config.CONCURRENCY)
    f = sub.add_parser("fidelity")
    f.add_argument("--user", default="all")
    args = parser.parse_args(argv)
    ladder = load_ladder(config.LADDER_PATH)
    if args.command == "generate":
        return run_generate(args, ladder)
    if args.command == "fidelity":
        for path in persona_paths(args.user):
            write_fidelity(load_spec(path, ladder))
        return 0
    ok = all([run_user(path, ladder, args.command == "simulate", args.samples)
              for path in persona_paths(args.user)])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
