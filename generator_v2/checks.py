"""Invariant checks on persona specs and simulated world states.

Every function returns a list of human-readable errors; an empty list means the check
passed. `run.py` refuses to write outputs while any error remains.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from . import config
from .schema import ConditionalRule, NestedRule, PersonaSpec, Rule, WorldState
from .simulator import raw_mention_rate, reference_registry

MIN_REGIME_DAYS = 14
SEASONS = {"winter", "spring", "summer", "fall"}


# ── Spec checks ─────────────────────────────────────────────────────

def _rule_is_total(rule: "Rule | str", where: str) -> list[str]:
    errs = []
    if isinstance(rule, ConditionalRule):
        keys = set(rule.cases)
        need = {"is_workday": {"true", "false"}, "season": SEASONS}.get(rule.condition)
        if "default" not in keys and (need is None or not need <= keys):
            errs.append(f"{where}: conditional on {rule.condition} lacks cases {sorted((need or set()) - keys)} and has no default")
        for k, c in rule.cases.items():
            if not isinstance(c, str):
                errs += _rule_is_total(c, f"{where}[{k}]")
    if isinstance(rule, NestedRule):
        for i, b in enumerate(rule.blocks):
            errs += _rule_is_total(b, f"{where}.block{i}")
    return errs


def check_spec(spec: PersonaSpec) -> list[str]:
    errs: list[str] = []
    N = spec.num_days
    registry = reference_registry(spec)
    max_lag = spec.difficulty.lag_days.max if spec.difficulty and spec.difficulty.lag_days.max else 10

    ids = ([c.id for f in spec.facts for c in f.changes]
           + [s.id for e in spec.events for s in e.states] + [e.chain_id for e in spec.events]
           + [r.id for p in spec.preferences for r in p.regimes] + [d.id for d in spec.distractors])
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        errs.append(f"duplicate ids: {dupes}")

    def in_range(day: int, what: str) -> None:
        if not 1 <= day <= N:
            errs.append(f"{what}: day {day} outside 1..{N}")

    # Facts: ops respect cardinality, removes target active values.
    for f in spec.facts:
        changes = sorted(f.changes, key=lambda c: c.day)
        if f.cardinality == "single":
            if not changes or changes[0].day != 1 or changes[0].op != "set":
                errs.append(f"fact {f.entity}: single-valued facts must start with a set on day 1")
            if any(c.op != "set" for c in changes):
                errs.append(f"fact {f.entity}: single-valued facts only use op=set")
        elif any(c.op == "set" for c in changes):
            errs.append(f"fact {f.entity}: multi-valued facts use add/remove, not set")
        active: set[str] = set()
        for c in changes:
            in_range(c.day, c.id)
            if c.op == "remove" and c.value not in active:
                errs.append(f"{c.id}: removes '{c.value}' which is not active on day {c.day}")
            if c.op == "add" and c.value in active:
                errs.append(f"{c.id}: adds '{c.value}' which is already active")
            if c.op == "set":
                active = {c.value}
            elif c.op == "add":
                active.add(c.value)
            else:
                active.discard(c.value)
            if c.cause_event:
                target = registry.get(c.cause_event)
                if target is None:
                    errs.append(f"{c.id}: unknown cause_event '{c.cause_event}'")
                elif target.day > c.day:
                    errs.append(f"{c.id}: cause_event happens after the change")

    for e in spec.events:
        days = [s.day for s in e.states]
        if days != sorted(days):
            errs.append(f"event {e.chain_id}: states out of order")
        for s in e.states:
            in_range(s.day, s.id)
        terminal = [i for i, s in enumerate(e.states) if s.status in ("happened", "cancelled")]
        if terminal and e.states[-1].status == "planned":
            errs.append(f"event {e.chain_id}: a planned state follows a terminal one")

    # Regimes: ordered, non-overlapping, long enough, causes consistent with lags.
    domains = {p.domain for p in spec.preferences}
    for p in spec.preferences:
        regs = p.regimes
        if [r.start for r in regs] != sorted({r.start for r in regs}):
            errs.append(f"{p.domain}: regime starts must be strictly increasing")
            continue
        if regs[0].start != 1 or regs[0].cause is not None or regs[0].temporary:
            errs.append(f"{p.domain}: the first regime must start on day 1, uncaused, not temporary")
        if spec.difficulty and spec.difficulty.exception_rate > 0 and not p.exception_pool and not p.exceptions:
            errs.append(f"{p.domain}: exception rate > 0 but no exception_pool")
        prev_rule = None
        for i, r in enumerate(regs):
            nxt = regs[i + 1].start if i + 1 < len(regs) else N + 1
            errs += _rule_is_total(r.rule, r.id)
            if r.temporary:
                if r.end is None:
                    errs.append(f"{r.id}: temporary regime needs an end")
                    continue
                if not r.start <= r.end < nxt - 1:
                    errs.append(f"{r.id}: temporary end {r.end} must leave room to revert before day {nxt}")
                elif nxt - 1 - r.end < MIN_REGIME_DAYS:
                    errs.append(f"{r.id}: reversion lasts {nxt - 1 - r.end} days (< {MIN_REGIME_DAYS})")
                length = r.end - r.start + 1
            else:
                if r.end is not None:
                    errs.append(f"{r.id}: only temporary regimes may set end")
                length = nxt - r.start
            if length < MIN_REGIME_DAYS:
                errs.append(f"{r.id}: lasts {length} days (< {MIN_REGIME_DAYS})")
            if prev_rule is not None and r.rule == prev_rule:
                errs.append(f"{r.id}: same rule as the regime before it - not a real shift")
            prev_rule = r.rule if not r.temporary else prev_rule
            if r.cause:
                target = registry.get(r.cause.event_ref)
                if target is None:
                    errs.append(f"{r.id}: unknown event_ref '{r.cause.event_ref}'")
                    continue
                if r.cause.day is not None and r.cause.day != target.day:
                    errs.append(f"{r.id}: cause.day {r.cause.day} != referenced day {target.day}")
                if target.day > r.start:
                    errs.append(f"{r.id}: cause on day {target.day} is after the shift on day {r.start}")
                if target.day + r.cause.lag_days != r.start:
                    errs.append(f"{r.id}: start {r.start} != cause day {target.day} + lag {r.cause.lag_days}")
            if r.revert_reason and not r.temporary:
                errs.append(f"{r.id}: revert_reason only applies to temporary regimes")

    # Distractors must not sit on an uncaused shift in the same domain unless flagged.
    uncaused = {(p.domain, r.start) for p in spec.preferences for r in p.regimes[1:] if r.cause is None}
    cause_refs = {registry[r.cause.event_ref].id for p in spec.preferences for r in p.regimes
                  if r.cause and r.cause.event_ref in registry}
    for d in spec.distractors:
        in_range(d.day, d.id)
        if d.domain not in domains:
            errs.append(f"{d.id}: unknown domain '{d.domain}'")
        if d.id in cause_refs:
            errs.append(f"{d.id}: a distractor cannot be a cause")
        near = [s for dom, s in uncaused if dom == d.domain and abs(s - d.day) <= max_lag]
        if near and not d.confounder:
            errs.append(f"{d.id}: within ±{max_lag} days of an uncaused {d.domain} shift on day {near[0]} "
                        "(set confounder: true if deliberate)")

    fact_entities = {f.entity for f in spec.facts}
    for p in spec.other_people:
        for fact in p.facts:
            in_range(fact.day, f"{p.person} fact")
            if fact.confusable_with and fact.confusable_with not in fact_entities:
                errs.append(f"{p.person}: confusable_with '{fact.confusable_with}' is not a user fact")
    return errs


# ── World checks ────────────────────────────────────────────────────

def check_world(spec: PersonaSpec, world: WorldState) -> list[str]:
    errs: list[str] = []
    if len(world.days) != spec.num_days:
        errs.append(f"world has {len(world.days)} days, expected {spec.num_days}")

    single = [f.entity for f in spec.facts if f.cardinality == "single"]
    for day in world.days:
        for e in single:
            if len(day.active_facts[e]) != 1:
                errs.append(f"day {day.day}: single-valued '{e}' has {day.active_facts[e]}")
                break

    for dom in (p.domain for p in spec.preferences):
        regs = world.regimes_for(dom)
        covered = Counter(d for r in regs for d in range(r.start, r.end + 1))
        bad = [d for d in range(1, spec.num_days + 1) if covered[d] != 1]
        if bad:
            errs.append(f"{dom}: days {bad[:5]}… are covered by {covered[bad[0]]} regimes")

    stated = {s.id: s.stated_day for s in world.statements}
    for s in world.statements:
        if s.stated_day is None:
            errs.append(f"statement {s.id} (day {s.effective_day}) never reaches a session")

    for dom in (p.domain for p in spec.preferences):
        days = [d.day for d in world.days if d.preferences[dom].is_exception]
        close = [(a, b) for a, b in zip(days, days[1:]) if b - a <= config.EXCEPTION_MIN_SPACING]
        if close:
            errs.append(f"{dom}: exceptions on days {close[0]} are too close together to read as one-offs")

    first_instr = Counter(ci.regime_id for d in world.days for ci in d.cause_instructions
                          if d.day == next((r.first_mention_day for r in world.regimes if r.id == ci.regime_id), None))
    registry = reference_registry(spec)
    for r in world.regimes:
        if len(r.mention_days) < config.MIN_REGIME_MENTIONS:
            errs.append(f"{r.id}: observed {len(r.mention_days)} times (< {config.MIN_REGIME_MENTIONS})")
        if r.kind == "initial":
            continue
        if r.first_mention_day is None:
            errs.append(f"{r.id}: never observed")
            continue
        if first_instr[r.id] < 1:
            errs.append(f"{r.id}: no cause instruction on its first observation")
        if r.visibility == "explicit" and r.cause_ref:
            day = stated.get(registry[r.cause_ref].id)
            if day is None or day > r.first_mention_day:
                errs.append(f"{r.id}: explicit cause stated on day {day}, after the link on day {r.first_mention_day}")

    for day in world.days:
        for dom, p in day.preferences.items():
            if p.is_exception and not (day.has_session and p.mentioned):
                errs.append(f"day {day.day} {dom}: exception is not observed")
            if p.is_exception and p.value == p.base_value:
                errs.append(f"day {day.day} {dom}: exception equals the regular value")
        if not day.has_session and (day.cause_instructions or day.facts_to_state_today):
            errs.append(f"day {day.day}: instructions on a day without a session")
    return errs


# ── Knobs ───────────────────────────────────────────────────────────

@dataclass
class KnobRow:
    knob: str
    target: str
    measured: str
    ok: bool


def measure_knobs(spec: PersonaSpec, world: WorldState) -> dict:
    registry = reference_registry(spec)
    shifts = [r for r in world.regimes if r.kind in ("shift", "temporary")]
    per_domain = Counter(r.domain for r in shifts)
    vis = Counter(r.visibility for r in shifts)
    lags = [r.lag_days for r in shifts if r.lag_days is not None]
    ref_domains: dict[str, set[str]] = {}
    for p in spec.preferences:
        for r in p.regimes:
            if r.cause and r.cause.event_ref in registry:
                ref_domains.setdefault(registry[r.cause.event_ref].id, set()).add(p.domain)
    session_days = [d for d in world.days if d.has_session]
    cells = len(session_days) * len(spec.preferences)
    exceptions = sum(p.is_exception for d in world.days for p in d.preferences.values())
    return {
        "preference_domains": len(spec.preferences),
        "rule_types": sorted({r.rule.type for p in spec.preferences for r in p.regimes}),
        "shifts_per_domain": (min(per_domain.values(), default=0), max(per_domain.values(), default=0)),
        "visibility_mix": {k: round(vis[k] / len(shifts), 3) if shifts else 0.0
                           for k in ("explicit", "implicit", "uncaused")},
        "lag_days": (min(lags), max(lags)) if lags else None,
        "temporary_shifts": sum(r.kind == "temporary" for r in shifts),
        "cross_domain_causes": sum(len(v) >= 2 for v in ref_domains.values()),
        "distractors": len(spec.distractors),
        "exception_rate": round(exceptions / cells, 4) if cells else 0.0,
        "p_session": round(len(session_days) / len(world.days), 4),
        "p_mention": round(raw_mention_rate(spec, world), 4),
        "fact_changes": sum(c.day > 1 for f in spec.facts for c in f.changes),
        "other_people": bool(spec.other_people),
        "total_shifts": len(shifts),
        "all_domain_shift_counts": dict(per_domain),
    }


def compare_knobs(spec: PersonaSpec, m: dict) -> list[KnobRow]:
    t = spec.difficulty
    rows = [KnobRow("preference_domains", str(t.preference_domains), str(m["preference_domains"]),
                    m["preference_domains"] == t.preference_domains)]
    used = set(m["rule_types"])
    rows.append(KnobRow("rule_types", f"required {t.rule_types.required}, allowed {t.rule_types.allowed}",
                        ", ".join(m["rule_types"]),
                        set(t.rule_types.required) <= used <= set(t.rule_types.allowed)))
    lo, hi = m["shifts_per_domain"]
    rows.append(KnobRow("shifts_per_domain", str(t.shifts_per_domain), f"{lo}-{hi}" if lo != hi else str(lo),
                        t.shifts_per_domain.contains(lo) and t.shifts_per_domain.contains(hi)))
    mix_ok = all(abs(m["visibility_mix"][k] - t.visibility_mix[k]) <= config.VISIBILITY_MIX_TOLERANCE
                 for k in ("explicit", "implicit", "uncaused"))
    fmt = lambda d: "/".join(f"{round(d[k] * 100)}" for k in ("explicit", "implicit", "uncaused"))
    rows.append(KnobRow("visibility_mix (E/I/U %)", fmt(t.visibility_mix), fmt(m["visibility_mix"]), mix_ok))
    lags = m["lag_days"]
    rows.append(KnobRow("lag_days", str(t.lag_days), f"{lags[0]}-{lags[1]}" if lags else "n/a",
                        lags is None or (t.lag_days.contains(lags[0]) and t.lag_days.contains(lags[1]))))
    for knob in ("temporary_shifts", "fact_changes"):
        rng = getattr(t, knob)
        rows.append(KnobRow(knob, str(rng), str(m[knob]), rng.contains(m[knob])))
    for knob in ("cross_domain_causes", "distractors"):
        rows.append(KnobRow(knob, str(getattr(t, knob)), str(m[knob]), m[knob] == getattr(t, knob)))
    rows.append(KnobRow("exception_rate", f"{t.exception_rate:.2f}", f"{m['exception_rate']:.3f}",
                        abs(m["exception_rate"] - t.exception_rate) <= config.EXCEPTION_RATE_TOLERANCE))
    for knob in ("p_session", "p_mention"):
        rows.append(KnobRow(knob, f"{getattr(t, knob):.2f}", f"{m[knob]:.3f}",
                            abs(m[knob] - getattr(t, knob)) <= config.PROB_TOLERANCE))
    rows.append(KnobRow("other_people", str(t.other_people), str(m["other_people"]),
                        m["other_people"] == t.other_people))
    return rows


def knob_errors(rows: list[KnobRow]) -> list[str]:
    return [f"knob {r.knob}: measured {r.measured}, ladder {r.target}" for r in rows if not r.ok]
