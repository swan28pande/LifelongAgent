"""Persona spec + seed → the full world state for every day. No LLM calls.

Order of operations (each step draws from its own seeded RNG stream, so changing one
step's sampling never reshuffles another's):

    sessions → fact timelines → statement scheduling → regimes → daily values
    → sampled mentions → exceptions → regime-start mentions → top-up mentions
    → cause instructions → day states
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from . import config, rules
from .schema import (
    CauseInstruction, DayState, PersonaSpec, PrefState, ResolvedRegime, Statement, WorldState,
)


def _rng(spec: PersonaSpec, stream: str) -> random.Random:
    return random.Random(f"{spec.seed}:{stream}")


@dataclass(frozen=True)
class RefTarget:
    id: str
    day: int
    text: str


# ── References ──────────────────────────────────────────────────────

def fact_change_text(label: str, op: str, value: str, reason: Optional[str], text: Optional[str]) -> str:
    if text:
        return text.format(value=value)
    base = {"set": f"{label} is now {value}",
            "add": f"{label} now include {value}",
            "remove": f"{label} no longer include {value}"}[op]
    return f"{base} ({reason})" if reason else base


def reference_registry(spec: PersonaSpec) -> dict[str, RefTarget]:
    """Every id a cause may point at: fact changes, event states, and chain ids."""
    reg: dict[str, RefTarget] = {}
    for f in spec.facts:
        for c in f.changes:
            reg[c.id] = RefTarget(c.id, c.day, fact_change_text(f.label, c.op, c.value, c.reason, c.text))
    for e in spec.events:
        for s in e.states:
            reg[s.id] = RefTarget(s.id, s.day, s.text)
        first = min(e.states, key=lambda s: s.day)
        reg.setdefault(e.chain_id, RefTarget(first.id, first.day, first.text))
    return reg


def cause_ids(spec: PersonaSpec, registry: dict[str, RefTarget]) -> set[str]:
    out = set()
    for p in spec.preferences:
        for r in p.regimes:
            if r.cause and r.cause.event_ref in registry:
                out.add(registry[r.cause.event_ref].id)
    for f in spec.facts:
        for c in f.changes:
            if c.cause_event and c.cause_event in registry:
                out.add(registry[c.cause_event].id)
    return out


# ── Facts ───────────────────────────────────────────────────────────

def fact_timeline(spec: PersonaSpec) -> dict[str, list[list[str]]]:
    """entity → per-day (index day-1) list of values that are true that day."""
    out = {}
    for f in spec.facts:
        by_day: dict[int, list] = {}
        for c in sorted(f.changes, key=lambda c: c.day):
            by_day.setdefault(c.day, []).append(c)
        active: list[str] = []
        per_day = []
        for d in range(1, spec.num_days + 1):
            for c in by_day.get(d, []):
                v = c.value
                if c.op == "set":
                    active = [v]
                elif c.op == "add" and v not in active:
                    active = active + [v]
                elif c.op == "remove":
                    active = [x for x in active if x != v]
            per_day.append(list(active))
        out[f.entity] = per_day
    return out


# ── Statements ──────────────────────────────────────────────────────

def build_statements(spec: PersonaSpec, registry: dict[str, RefTarget]) -> list[Statement]:
    causes = cause_ids(spec, registry)
    out: list[Statement] = []
    for f in spec.facts:
        for c in f.changes:
            out.append(Statement(
                id=c.id, kind="background_fact" if c.day == 1 else "fact_change",
                text=fact_change_text(f.label, c.op, c.value, c.reason, c.text),
                entity=f.entity, op=c.op, value=c.value,
                effective_day=c.day, is_cause=c.id in causes,
            ))
    for e in spec.events:
        for s in e.states:
            out.append(Statement(id=s.id, kind="event_update", text=f"[{e.title}] {s.text} ({s.status})",
                                 effective_day=s.day, is_cause=s.id in causes))
    for p in spec.other_people:
        for i, fact in enumerate(p.facts):
            out.append(Statement(id=f"other.{p.person.lower()}.{i}", kind="other_person",
                                 text=f"{p.person} ({spec.name}'s {p.relation}): {fact.text}",
                                 entity=fact.noun, value=fact.value, effective_day=fact.day))
    for d in spec.distractors:
        out.append(Statement(id=d.id, kind="distractor", text=d.text, effective_day=d.day))
    return out


def schedule_statements(statements: list[Statement], session_days: list[int]) -> None:
    """Move every statement to a session day on or after it takes effect, respecting caps."""
    load: dict[int, int] = {}
    background_load: dict[int, int] = {}
    order = sorted(statements, key=lambda s: (s.effective_day, not s.is_cause,
                                              s.kind == "background_fact", s.id))
    for s in order:
        for day in (d for d in session_days if d >= s.effective_day):
            if load.get(day, 0) >= config.MAX_STATEMENTS_PER_SESSION:
                continue
            if s.kind == "background_fact" and background_load.get(day, 0) >= config.BACKGROUND_FACTS_PER_SESSION:
                continue
            s.stated_day = day
            load[day] = load.get(day, 0) + 1
            if s.kind == "background_fact":
                background_load[day] = background_load.get(day, 0) + 1
            break
        s.retrospective = (s.stated_day is not None and s.stated_day > s.effective_day
                           and s.kind != "background_fact")


def known_timeline(spec: PersonaSpec, statements: list[Statement]) -> dict[str, list[list[str]]]:
    """What the user has told the assistant so far (facts become known when stated)."""
    by_day: dict[int, list[Statement]] = {}
    for s in statements:
        if s.entity and s.kind in ("background_fact", "fact_change") and s.stated_day:
            by_day.setdefault(s.stated_day, []).append(s)
    entities = [f.entity for f in spec.facts]
    active = {e: [] for e in entities}
    out = {e: [] for e in entities}
    for d in range(1, spec.num_days + 1):
        for s in sorted(by_day.get(d, []), key=lambda s: (s.effective_day, s.id)):
            cur = active[s.entity]
            if s.op == "set":
                active[s.entity] = [s.value]
            elif s.op == "add" and s.value not in cur:
                active[s.entity] = cur + [s.value]
            elif s.op == "remove":
                active[s.entity] = [x for x in cur if x != s.value]
        for e in entities:
            out[e].append(list(active[e]))
    return out


# ── Regimes ─────────────────────────────────────────────────────────

def resolve_regimes(spec: PersonaSpec, registry: dict[str, RefTarget]) -> list[ResolvedRegime]:
    out: list[ResolvedRegime] = []
    for p in spec.preferences:
        regs = sorted(p.regimes, key=lambda r: r.start)
        base: Optional[ResolvedRegime] = None
        for i, r in enumerate(regs):
            nxt = regs[i + 1].start if i + 1 < len(regs) else spec.num_days + 1
            kind = "initial" if i == 0 else ("temporary" if r.temporary else "shift")
            end = r.end if r.temporary and r.end is not None else nxt - 1
            cause_ref = cause_day = cause_text = lag = None
            if r.cause:
                target = registry.get(r.cause.event_ref)
                cause_ref, lag = r.cause.event_ref, r.cause.lag_days
                if target:
                    cause_day, cause_text = target.day, target.text
            visibility = "initial" if i == 0 else (r.cause.visibility if r.cause else "uncaused")
            resolved = ResolvedRegime(
                id=r.id, domain=p.domain, kind=kind, start=r.start, end=end, anchor=r.start,
                rule=r.rule, label=r.label or rules.short_label(r.rule), visibility=visibility,
                cause_ref=cause_ref, cause_day=cause_day, cause_text=cause_text, lag_days=lag,
                revert_reason=r.revert_reason,
            )
            out.append(resolved)
            if kind != "temporary":
                base = resolved
            elif base is not None and end + 1 <= nxt - 1:
                out.append(ResolvedRegime(
                    id=f"{r.id}.revert", domain=p.domain, kind="reversion", start=end + 1, end=nxt - 1,
                    anchor=base.anchor, rule=base.rule, label=base.label,
                    visibility="explicit" if r.revert_reason else "uncaused",
                    cause_text=r.revert_reason, reverts_to=base.id,
                ))
    return out


def regime_on(regimes: list[ResolvedRegime], day: int) -> ResolvedRegime:
    for r in regimes:
        if r.start <= day <= r.end:
            return r
    raise ValueError(f"no regime covers day {day}")


def _spread(candidates: list[int], k: int) -> list[int]:
    if k <= 0 or not candidates:
        return []
    if k >= len(candidates):
        return list(candidates)
    if k == 1:
        return [candidates[len(candidates) // 2]]
    return sorted({candidates[round(i * (len(candidates) - 1) / (k - 1))] for i in range(k)})


# ── Main entry point ────────────────────────────────────────────────

def simulate(spec: PersonaSpec) -> WorldState:
    diff = spec.difficulty
    if diff is None:
        raise ValueError("spec has no difficulty block; load it with schema.load_spec")
    N = spec.num_days
    all_days = range(1, N + 1)
    dates = {d: spec.date_of(d) for d in all_days}
    domains = [p.domain for p in spec.preferences]
    pref_by_domain = {p.domain: p for p in spec.preferences}

    # Sessions. Day 1 always has one; explicit exception days are forced to have one.
    r = _rng(spec, "sessions")
    draws = {d: r.random() for d in all_days}
    forced = {e.day for p in spec.preferences for e in p.exceptions}
    has_session = {d: d == 1 or d in forced or draws[d] < diff.p_session for d in all_days}
    session_days = [d for d in all_days if has_session[d]]

    registry = reference_registry(spec)
    truth = fact_timeline(spec)
    statements = build_statements(spec, registry)
    schedule_statements(statements, session_days)
    known = known_timeline(spec, statements)

    regimes = resolve_regimes(spec, registry)
    by_domain = {dom: [x for x in regimes if x.domain == dom] for dom in domains}

    # Daily ground-truth values before exceptions.
    active_regime: dict[tuple[int, str], ResolvedRegime] = {}
    base_value: dict[tuple[int, str], str] = {}
    for d in all_days:
        city = (truth.get("city") or [[None]] * N)[d - 1]
        ctx = rules.context_for(dates[d], city[0] if city else None)
        for dom in domains:
            reg = regime_on(by_domain[dom], d)
            active_regime[(d, dom)] = reg
            base_value[(d, dom)] = rules.evaluate(reg.rule, d, dates[d], reg.anchor, ctx)

    # Sampled mentions (raw rate - measured against the ladder before any forcing).
    r = _rng(spec, "mentions")
    mentioned: dict[tuple[int, str], str] = {}
    for d in session_days:
        for dom in domains:
            if r.random() < diff.p_mention:
                mentioned[(d, dom)] = "sampled"

    # Exceptions: explicit ones first, then pool draws to hit the ladder rate.
    exceptions: dict[tuple[int, str], tuple[str, str]] = {}
    for p in spec.preferences:
        for e in p.exceptions:
            exceptions[(e.day, p.domain)] = (e.value, e.reason)
    boundaries = {dom: {x.start for x in by_domain[dom] if x.start > 1} for dom in domains}
    gap = config.EXCEPTION_BOUNDARY_GAP
    eligible = [
        (d, dom) for d in session_days for dom in domains
        if (d, dom) not in exceptions
        and not any(abs(d - b) <= gap for b in boundaries[dom])
        and any(o.value != base_value[(d, dom)] for o in pref_by_domain[dom].exception_pool)
    ]
    target = round(diff.exception_rate * len(session_days) * len(domains)) - len(exceptions)
    r = _rng(spec, "exceptions")
    r.shuffle(eligible)
    taken = {dom: [d for d, x in exceptions if x == dom] for dom in domains}
    picked = []
    for d, dom in eligible:
        if len(picked) >= target:
            break
        if all(abs(d - o) > config.EXCEPTION_MIN_SPACING for o in taken[dom]):
            picked.append((d, dom))
            taken[dom].append(d)
    for cell in sorted(picked):
        options = [o for o in pref_by_domain[cell[1]].exception_pool if o.value != base_value[cell]]
        choice = r.choice(options)
        exceptions[cell] = (choice.value, choice.reason)
    for cell in exceptions:
        mentioned.setdefault(cell, "exception")

    # A new regime is usually observed on its first session day.
    r = _rng(spec, "regime_start")
    for reg in regimes:
        if reg.kind == "initial":
            continue
        first = next((d for d in session_days if reg.start <= d <= reg.end
                      and (d, reg.domain) not in exceptions), None)
        if first is not None and r.random() < diff.p_regime_start_mention:
            mentioned.setdefault((first, reg.domain), "regime_start")

    # Top up so every regime is observed often enough to be learnable.
    for reg in regimes:
        span = [d for d in session_days if reg.start <= d <= reg.end and (d, reg.domain) not in exceptions]
        seen = [d for d in span if (d, reg.domain) in mentioned]
        missing = config.MIN_REGIME_MENTIONS - len(seen)
        for d in _spread([d for d in span if (d, reg.domain) not in mentioned], missing):
            mentioned[(d, reg.domain)] = "top_up"

    for reg in regimes:
        reg.mention_days = sorted(d for d in all_days if (d, reg.domain) in mentioned
                                  and reg.start <= d <= reg.end and (d, reg.domain) not in exceptions)
        reg.first_mention_day = reg.mention_days[0] if reg.mention_days else None

    # Cause instructions: on the first observation of each new regime, and on the day an
    # implicit cause is stated (state it, but never tie it to the change).
    instructions: dict[int, list[CauseInstruction]] = {}
    stated_on = {s.id: s.stated_day for s in statements}
    for reg in regimes:
        if reg.kind == "initial" or reg.first_mention_day is None:
            continue
        kind = {"explicit": "link_explicitly", "implicit": "mention_without_link",
                "uncaused": "do_not_mention"}[reg.visibility]
        instructions.setdefault(reg.first_mention_day, []).append(CauseInstruction(
            domain=reg.domain, regime_id=reg.id, instruction=kind, cause_ref=reg.cause_ref,
            cause_text=reg.cause_text if reg.visibility != "uncaused" else None,
            new_routine=reg.label,
        ))
        if reg.visibility == "implicit" and reg.cause_ref:
            day = stated_on.get(registry[reg.cause_ref].id)
            if day is not None:
                instructions.setdefault(day, []).append(CauseInstruction(
                    domain=reg.domain, regime_id=reg.id, instruction="mention_without_link",
                    cause_ref=reg.cause_ref, cause_text=reg.cause_text, new_routine=reg.label,
                ))

    # Day states.
    stated_by_day: dict[int, list[Statement]] = {}
    for s in statements:
        if s.stated_day is not None:
            stated_by_day.setdefault(s.stated_day, []).append(s)
    chains = {e.chain_id: e for e in spec.events}
    r = _rng(spec, "followups")
    days: list[DayState] = []
    for d in all_days:
        prefs = {}
        for dom in domains:
            exc = exceptions.get((d, dom))
            prefs[dom] = PrefState(
                value=exc[0] if exc else base_value[(d, dom)],
                base_value=base_value[(d, dom)],
                regime_id=active_regime[(d, dom)].id,
                is_exception=exc is not None,
                exception_reason=exc[1] if exc else None,
                mentioned=(d, dom) in mentioned,
                mention_reason=mentioned.get((d, dom)),
            )
        today = stated_by_day.get(d, [])
        known_today = {e: v[d - 1] for e, v in known.items()}
        followups = [f"{e}: {', '.join(v)}" for e, v in known_today.items() if v]
        for c in chains.values():
            told = [s for s in c.states if stated_on.get(s.id) is not None and stated_on[s.id] < d]
            if told and max(told, key=lambda s: s.day).status == "planned":
                followups.append(f"open plan - {c.title}: {max(told, key=lambda s: s.day).text}")
        k = min(len(followups), r.randint(2, 4))
        days.append(DayState(
            day=d, date=dates[d], weekday=dates[d].strftime("%A"), has_session=has_session[d],
            preferences=prefs,
            active_facts={e: v[d - 1] for e, v in truth.items()},
            known_facts=known_today,
            facts_to_state_today=[s for s in today if s.kind in ("background_fact", "fact_change")],
            event_updates_today=[s for s in today if s.kind == "event_update"],
            other_people_today=[s for s in today if s.kind == "other_person"],
            distractors_today=[s for s in today if s.kind == "distractor"],
            cause_instructions=instructions.get(d, []) if has_session[d] else [],
            followup_candidates=sorted(r.sample(followups, k)) if has_session[d] else [],
            forbidden_inventions=list(config.FORBIDDEN_INVENTIONS) if has_session[d] else [],
        ))

    return WorldState(
        user_id=spec.user_id, name=spec.name, seed=spec.seed, start_date=spec.start_date,
        num_days=N, regimes=regimes, statements=statements, days=days,
    )


def raw_mention_rate(spec: PersonaSpec, world: WorldState) -> float:
    cells = [p for day in world.days if day.has_session for p in day.preferences.values()]
    sampled = [p for p in cells if p.mention_reason == "sampled"]
    return len(sampled) / len(cells) if cells else 0.0
