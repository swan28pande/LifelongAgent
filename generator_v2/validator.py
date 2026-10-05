"""LLM extractor → structured claims → diff against the day's ground truth, in code."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from . import config, rules
from .llm import LLM
from .schema import DayState, PersonaSpec, WorldState

SYSTEM = (config.PACKAGE_DIR / "prompts" / "validator_system.txt").read_text()


@dataclass
class Check:
    """What a session must contain, derived from the world state only."""
    preferences: dict[str, tuple[str, list[str]]]   # domain → (expected, options)
    statements: dict[str, str]                        # id → text
    causes: dict[str, dict]                           # id → {topic, event, instruction, cause_day}
    known: list[str] = field(default_factory=list)
    date: str = ""


def options_for(spec: PersonaSpec, world: WorldState, domain: str) -> list[str]:
    pref = next(p for p in spec.preferences if p.domain == domain)
    vals = set().union(*(rules.values_of(r.rule) for r in world.regimes_for(domain)))
    vals |= {o.value for o in pref.exception_pool} | {e.value for e in pref.exceptions}
    return sorted(vals)


def statement_text(spec: PersonaSpec, s) -> str:
    if s.kind == "background_fact":
        label = next(f.label for f in spec.facts if f.entity == s.entity)
        return f"{spec.name}'s {label}: {s.value}"
    text = s.text
    if s.retrospective:
        text += f" (this happened on {spec.date_of(s.effective_day).isoformat()})"
    return text


def known_events(world: WorldState, day: DayState) -> list[str]:
    """Previously disclosed updates, including dated fact-change backstory."""
    return [f"{s.text} (happened on {(world.start_date + dt.timedelta(days=s.effective_day - 1)).isoformat()})"
            for s in world.statements
            if s.kind in ("fact_change", "event_update", "other_person", "distractor")
            and s.stated_day is not None and s.stated_day < day.day]


def build_check(spec: PersonaSpec, world: WorldState, day: DayState) -> Check:
    nouns = {p.domain: p.noun for p in spec.preferences}
    prefs = {dom: (p.value if p.mentioned else "not mentioned", options_for(spec, world, dom))
             for dom, p in day.preferences.items()}
    today = (day.facts_to_state_today + day.event_updates_today + day.other_people_today
             + day.distractors_today)
    statements = {s.id: statement_text(spec, s) for s in today}
    regimes = {r.id: r for r in world.regimes}
    causes = {}
    for i, ci in enumerate(day.cause_instructions):
        reg = regimes[ci.regime_id]
        causes[f"c{i}"] = {"topic": nouns[ci.domain], "event": ci.cause_text or "(none)",
                           "instruction": ci.instruction,
                           "on_regime_day": reg.first_mention_day == day.day}
    known = [f"{e}: {', '.join(v)}" for e, v in day.known_facts.items() if v]
    known += [f"earlier: {t}" for t in known_events(world, day)]
    return Check(prefs, statements, causes, known, day.date.isoformat())


def validator_prompt(spec: PersonaSpec, check: Check, turns: list[dict]) -> str:
    nouns = {p.domain: p.noun for p in spec.preferences}
    lines = [f"DATE: {check.date}", f"USER: {spec.name}", "", "PREFERENCE TOPICS:"]
    for dom, (_, options) in check.preferences.items():
        lines.append(f"- {dom} ({nouns[dom]}). OPTIONS: " + "; ".join(options))
    lines += ["", "STATEMENTS:"] + [f"- {sid}: {text}" for sid, text in check.statements.items()]
    if not check.statements:
        lines.append("(none)")
    lines += ["", "CAUSES:"] + [f"- {cid}: topic = {c['topic']}; event = {c['event']}"
                                 for cid, c in check.causes.items()]
    if not check.causes:
        lines.append("(none)")
    lines += ["", "KNOWN FACTS about the user:"] + [f"- {k}" for k in check.known] + ["", "CONVERSATION:"]
    lines += [f"{t['speaker']}: {t['text']}" for t in turns]
    return "\n".join(lines)


def diff(check: Check, extracted: dict) -> list[str]:
    fails = []
    got_prefs = {k.lower(): str(v).strip().lower() for k, v in (extracted.get("preferences") or {}).items()}
    for dom, (expected, _) in check.preferences.items():
        got = got_prefs.get(dom.lower(), "missing")
        if got != expected.lower():
            kind = ("pref_absent" if expected != "not mentioned" and got in ("not mentioned", "missing")
                    else "pref_leak" if expected == "not mentioned" else "pref_wrong")
            fails.append(f"{kind}: {dom} should be '{expected}', conversation gives '{got}'")
    stated = extracted.get("statements") or {}
    for sid, text in check.statements.items():
        if stated.get(sid) is not True:
            fails.append(f"statement_missing: {sid} ({text})")
    causes = extracted.get("causes") or {}
    for cid, c in check.causes.items():
        got = causes.get(cid) or {}
        ins = c["instruction"]
        if ins == "link_explicitly" and not got.get("linked"):
            fails.append(f"cause_not_linked: {c['topic']} change must be explained by '{c['event']}'")
        elif ins == "mention_without_link" and got.get("linked"):
            fails.append(f"cause_linked: {c['topic']} must not be connected to '{c['event']}'")
        elif ins == "do_not_mention" and got.get("reason_given"):
            fails.append(f"reason_given: no reason may be given for today's {c['topic']}")
    for inv in extracted.get("inventions") or []:
        fails.append(f"invention: {inv}")
    return fails


async def validate(llm: LLM, spec: PersonaSpec, check: Check, turns: list[dict], tag: str) -> tuple[list[str], dict]:
    extracted, _ = await llm.json(SYSTEM, validator_prompt(spec, check, turns), tag)
    return diff(check, extracted), extracted
