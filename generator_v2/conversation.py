"""World state → one validated session per session day. Async, parallel and resumable:
each finished session is written to sessions/{date}.json and skipped on the next run.

A session is generated from that day's DayState and the facts known before it, never from
earlier generated text, so days are independent."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

from . import config
from .llm import LLM
from .schema import DayState, PersonaSpec, WorldState
from .validator import build_check, known_events, statement_text, validate

SYSTEM = (config.PACKAGE_DIR / "prompts" / "conversation_system.txt").read_text()

WORDING = {
    "exact_terms": "EXACT — the user uses the exact wording of each value (value 'oat milk latte' → says 'oat milk latte').",
    "light_paraphrase": "LIGHT PARAPHRASE — mostly the value's own words; small natural variations are fine if the value is unmistakable.",
    "paraphrase": "PARAPHRASE — the user describes each value in their own words; it must still be unambiguous which value it is.",
    "heavy_paraphrase": "HEAVY PARAPHRASE — never the value's exact wording; describe it indirectly but unambiguously.",
    "heavy_paraphrase_with_other_people": "HEAVY PARAPHRASE — never the value's exact wording; describe it indirectly but "
                                          "unambiguously. Also mention in passing what other people in the user's life "
                                          "chose, clearly attributed to them.",
}


def writer_prompt(spec: PersonaSpec, world: WorldState, day: DayState, feedback: list[str]) -> str:
    nouns = {p.domain: p.noun for p in spec.preferences}
    before = world.days[day.day - 2].known_facts if day.day > 1 else {}
    regimes = {r.id: r for r in world.regimes}
    lines = [f"DATE: {day.weekday}, {day.date.isoformat()}", f"USER: {spec.name}",
             f"WORDING LEVEL: {WORDING[spec.difficulty.wording]}", "",
             f"WHAT THE ASSISTANT ALREADY KNOWS ABOUT {spec.name.upper()}:"]
    known = [f"- {e.replace('_', ' ')}: {', '.join(v)}" for e, v in before.items() if v]
    lines += known or ["- (nothing yet — this is one of the first conversations)"]
    earlier = known_events(world, day)
    if earlier:
        lines += ["", "EVENTS FROM EARLIER CONVERSATIONS (known; may be referred to, never changed):"]
        lines += [f"- {t}" for t in earlier]

    lines += ["", "TODAY'S CHOICES — the user mentions each:"]
    mentioned = {d: p for d, p in day.preferences.items() if p.mentioned}
    for dom, p in mentioned.items():
        line = f"- {nouns[dom]}: {p.value}"
        if p.is_exception:
            line += f"  (ONE-OFF: instead of the usual {p.base_value}, because {p.exception_reason})"
        lines.append(line)
    absent = [d for d in day.preferences if d not in mentioned]
    lines += ["", "MUST NOT COME UP: " + (", ".join(f"{d} ({nouns[d]})" for d in absent)
                                             if absent else "(nothing)")]
    if absent:
        lines.append("Neither speaker may mention, ask about, or allude to these topics, "
                     "including a past or future choice or another person's choice. "
                     "Known hobbies and jobs do not grant permission to discuss an absent topic.")
    lines.append("Keep each of today's choices distinct from similar choices: do not add "
                 "details that change the specified value into another option. In particular, "
                 "'no commute' does not mean 'work from home', and an unqualified stir-fry "
                 "must not be described as a vegetarian stir-fry.")

    required = day.facts_to_state_today + day.event_updates_today
    lines += ["", "REQUIRED UPDATES — the user states each explicitly:"]
    for s in required:
        text = statement_text(spec, s)
        if s.kind == "background_fact":
            pass
        elif s.retrospective:
            text += f" — happened {(day.day - s.effective_day)} day(s) ago; describe it in the past"
        else:
            text += " — this happened today (or is being announced today); do not place it on another day"
        lines.append(f"- {text}")
    if not required:
        lines.append("- (none)")

    if day.cause_instructions:
        lines += ["", "CAUSE HANDLING:"]
        for ci in day.cause_instructions:
            reg = regimes[ci.regime_id]
            noun = nouns[ci.domain]
            if ci.instruction == "link_explicitly":
                verb = "gone back to their earlier" if reg.kind == "reversion" else "changed their"
                lines.append(f"- {noun}: LINK — the user says they have recently {verb} {noun} routine, "
                             f"because: {ci.cause_text}.")
            elif ci.instruction == "mention_without_link":
                if reg.first_mention_day == day.day:
                    lines.append(f"- {noun}: NO LINK — report today's {noun} without saying the routine changed, "
                                 f"and without connecting it to: {ci.cause_text}.")
                else:
                    lines.append(f"- NO LINK — '{ci.cause_text}' is mentioned today, but nobody connects it "
                                 f"to the user's {noun}.")
            else:
                lines.append(f"- {noun}: NO REASON — the user reports today's {noun} with no reason given or hinted.")

    if day.other_people_today:
        lines += ["", "NEWS ABOUT OTHER PEOPLE (clearly about them, not the user):"]
        lines += [f"- {s.text}" for s in day.other_people_today]
    if day.distractors_today:
        lines += ["", "DISTRACTORS (small talk, never connected to the user's routines):"]
        lines += [f"- {s.text}" for s in day.distractors_today]
    if day.followup_candidates:
        lines += ["", "FOLLOW-UP TOPICS the assistant may ask about:"]
        lines += [f"- {f}" for f in day.followup_candidates]
    lines += ["", "NEVER INVENT: " + "; ".join(day.forbidden_inventions)]
    if feedback:
        lines += ["", "YOUR PREVIOUS ATTEMPT FAILED THESE CHECKS — fix every one:"]
        lines += [f"- {f}" for f in feedback]
    return "\n".join(lines)


def shape_errors(turns) -> list[str]:
    if not isinstance(turns, list) or not turns:
        return ["format: no turns returned"]
    errs = []
    if not config.MIN_TURNS <= len(turns) <= config.MAX_TURNS:
        errs.append(f"format: {len(turns)} turns, need {config.MIN_TURNS}-{config.MAX_TURNS}")
    if any(not isinstance(t, dict) or t.get("speaker") not in ("assistant", "user") or not t.get("text")
           for t in turns):
        errs.append("format: every turn needs speaker 'assistant' or 'user' and non-empty text")
    return errs


async def generate_day(spec: PersonaSpec, world: WorldState, day: DayState, writer: LLM,
                       checker: LLM, out: Path) -> dict:
    check = build_check(spec, world, day)
    best: dict | None = None
    feedback: list[str] = []
    for attempt in range(1, config.MAX_RETRIES + 1):
        tag = f"{day.date.isoformat()}#{attempt}"
        try:
            result, _ = await writer.json(SYSTEM, writer_prompt(spec, world, day, feedback), f"write {tag}")
            turns = result.get("turns") if isinstance(result, dict) else None
        except Exception as e:
            turns, fails, extracted = [], [f"format: {type(e).__name__}"], {}
        else:
            fails = shape_errors(turns)
            extracted = {}
            if not fails:
                try:
                    fails, extracted = await validate(checker, spec, check, turns, f"check {tag}")
                except Exception as e:
                    fails = [f"validator_error: {type(e).__name__}"]
        record = {"user_id": spec.user_id, "day": day.day, "date": day.date.isoformat(),
                  "weekday": day.weekday, "turns": turns or [], "model": writer.model,
                  "validator_model": checker.model, "attempts": attempt,
                  "passed_validation": not fails, "failures": fails, "extracted": extracted,
                  "first_try_failures": best["first_try_failures"] if best else fails}
        if best is None or len(fails) < len(best["failures"]):
            best = record
        best["attempts"] = attempt
        if not fails:
            break
        feedback = fails
    out.write_text(json.dumps(best, indent=2, ensure_ascii=False))
    return best


def parse_days(text: str | None, n: int) -> tuple[int, int]:
    if not text:
        return 1, n
    a, _, b = text.partition("-")
    return int(a), int(b or a)


async def generate(spec: PersonaSpec, world: WorldState, days: tuple[int, int], model: str,
                   validator_model: str, concurrency: int) -> list[dict]:
    out_dir = config.OUTPUT_DIR / spec.user_id / "sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = config.OUTPUT_DIR / spec.user_id / "llm_log.jsonl"
    writer = LLM(model, config.WRITER_TEMPERATURE, log)
    checker = LLM(validator_model, 0.0, log)
    todo = [d for d in world.days[days[0] - 1:days[1]]
            if d.has_session and not (out_dir / f"{d.date.isoformat()}.json").exists()]
    print(f"{spec.user_id}: {len(todo)} sessions to generate in days {days[0]}-{days[1]} "
          f"(writer {model}, validator {validator_model}, concurrency {concurrency})", flush=True)
    sem = asyncio.Semaphore(concurrency)
    done: list[dict] = []

    async def one(day: DayState) -> None:
        async with sem:
            rec = await generate_day(spec, world, day, writer, checker, out_dir / f"{day.date.isoformat()}.json")
            done.append(rec)
            status = "ok" if rec["passed_validation"] else f"FLAGGED ({len(rec['failures'])} issues)"
            print(f"  [{len(done)}/{len(todo)}] {rec['date']} attempts={rec['attempts']} {status}", flush=True)

    await asyncio.gather(*(one(d) for d in todo))
    return done


# ── Aggregation ─────────────────────────────────────────────────────

def load_sessions(user_id: str) -> list[dict]:
    d = config.OUTPUT_DIR / user_id / "sessions"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def fidelity(user_id: str) -> dict:
    sessions = load_sessions(user_id)
    kinds = Counter(f.split(":")[0] for s in sessions for f in s["failures"])
    first = Counter(f.split(":")[0] for s in sessions for f in s.get("first_try_failures", []))
    log = config.OUTPUT_DIR / user_id / "llm_log.jsonl"
    usage = Counter()
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            key = r["model"]
            usage[f"{key}.calls"] += 1
            usage[f"{key}.input_tokens"] += r["input_tokens"]
            usage[f"{key}.output_tokens"] += r["output_tokens"]
            usage[f"{key}.reasoning_tokens"] += r["reasoning_tokens"]
            usage[f"{key}.errors"] += bool(r["error"])
    n = len(sessions)
    return {
        "sessions": n,
        "passed_first_try": sum(s["passed_validation"] and s["attempts"] == 1 for s in sessions),
        "passed_after_retries": sum(s["passed_validation"] for s in sessions),
        "flagged": [s["date"] for s in sessions if not s["passed_validation"]],
        "first_try_failure_types": dict(first),
        "remaining_failure_types": dict(kinds),
        "mean_attempts": round(sum(s["attempts"] for s in sessions) / n, 2) if n else 0,
        "mean_turns": round(sum(len(s["turns"]) for s in sessions) / n, 1) if n else 0,
        "usage": dict(usage),
    }


def merge(spec: PersonaSpec) -> dict:
    sessions = load_sessions(spec.user_id)
    return {"user_id": spec.user_id, "name": spec.name,
            "sessions": {s["date"]: {"day": s["day"], "date": s["date"], "weekday": s["weekday"],
                                     "turns": s["turns"]} for s in sessions}}
