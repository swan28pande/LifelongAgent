"""system_3: a grounded user agent talks live with a memory-backed assistant, day by day.

Per session day of the v2 world state:
1. The user agent chats with the assistant, covering the day's checklist under guardrails.
2. After the first ASK_AFTER_DAY days, with probability ASK_P, the checklist also holds one question
   the user puts to the assistant mid-conversation: mostly (CURRENT_SHARE) "what's my X today?" for a
   preference they talk about that day, on a day its pattern does not change; otherwise a
   dataset question about the past ("what did I have for lunch on ..."). The user agent knows the true answer, rates the assistant's reply (correct, partial,
   wrong, dont_know) and reacts like a real person, correcting the assistant when it is wrong.
3. The generator's validator checks the day's chat (without the question and its answer) against
   the world state; any item still missing triggers up to MAX_REPAIRS extra user turns. Other
   findings (leaks, wrong values, inventions) are recorded as flags.
4. The whole day, question and correction included, is written to the assistant's memory.
After the run, the external LLM judge also grades every answer, for comparison with other protocols.

Results: results/system_3/<method>/<run>/<user>/{days,asks,grades}.jsonl, summary.json and LLM logs.
Resumes per completed day.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from generator_v2 import config as gen_config
from generator_v2.llm import LLM
from generator_v2.qa import QABuilder
from generator_v2.schema import load_ladder, load_spec
from generator_v2.simulator import simulate
from generator_v2.validator import build_check, validate

from experiments import config
from experiments.core.grading import JUDGE_SYSTEM, judge_prompt
from experiments.core.method import UsageCounter
from experiments.core.types import Instance, Question
from .methods import build
from .user_agent import GroundedUser

RESULTS_DIR = config.RESULTS_DIR.parent / "system_3"
MAX_TURNS = 8          # user messages before the agent must state everything still pending
MAX_REPAIRS = 2        # extra rounds when the validator finds checklist items missing
COVERAGE = ("statement_missing", "pref_absent", "cause_not_linked")
ASK_AFTER_DAY = 30     # no questions before there is a month of history
ASK_P = 0.15           # chance that a session day includes a question to the assistant
CURRENT_SHARE = 0.7    # of those, the share asking for today's preference; the rest are dataset questions
# Answers to these depend on the date they are asked ("current", "still", "over time", "next"); they
# are asked only on their own viewpoint day. All other questions are anchored to fixed dates or events.
RELATIVE = {"fact_current", "pattern_current", "prediction", "fact_history", "reversion"}
def unknowable(q: dict) -> bool:
    """Abstention questions whose answer was never given; others ("Do you know Leo's diet?") have one."""
    return q["type"] == "abstention" and q["answer"].lower().startswith("not known")


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def _append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _measure(usage: UsageCounter, fn, *args):
    """Run an assistant step; return its result with wall time and the assistant's own LLM usage."""
    before, started = usage.snapshot(), time.monotonic()
    result = fn(*args)
    after = usage.snapshot()
    return result, {"seconds": round(time.monotonic() - started, 3),
                    **{k: after[k] - before[k] for k in after}}


def _missing_ids(fails: list[str], items) -> set[str]:
    """Map validator coverage failures back to checklist item ids."""
    ids = set()
    for f in fails:
        kind, _, rest = f.partition(": ")
        if kind == "statement_missing":
            ids.add(rest.split(" ", 1)[0])
        elif kind == "pref_absent":
            ids.add("pref:" + rest.split(" ", 1)[0])
        elif kind == "cause_not_linked":
            ids |= {i.id for i in items if i.id.startswith("c")}
    return ids


def question_pool(user_id: str) -> list[dict]:
    """The user's dataset questions: the 200 monthly probes plus the wider question pool, deduplicated."""
    base = config.V2_DATA_DIR / user_id
    probes = [q for p in json.loads((base / "probing_questions.json").read_text())["probes"] for q in p["questions"]]
    pool, seen = [], set()
    for q in probes + json.loads((base / "qa_pool.json").read_text()):
        key = (q["question"], q["answer"], q["viewpoint_day"])
        if key not in seen:
            seen.add(key)
            pool.append(q)
    return pool


def eligible(q: dict, day: int, date: str) -> bool:
    """Askable on this day: all evidence was said on earlier days, every date the question names is
    in the past, and the reference still holds."""
    if any(e >= day for e in q["evidence_days"]) or any(d >= date for d in re.findall(r"\d{4}-\d{2}-\d{2}", q["question"])):
        return False
    return q["viewpoint_day"] == day if q["type"] in RELATIVE else True


def current_question(qa: QABuilder, day, rng: random.Random) -> dict | None:
    """"What's my X today?" for a preference the user talks about today, asked before they say it.
    The answer is today's value under the routine in effect. Asked only when the pattern does not change
    that day: the routine started on an earlier day, today is not a one-off exception, and every part of
    the routine was observed on earlier days (so today's value follows from memory)."""
    options = []
    for dom, pref in sorted(day.preferences.items()):
        if not pref.mentioned or pref.is_exception:
            continue
        reg = qa.regime_at(dom, day.day)
        evidence = qa.observations.pattern_days(reg, day.day - 1) if reg.start < day.day else []
        if evidence:
            options.append((dom, reg, evidence, pref.value))
    if not options:
        return None
    dom, reg, evidence, value = rng.choice(options)
    when = f"{day.weekday}, {day.date.isoformat()}"
    return {"id": f"today:{dom}:{day.date.isoformat()}", "type": "preference_today", "domain": dom,
            "question": f"What will {qa.name}'s {qa.noun[dom]} be today ({when})?",
            "answer": value, "accept": [value], "answer_type": "value", "grading": "exact",
            "evidence_days": evidence, "viewpoint_day": day.day, "routine": qa.describe(reg),
            "regime_kind": reg.kind, "regime_start": qa.date(reg.start).isoformat()}


def pick_question(user_id: str, day, pool: list[dict], asked: set[str], qa: QABuilder) -> dict | None:
    """Deterministic per user and date, so every method faces the same questions on the same days."""
    date = day.date.isoformat()
    rng = random.Random(f"{user_id}|{date}")
    if day.day <= ASK_AFTER_DAY or rng.random() >= ASK_P:
        return None
    current = current_question(qa, day, rng) if rng.random() < CURRENT_SHARE else None
    if current:
        return current
    options = sorted((q for q in pool if q["id"] not in asked and q["question"] not in asked
                      and eligible(q, day.day, date)), key=lambda q: q["id"])
    return rng.choice(options) if options else current_question(qa, day, rng)


def question_schedule(user_id: str, spec, world) -> dict[str, dict]:
    """date -> question, shared by every method. Built once from the seeded picks and saved to
    results/system_3/schedule/<user>.json; later runs read that file, so all methods face the same
    question on the same day. Edit the file to choose a question for a day by hand."""
    path = RESULTS_DIR / "schedule" / f"{user_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    pool, qa, asked, schedule = question_pool(user_id), QABuilder(spec, world), set(), {}
    for day in world.days:
        if day.has_session:
            q = pick_question(user_id, day, pool, asked, qa)
            if q:
                asked.update((q["id"], q["question"]))
                schedule[day.date.isoformat()] = q
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schedule, indent=1, ensure_ascii=False))
    return schedule


def simulate_user(method_name, user_id, run_name, model, through, loop, max_days=None) -> dict:
    out = RESULTS_DIR / method_name / run_name / user_id
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(gen_config.PERSONA_DIR / f"{user_id}.yaml", load_ladder(gen_config.LADDER_PATH))
    world = simulate(spec)
    schedule, nouns = question_schedule(user_id, spec, world), {p.domain: p.noun for p in spec.preferences}

    user = GroundedUser(spec, world, LLM(model, 0.7, out / "user_agent_log.jsonl"))
    validator = LLM(gen_config.VALIDATOR_MODEL, 0.0, out / "validator_log.jsonl")
    usage = UsageCounter()
    method, assistant = build(method_name, out / "store", Instance(user_id, [], [], spec.name), model, usage)
    done = {d["date"] for d in _load(out / "days.jsonl")}
    # Drop questions from a day that was interrupted before it was stored; that day is replayed.
    asks = [a for a in _load(out / "asks.jsonl") if a["asked_on"] in done]
    (out / "asks.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in asks))
    days = [d for d in world.days if d.has_session and (through is None or d.date.isoformat()[:7] <= through)]
    days = days[:max_days] if max_days else days

    for day in days:
        date = day.date.isoformat()
        if date in done:
            continue
        print(f"[{user_id}/{method_name}] {date}", flush=True)
        assistant.start_day(date)
        items, covered, turns, replies = user.checklist(day), set(), [], []
        q = schedule.get(date)
        ask_id = f"ask:{q['id']}" if q else None
        if q:
            items.append(user.ask_item(q, nouns.get(q.get("domain")) if q["id"].startswith("today:") else None))
            for i in items:   # today's choice in that domain comes only after the question is answered
                if q["id"].startswith("today:") and i.id == f"pref:{q['domain']}":
                    i.text += " (say this only after your ask: question has been answered)"
        pending: dict | None = None   # the asked question, until the user has rated the answer

        def exchange(message: str, qa: bool = False) -> None:
            text = f"[{day.weekday}, {date}] {message}" if not turns else message
            turns.append({"speaker": "user", "text": text, "qa": qa})
            reply, cost = _measure(usage, assistant.reply, text)
            replies.append(cost)
            turns.append({"speaker": "assistant", "text": reply, "qa": qa})

        def user_turn(tag: str, insist: bool = False) -> dict:
            nonlocal pending
            msg = loop.run_until_complete(user.next_message(
                day, items, covered, turns, tag, insist=insist,
                evaluate={**q, "unknowable": unknowable(q)} if pending else None))
            new = set(msg["covered"]) - covered
            covered.update(msg["covered"])
            if pending:
                right = msg["verdict"] == ("dont_know" if unknowable(q) else "correct")
                rec = {**pending, "verdict": msg["verdict"], "user_correct": right, "why": msg["why"],
                       "reaction": msg["message"]}
                _append(out / "asks.jsonl", rec)
                pending = None
            asking = ask_id in new
            exchange(msg["message"], qa=asking)
            if asking:
                pending = {"id": q["id"], "type": q["type"], "question": q["question"], "answer": q["answer"],
                           "accept": q.get("accept") or [q["answer"]], "answer_type": q["answer_type"],
                           "grading": q["grading"], "asked_on": date, "day": day.day,
                           "evidence_days": q["evidence_days"], "viewpoint_day": q["viewpoint_day"],
                           "natural_question": turns[-2]["text"], "response": turns[-1]["text"], "cost": replies[-1],
                           "kind": "today" if q["id"].startswith("today:") else "dataset",
                           **{k: q[k] for k in ("domain", "routine", "regime_kind", "regime_start") if k in q}}
            return msg

        for k in range(MAX_TURNS):
            msg = user_turn(f"user {date}#{k}", insist=k == MAX_TURNS - 1)
            if not pending and k >= 1 and (msg["done"] or covered >= {i.id for i in items}):
                break
        if pending:
            user_turn(f"react {date}")

        chat = lambda: [t for t in turns if not t["qa"]]   # the question and its answer are not checked
        check = build_check(spec, world, day)
        fails, _ = loop.run_until_complete(validate(validator, spec, check, chat(), f"check {date}#0"))
        repairs = 0
        while repairs < MAX_REPAIRS and any(f.startswith(COVERAGE) for f in fails):
            covered -= _missing_ids([f for f in fails if f.startswith(COVERAGE)], items)
            user_turn(f"repair {date}#{repairs}", insist=True)
            if pending:
                user_turn(f"react {date}#{repairs}")
            repairs += 1
            fails, _ = loop.run_until_complete(validate(validator, spec, check, chat(), f"check {date}#{repairs}"))

        stats, write = _measure(usage, assistant.end_day, turns)
        _append(out / "days.jsonl", {
            "date": date, "turns": turns, "scripted_turns": len(turns), "items": [i.__dict__ for i in items],
            "question": q["id"] if q else None,
            "question_asked": bool(q) and any(t["qa"] for t in turns),
            "missing": [f for f in fails if f.startswith(COVERAGE)],
            "flags": [f for f in fails if not f.startswith(COVERAGE)],
            "repairs": repairs, "ingest": stats, "cost": {"replies": replies, "memory_write": write}})

    if hasattr(method, "close"):
        method.close()
    return grade_user(out, loop)


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(key, "dataset")].append(r)
    return groups


def grade_user(out: Path, loop) -> dict:
    """User-agent verdicts are the primary score; the external judge grades the same answers too."""
    asks = _load(out / "asks.jsonl")
    grades = {g["id"]: g for g in _load(out / "grades.jsonl")}
    judge = LLM(config.JUDGE_MODEL, 0.0, out / "judge_log.jsonl")

    async def grade_all():
        sem = asyncio.Semaphore(config.JUDGE_CONCURRENCY)

        async def one(a):
            q = Question(a["id"], a["question"], a["answer"], a["accept"], a["answer_type"], a["grading"],
                         a["type"], a["asked_on"])
            async with sem:
                result, _ = await judge.json(JUDGE_SYSTEM, judge_prompt(q, a["response"]), f"judge {a['id']}")
            if isinstance(result, list) and len(result) == 1:
                result = result[0]
            g = {"id": a["id"], "type": a["type"], "judge": result.get("verdict") == "RIGHT",
                 "why": str(result.get("why", ""))}
            _append(out / "grades.jsonl", g)
            grades[a["id"]] = g

        await asyncio.gather(*(one(a) for a in asks if a["id"] not in grades))

    loop.run_until_complete(grade_all())
    days = _load(out / "days.jsonl")
    rate = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    by_type = defaultdict(list)
    for a in asks:
        by_type[a["type"]].append(a)
    summary = {
        "days": len(days), "questions_planned": sum(1 for d in days if d.get("question")),
        "questions_asked": len(asks),
        "user_accuracy": rate([a["user_correct"] for a in asks]),
        "judge_accuracy": rate([grades[a["id"]]["judge"] for a in asks if a["id"] in grades]),
        "user_judge_agreement": rate([a["user_correct"] == grades[a["id"]]["judge"] for a in asks if a["id"] in grades]),
        "verdicts": dict(Counter(a["verdict"] for a in asks)),
        "by_kind": {k: {"n": len(v), "user": rate([a["user_correct"] for a in v]),
                        "judge": rate([grades[a["id"]]["judge"] for a in v if a["id"] in grades])}
                    for k, v in sorted(_group(asks, "kind").items())},
        "by_type": {t: {"n": len(v), "user": rate([a["user_correct"] for a in v]),
                        "judge": rate([grades[a["id"]]["judge"] for a in v if a["id"] in grades])}
                    for t, v in sorted(by_type.items())},
        "days_with_missing_items": sum(1 for d in days if d["missing"]),
        "days_flagged": sum(1 for d in days if d["flags"]),
        "repairs": sum(d["repairs"] for d in days),
        "avg_turns": round(statistics.mean(d["scripted_turns"] for d in days), 1) if days else 0,
    }
    summary["flag_kinds"] = dict(Counter(f.split(":")[0] for d in days for f in d["flags"]))
    summary["cost"] = _cost_summary(out, days)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _cost_summary(out: Path, days: list[dict]) -> dict:
    """Latency and tokens: every assistant reply, its answers to questions, memory writes, and overhead."""
    replies = [c for d in days for c in d.get("cost", {}).get("replies", [])]
    ask_costs = [a["cost"] for a in _load(out / "asks.jsonl")]
    writes = [d["cost"]["memory_write"] for d in days if d.get("cost")]

    def stats(costs):
        if not costs:
            return None
        secs = sorted(c["seconds"] for c in costs)
        return {"n": len(costs), "seconds_mean": round(statistics.mean(secs), 2),
                "seconds_p50": secs[len(secs) // 2], "seconds_p95": secs[min(int(len(secs) * .95), len(secs) - 1)],
                "seconds_total": round(sum(secs), 1),
                **{k: sum(c[k] for c in costs) for k in ("calls", "input_tokens", "output_tokens")},
                "tokens_per_item": round(sum(c["input_tokens"] + c["output_tokens"] for c in costs) / len(costs))}

    def log_tokens(name):
        rows = _load(out / name)
        return {"calls": len(rows), **{k: sum(r.get(k, 0) for r in rows)
                                       for k in ("input_tokens", "output_tokens", "reasoning_tokens")}}

    return {"assistant_replies": stats(replies), "answers_to_questions": stats(ask_costs),
            "memory_writes": stats(writes),
            "user_agent": log_tokens("user_agent_log.jsonl"), "validator": log_tokens("validator_log.jsonl"),
            "judge": log_tokens("judge_log.jsonl")}


def run(method_name: str, users: list[str], run_name: str, model: str, through: str | None,
        max_days: int | None = None) -> dict:
    loop = asyncio.new_event_loop()
    try:
        return {u: simulate_user(method_name, u, run_name, model, through, loop, max_days) for u in users}
    finally:
        loop.close()
