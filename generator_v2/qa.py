"""World state → QA pairs. Every answer is computed from the world state, never from text.

Questions are asked on the viewpoint day (num_days + 1). Recall, fact_at_time and
pattern_at_time refer to an earlier point in time but are still asked from there.

Sampling is seeded, stratified (regime × cycle phase, weekday), weighted towards regime
boundaries, and capped per answer so no (type, domain) group is dominated by one answer.
Yes/no types always come with positive controls so "no" is never the constant answer.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from . import config, rules
from .schema import PersonaSpec, QAItem, ResolvedRegime, WorldState
from .simulator import reference_registry

CAPABILITY = {
    "recall": "episodic_recall",
    "pattern_current": "pattern_abstraction",
    "pattern_at_time": "pattern_abstraction",
    "prediction": "pattern_extrapolation",
    "change_detection": "temporal_change",
    "attribution": "causal_reasoning",
    "exception_vs_shift": "exception_handling",
    "reversion": "temporal_change",
    "fact_current": "knowledge_update",
    "fact_at_time": "temporal_fact",
    "fact_history": "knowledge_update",
    "event_status": "event_tracking",
    "duration": "temporal_arithmetic",
    "distractor_probe": "causal_reasoning",
    "other_person": "entity_disambiguation",
    "abstention": "abstention",
}
TYPE_ORDER = list(CAPABILITY)
NOT_KNOWN = "Not known - this was never mentioned."


def fmt(d: dt.date) -> str:
    return f"{d.strftime('%A')}, {d.isoformat()}"


@dataclass
class Draft:
    type: str
    domain: str
    question: str
    answer: str
    answer_type: str
    accept: list[str]
    evidence: list[int]
    grading: str
    tags: list[str] = field(default_factory=list)


class QABuilder:
    def __init__(self, spec: PersonaSpec, world: WorldState):
        self.spec, self.world = spec, world
        self.name = spec.name
        self.N = spec.num_days
        self.V = spec.viewpoint_day
        self.level = spec.difficulty.level
        self.rng = random.Random(f"{spec.seed}:qa")
        self.drafts: list[Draft] = []
        self._questions: set[str] = set()
        self.noun = {p.domain: p.noun for p in spec.preferences}
        self.facts = {f.entity: f for f in spec.facts}
        self.stated = {s.id: s for s in world.statements}
        self.registry = reference_registry(spec)
        final_city = (world.days[-1].active_facts.get("city") or [None])[0]
        self._ctx = {}
        for day in world.days:
            city = (day.active_facts.get("city") or [None])[0]
            self._ctx[day.day] = rules.context_for(day.date, city)
        self._final_city = final_city
        self.boundaries = defaultdict(set)
        for r in world.regimes:
            if r.start > 1:
                self.boundaries[r.domain].add(r.start)

    # ── helpers ─────────────────────────────────────────────────────
    def date(self, day: int) -> dt.date:
        return self.spec.date_of(day)

    def ctx(self, day: int) -> rules.Context:
        if day in self._ctx:
            return self._ctx[day]
        return rules.context_for(self.date(day), self._final_city)

    def add(self, d: Draft) -> bool:
        if d.question in self._questions:
            return False
        self._questions.add(d.question)
        self.drafts.append(d)
        return True

    def regimes(self, domain: str) -> list[ResolvedRegime]:
        return self.world.regimes_for(domain)

    def regime_at(self, domain: str, day: int) -> ResolvedRegime:
        day = min(day, self.N)
        return next(r for r in self.regimes(domain) if r.start <= day <= r.end)

    def previous(self, reg: ResolvedRegime) -> Optional[ResolvedRegime]:
        regs = self.regimes(reg.domain)
        i = regs.index(reg)
        return regs[i - 1] if i > 0 else None

    def items(self, reg: ResolvedRegime) -> str:
        """Names a routine by what it involves, never by its rule, so questions don't leak patterns."""
        vals = sorted(rules.values_of(reg.rule))
        if len(vals) == 1:
            return vals[0]
        shown = " / ".join(vals) if len(vals) <= 5 else " / ".join(vals[:4]) + " etc."
        return f"the {shown} routine"

    def change_number(self, reg: ResolvedRegime) -> str:
        n = [r.id for r in self.regimes(reg.domain)].index(reg.id)
        return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}.get(n, f"{n}th")

    def describe(self, reg: ResolvedRegime) -> str:
        return rules.describe(reg.rule, self.date(reg.anchor))

    def near_boundary(self, domain: str, day: int) -> bool:
        return any(abs(day - b) <= config.BOUNDARY_WINDOW for b in self.boundaries[domain])

    def stated_day(self, sid: str) -> int:
        s = self.stated.get(sid)
        return s.stated_day if s and s.stated_day else 1

    def sample(self, cands: list[tuple], k: int) -> list[tuple]:
        """Greedy weighted sampling. Each candidate is (answer, weight, stratum, weekday, payload).
        Weights shrink as a stratum or weekday fills up; no answer may exceed ~half of k."""
        remaining = set(range(len(cands)))
        chosen: list[tuple] = []
        by_stratum, by_weekday, by_answer = Counter(), Counter(), Counter()
        cap = max(1, math.floor(0.5 * k))
        while remaining and len(chosen) < k:
            idx = sorted(remaining)
            ok = [i for i in idx if by_answer[cands[i][0]] < cap] or idx
            weights = [cands[i][1] / (1 + 3 * by_stratum[cands[i][2]]) / (1 + by_weekday[cands[i][3]])
                       for i in ok]
            i = self.rng.choices(ok, weights=weights, k=1)[0]
            remaining.discard(i)
            c = cands[i]
            chosen.append(c)
            by_stratum[c[2]] += 1
            by_weekday[c[3]] += 1
            by_answer[c[0]] += 1
        return chosen

    # ── preference questions ────────────────────────────────────────
    def recall(self, k_total: int) -> None:
        domains = list(self.noun)
        for i, dom in enumerate(domains):
            k = k_total // len(domains) + (1 if i < k_total % len(domains) else 0)
            cands = []
            for day in self.world.days:
                p = day.preferences[dom]
                reg = self.regime_at(dom, day.day)
                phase = rules.phase_key(reg.rule, day.day, day.date, reg.anchor, self.ctx(day.day))
                w = config.BOUNDARY_WEIGHT if self.near_boundary(dom, day.day) else 1.0
                cands.append((p.value, w, (reg.id, phase), day.weekday, day))
            for value, _, (reg_id, phase), _, day in self.sample(cands, k):
                p = day.preferences[dom]
                reg = next(r for r in self.regimes(dom) if r.id == reg_id)
                tags = []
                if p.mentioned:
                    evidence = [day.day]
                else:
                    tags.append("inferred")
                    evidence = self._nearest_mentions(reg, day.day)
                    # Ensure evidence doesn't come after the question date
                    before = [d for d in evidence if d <= day.day]
                    if not before:
                        continue
                    evidence = before
                if p.is_exception:
                    tags.append("exception")
                if not day.has_session:
                    tags.append("no_session")
                if self.near_boundary(dom, day.day):
                    tags.append("boundary")
                self.add(Draft("recall", dom, f"What was {self.name}'s {self.noun[dom]} on {fmt(day.date)}?",
                               value, "value", [value], evidence, "exact", tags))

    def _nearest_mentions(self, reg: ResolvedRegime, day: int, n: int = 3) -> list[int]:
        target = rules.phase_key(reg.rule, day, self.date(day), reg.anchor, self.ctx(day))
        same = [d for d in reg.mention_days
                if rules.phase_key(reg.rule, d, self.date(d), reg.anchor, self.ctx(d)) == target]
        pool = same or reg.mention_days
        return sorted(sorted(pool, key=lambda d: abs(d - day))[:n])

    def prediction(self, k_total: int) -> None:
        # A domain whose current rule is constant has a one-value answer space: ask it
        # twice and give its share to the domains where prediction is actually informative.
        finals = {dom: self.regime_at(dom, self.N) for dom in self.noun}
        small = [d for d, r in finals.items() if len(rules.values_of(r.rule)) == 1]
        varied = [d for d in finals if d not in small] or small
        quota = {d: 2 for d in small}
        spread = k_total - sum(quota.values()) if varied != small else k_total
        for i, dom in enumerate(varied):
            quota[dom] = spread // len(varied) + (1 if i < spread % len(varied) else 0)
        for dom, k in quota.items():
            reg = finals[dom]
            tags = ["small_answer_space"] if dom in small else []
            cands = []
            for day in range(self.V, self.V + config.PREDICTION_HORIZON):
                d = self.date(day)
                value = rules.evaluate(reg.rule, day, d, reg.anchor, self.ctx(day))
                phase = rules.phase_key(reg.rule, day, d, reg.anchor, self.ctx(day))
                cands.append((value, 1.0, phase, d.strftime("%A"), day))
            evidence = reg.mention_days[-3:]
            for value, _, _, _, day in self.sample(cands, k):
                d = self.date(day)
                why = rules.explain(reg.rule, day, d, reg.anchor, self.ctx(day))
                answer = (f"{value}. Reasoning: since {self.date(reg.start).isoformat()} the routine has been "
                          f"{self.describe(reg)}. {why[0].upper() + why[1:]}.")
                self.add(Draft("prediction", dom,
                               f"What will {self.name}'s {self.noun[dom]} most likely be on {fmt(d)}? Explain the reasoning.",
                               answer, "value", [value], evidence, "exact",
                               tags + [f"horizon_{day - self.N}d"]))

    def patterns(self) -> None:
        for dom, noun in self.noun.items():
            reg = self.regime_at(dom, self.N)
            self.add(Draft("pattern_current", dom, f"What is {self.name}'s current {noun} routine?",
                           f"{self.describe(reg)} (since {self.date(reg.start).isoformat()})",
                           "free_text", [reg.label], reg.mention_days[:2] + reg.mention_days[-2:],
                           "llm_judge", [reg.kind]))
        cands = [r for r in self.world.regimes if r.end - r.start + 1 >= 14]
        k = min(config.QA_TARGETS["pattern_at_time"], len(cands))
        picked = sorted(self.rng.sample(cands, k), key=lambda r: (r.domain, r.start))
        for reg in picked:
            month = self._full_month_inside(reg)
            when = f"in {month.strftime('%B %Y')}" if month else f"around {self.date((reg.start + reg.end) // 2).isoformat()}"
            self.add(Draft("pattern_at_time", reg.domain,
                           f"What was {self.name}'s {self.noun[reg.domain]} routine {when}?",
                           self.describe(reg), "free_text", [reg.label], reg.mention_days[:3], "llm_judge",
                           [reg.kind]))

    def _full_month_inside(self, reg: ResolvedRegime) -> Optional[dt.date]:
        start, end = self.date(reg.start), self.date(reg.end)
        months = []
        d = dt.date(start.year, start.month, 1)
        while d <= end:
            nxt = dt.date(d.year + d.month // 12, d.month % 12 + 1, 1)
            if d >= start and nxt - dt.timedelta(days=1) <= end:
                months.append(d)
            d = nxt
        return self.rng.choice(months) if months else None

    def changes_and_causes(self) -> None:
        for reg in self.world.regimes:
            if reg.kind == "initial" or reg.first_mention_day is None:
                continue
            prev = self.previous(reg)
            noun = self.noun[reg.domain]
            first = reg.first_mention_day
            accept = [self.date(d).isoformat() for d in range(reg.start, first + 1)]
            tags = [reg.kind, reg.visibility] + (["bounded"] if first > reg.start else [])
            self.add(Draft("change_detection", reg.domain,
                           (f"When did {self.name}'s {noun} routine change from {self.items(prev)} to {self.items(reg)}?"
                            if self.items(prev) != self.items(reg) else
                            f"When did {self.name}'s {noun} routine change for the {self.change_number(reg)} time?"),
                           f"{self.date(reg.start).isoformat()}"
                           + (f" (first observed {self.date(first).isoformat()})" if first > reg.start else ""),
                           "date", accept, sorted({prev.mention_days[-1], first}), "date_exact", tags))

            if reg.kind == "reversion":
                if reg.revert_reason:
                    self.add(Draft("attribution", reg.domain,
                                   f"Why did {self.name}'s {noun} go back to {self.items(reg)} around {self.date(reg.start).isoformat()}?",
                                   reg.revert_reason, "free_text", [reg.revert_reason], [first], "llm_judge",
                                   ["reversion", "explicit"]))
                continue
            if reg.visibility == "uncaused":
                answer, atype, accept_c, ev = "No reason was stated.", "abstain", ["no reason was stated"], [first]
            else:
                answer, atype, accept_c = reg.cause_text, "free_text", [reg.cause_text]
                ev = sorted({self.stated_day(self._cause_id(reg)), first})
            shared = sum(r.cause_ref == reg.cause_ref for r in self.world.regimes if r.cause_ref) > 1
            self.add(Draft("attribution", reg.domain,
                           f"Why did {self.name}'s {noun} routine change around {self.date(reg.start).isoformat()}?",
                           answer, atype, accept_c, ev, "llm_judge",
                           [reg.visibility, reg.kind] + (["cross_domain"] if shared else [])))

    def _cause_id(self, reg: ResolvedRegime) -> str:
        return self.registry[reg.cause_ref].id

    def exceptions_vs_shifts(self) -> None:
        per_domain = max(1, config.QA_TARGETS["exception_vs_shift"] // (2 * len(self.noun)))
        for dom, noun in self.noun.items():
            excs = [d for d in self.world.days if d.preferences[dom].is_exception]
            positives = []
            for reg in self.regimes(dom):
                if reg.kind != "shift":
                    continue
                prev = self.previous(reg)
                for day in reg.mention_days:
                    new = self.world.days[day - 1].preferences[dom].value
                    old = rules.evaluate(prev.rule, day, self.date(day), prev.anchor, self.ctx(day))
                    if new != old:
                        positives.append((reg, day, old, new))
                        break
            k = min(per_domain, len(excs), len(positives))
            for day in sorted(self.rng.sample(excs, k), key=lambda d: d.day):
                p = day.preferences[dom]
                after = next((d.day for d in self.world.days[day.day:] if d.preferences[dom].mentioned), day.day)
                self.add(Draft("exception_vs_shift", dom,
                               f"On {fmt(day.date)}, {self.name}'s {noun} was {p.value} instead of the usual "
                               f"{p.base_value}. Was that a lasting change?",
                               f"No - it was a one-off ({p.exception_reason}); the usual routine continued.",
                               "yes_no", ["no"], [day.day, after], "llm_judge", ["exception"]))
            for reg, day, old, new in sorted(self.rng.sample(positives, k), key=lambda x: x[1]):
                self.add(Draft("exception_vs_shift", dom,
                               f"On {fmt(self.date(day))}, {self.name}'s {noun} was {new} instead of the usual "
                               f"{old}. Was that a lasting change?",
                               f"Yes - from {self.date(reg.start).isoformat()} the routine changed to {reg.label}.",
                               "yes_no", ["yes"], reg.mention_days[:3], "llm_judge", ["shift", reg.visibility]))

    def reversions(self) -> None:
        for reg in self.world.regimes:
            if reg.kind != "temporary":
                continue
            noun = self.noun[reg.domain]
            back = next(r for r in self.regimes(reg.domain) if r.start == reg.end + 1)
            current = self.regime_at(reg.domain, self.N)
            self.add(Draft("reversion", reg.domain,
                           f"Is {self.name}'s {noun} still {self.items(reg)}?",
                           f"No - that lasted from {self.date(reg.start).isoformat()} to "
                           f"{self.date(reg.end).isoformat()}; then {self.name} went back to {back.label}.",
                           "yes_no", ["no"], sorted({reg.mention_days[-1], back.first_mention_day or back.start}),
                           "llm_judge", ["temporary"]))
            self.add(Draft("reversion", reg.domain,
                           f"What did {self.name}'s {noun} go back to after the period of {self.items(reg)}?",
                           f"{back.label} - {self.describe(back)}", "free_text", [back.label],
                           back.mention_days[:3], "llm_judge", ["temporary"]))
            self.add(Draft("reversion", reg.domain,
                           f"Is {self.name}'s {noun} still {self.items(current)}?",
                           f"Yes - since {self.date(current.start).isoformat()}.", "yes_no", ["yes"],
                           current.mention_days[-3:], "llm_judge", ["control"]))

    # ── fact questions ──────────────────────────────────────────────
    def _intervals(self, entity: str) -> list[tuple[str, int, int]]:
        """(value, first day, last day) for every maximal run of a value being true."""
        out, open_ = [], {}
        for day in self.world.days:
            vals = set(day.active_facts[entity])
            for v in list(open_):
                if v not in vals:
                    out.append((v, open_.pop(v), day.day - 1))
            for v in day.active_facts[entity]:
                open_.setdefault(v, day.day)
        out += [(v, s, self.N) for v, s in open_.items()]
        return sorted(out, key=lambda x: (x[1], x[0]))

    def _fact_evidence(self, entity: str, value: str) -> list[int]:
        days = [s.stated_day for s in self.world.statements
                if s.entity == entity and s.value == value and s.stated_day
                and s.kind in ("background_fact", "fact_change")]
        return sorted(set(days)) or [1]

    def _join(self, vals: list[str]) -> str:
        return ", ".join(vals) if vals else "none"

    def facts_questions(self) -> None:
        last = self.world.days[-1].active_facts
        changing = []
        for f in self.spec.facts:
            vals = last[f.entity]
            noun = f.label
            ev = sorted({d for v in vals for d in self._fact_evidence(f.entity, v)}) or [1]
            # Skip facts with "none" answer and no real evidence
            if not vals and ev == [1]:
                continue
            if f.cardinality == "single":
                v = vals[0]
                self.add(Draft("fact_current", f.entity, f"What is {self.name}'s current {noun}?", v, "value",
                               [v, v.split(",")[0]], ev, "exact", []))
            else:
                self.add(Draft("fact_current", f.entity, f"What are {self.name}'s current {noun}?",
                               self._join(vals), "free_text", [self._join(vals)], ev, "llm_judge", ["set"]))
            intervals = self._intervals(f.entity)
            if any(c.day > 1 for c in f.changes):
                changing.append(f)
                parts = [f"{v} ({self.date(s).isoformat()} to {'now' if e == self.N else self.date(e).isoformat()})"
                         for v, s, e in intervals]
                verb = "has" if f.cardinality == "single" else "have"
                self.add(Draft("fact_history", f.entity, f"How {verb} {self.name}'s {noun} changed over time?",
                               "; ".join(parts), "free_text", [intervals[-1][0]],
                               sorted({d for v, _, _ in intervals for d in self._fact_evidence(f.entity, v)}),
                               "llm_judge", [f.cardinality]))
        self._facts_at_time(changing)

    def _facts_at_time(self, entities: list) -> None:
        if not entities:
            return
        k_total = config.QA_TARGETS["fact_at_time"]
        change_days = defaultdict(set)
        for f in entities:
            change_days[f.entity] |= {c.day for c in f.changes if c.day > 1}
        for i, f in enumerate(entities):
            k = k_total // len(entities) + (1 if i < k_total % len(entities) else 0)
            intervals = self._intervals(f.entity)
            cands = []
            for day in self.world.days:
                vals = day.active_facts[f.entity]
                stratum = tuple(vals)
                w = config.BOUNDARY_WEIGHT if any(abs(day.day - c) <= config.BOUNDARY_WINDOW
                                                  for c in change_days[f.entity]) else 1.0
                cands.append((self._join(vals), w, stratum, day.weekday, day))
            for answer, _, _, _, day in self.sample(cands, k):
                vals = day.active_facts[f.entity]
                tags = ["set"] if f.cardinality == "multi" else []
                if not vals:
                    tags.append("empty")
                ev = sorted({d for v in vals for d in self._fact_evidence(f.entity, v)}) or [1]
                # Skip "none" answers backed only by the day-1 fallback — no
                # conversation ever discussed this entity, so the question is
                # unanswerable from dialogue evidence.
                if not vals and ev == [1]:
                    continue
                if f.cardinality == "single":
                    q = f"What was {self.name}'s {f.label} on {day.date.isoformat()}?"
                    self.add(Draft("fact_at_time", f.entity, q, answer, "value", [answer, answer.split(",")[0]],
                                   ev, "exact", tags))
                else:
                    q = f"What were {self.name}'s {f.label} on {day.date.isoformat()}?"
                    self.add(Draft("fact_at_time", f.entity, q, answer, "free_text", [answer], ev, "llm_judge", tags))

    def events(self) -> None:
        for e in self.spec.events:
            final = e.states[-1]
            self.add(Draft("event_status", "events", f"What happened with {e.title}?",
                           f"{final.text} ({final.status}, {self.date(final.day).isoformat()})", "free_text",
                           [final.text], [self.stated_day(s.id) for s in e.states], "llm_judge", [final.status]))
            for a, b in zip(e.states, e.states[1:]):
                # Pick a probe day after state `a` is mentioned but before state `b`
                ev_day = self.stated_day(a.id)
                mid = (a.day + b.day) // 2
                if mid <= a.day:
                    continue
                # Ensure the question date isn't before the evidence
                probe_day = max(mid, ev_day)
                if probe_day >= b.day:
                    continue
                self.add(Draft("event_status", "events",
                               f"What was the status of {e.title} on {self.date(probe_day).isoformat()}?",
                               f"{a.text} ({a.status})", "free_text", [a.text], [ev_day],
                               "llm_judge", ["at_time", a.status]))

    def durations(self) -> None:
        cands = []
        for reg in self.world.regimes:
            if reg.kind == "reversion":
                continue
            noun = self.noun[reg.domain]
            n = reg.end - reg.start + 1
            if reg.end < self.N:
                q = f"How long did {self.name}'s {noun} stay {self.items(reg)}?"
                a = f"{n} days ({self.date(reg.start).isoformat()} to {self.date(reg.end).isoformat()})"
            else:
                q = f"As of {self.date(self.N).isoformat()}, how long had {self.name}'s {noun} been {self.items(reg)}?"
                a = f"{n} days (since {self.date(reg.start).isoformat()})"
            cands.append(Draft("duration", reg.domain, q, a, "free_text", [str(n)],
                               sorted({reg.first_mention_day or reg.start, reg.mention_days[-1]}),
                               "llm_judge", [reg.kind]))
        for f in self.spec.facts:
            for v, s, e in self._intervals(f.entity):
                if s == 1 or e == self.N:
                    continue
                n = e - s + 1
                q = (f"How long was {v} one of {self.name}'s {f.label}?" if f.cardinality == "multi"
                     else f"How long was {self.name}'s {f.label} {v}?")
                cands.append(Draft("duration", f.entity, q,
                                   f"{n} days ({self.date(s).isoformat()} to {self.date(e).isoformat()})",
                                   "free_text", [str(n)], self._fact_evidence(f.entity, v), "llm_judge", ["fact"]))
        k = min(config.QA_TARGETS["duration"], len(cands))
        for d in self.rng.sample(cands, k):
            self.add(d)

    def distractors(self) -> None:
        caused = [r for r in self.world.regimes if r.kind in ("shift", "temporary") and r.cause_text]
        used = set()
        for dis in self.spec.distractors:
            noun = self.noun[dis.domain]
            answer = f"No - nothing about {self.name}'s {noun} changed because of it."
            if dis.confounder:
                near = [r for r in self.regimes(dis.domain) if abs(r.start - dis.day) <= 10 and r.kind != "initial"]
                if near:
                    answer = (f"No - {self.name}'s {noun} did change around then (to {near[0].label} from "
                              f"{self.date(near[0].start).isoformat()}), but no link to this was ever stated.")
            self.add(Draft("distractor_probe", dis.domain,
                           f"Around {self.date(dis.day).isoformat()}, {dis.text}. Did that lead to any change in {self.name}'s {noun}?",
                           answer, "yes_no", ["no"], [self.stated_day(dis.id)], "llm_judge",
                           ["confounder"] if dis.confounder else []))
            pool = [r for r in caused if r.domain == dis.domain and r.id not in used] or \
                   [r for r in caused if r.id not in used]
            if pool:
                reg = self.rng.choice(pool)
                used.add(reg.id)
                rnoun = self.noun[reg.domain]
                self.add(Draft("distractor_probe", reg.domain,
                               f"Around {self.date(reg.cause_day).isoformat()}, {reg.cause_text}. "
                               f"Did that lead to any change in {self.name}'s {rnoun}?",
                               f"Yes - from {self.date(reg.start).isoformat()} {self.name}'s {rnoun} changed to {reg.label}.",
                               "yes_no", ["yes"], sorted({self.stated_day(self._cause_id(reg)),
                                                          reg.first_mention_day or reg.start}),
                               "llm_judge", ["control", reg.visibility]))

    def other_people(self) -> None:
        last = self.world.days[-1].active_facts
        for p in self.spec.other_people:
            for i, fact in enumerate(p.facts):
                sid = f"other.{p.person.lower()}.{i}"
                ev = [self.stated_day(sid)]
                q = fact.question or f"What is {self.name}'s {p.relation} {p.person}'s {fact.noun}?"
                self.add(Draft("other_person", "other_people", q, fact.value, "value", [fact.value], ev,
                               "llm_judge", [p.relation]))
                if not fact.confusable_with:
                    continue
                uf = self.facts[fact.confusable_with]
                mine = last[uf.entity]
                ask = ((lambda v: f"Is {v} one of {self.name}'s {uf.label}?") if uf.cardinality == "multi"
                       else (lambda v: f"Is {self.name}'s {uf.label} {v}?"))
                self.add(Draft("other_person", "other_people", ask(fact.value),
                               f"No - that is {p.person} ({self.name}'s {p.relation}), not {self.name}.",
                               "yes_no", ["no"], ev, "llm_judge", ["confusable"]))
                if mine:
                    self.add(Draft("other_person", "other_people", ask(mine[0]), "Yes.", "yes_no", ["yes"],
                                   self._fact_evidence(uf.entity, mine[0]), "llm_judge", ["control"]))

    def abstention(self) -> None:
        n = config.QA_TARGETS["abstention"] // 2
        excluded = set(self.spec.abstention_exclude) | set(self.facts)
        templates = [t for t in config.ABSTENTION_TEMPLATES if t[0] not in excluded]
        for key, tmpl in self.rng.sample(templates, min(n, len(templates))):
            self.add(Draft("abstention", "abstention", tmpl.format(name=self.name), NOT_KNOWN, "abstain",
                           ["not known"], [], "llm_judge", [key]))
        last = self.world.days[-1].active_facts
        answerable = [f for f in self.spec.facts if last[f.entity]]
        for f in self.rng.sample(answerable, min(n, len(answerable))):
            vals = last[f.entity]
            q = (f"Does {self.name} have any {f.label}, and if so what?" if f.cardinality == "multi"
                 else f"Do you know {self.name}'s {f.label}?")
            a = self._join(vals)
            self.add(Draft("abstention", "abstention", q, a, "value", [a],
                           sorted({d for v in vals for d in self._fact_evidence(f.entity, v)}), "llm_judge",
                           ["control", f.entity]))

    # ── assembly ────────────────────────────────────────────────────
    def build(self) -> list[QAItem]:
        self.patterns()
        self.changes_and_causes()
        self.exceptions_vs_shifts()
        self.reversions()
        self.facts_questions()
        self.events()
        self.durations()
        self.distractors()
        self.other_people()
        self.abstention()
        remaining = max(0, config.QA_TOTAL_TARGET - len(self.drafts))
        share = sum(config.FILLER_SPLIT.values())
        k = {t: min(config.QA_MAX_FILLER[t], max(config.QA_MIN_FILLER[t], remaining * w // share))
             for t, w in config.FILLER_SPLIT.items()}
        self.recall(k["recall"])
        self.prediction(k["prediction"])

        ordered = sorted(self.drafts, key=lambda d: TYPE_ORDER.index(d.type))
        items = []
        for i, d in enumerate(ordered):
            latest = max(d.evidence) if d.evidence else self.N
            items.append(QAItem(
                id=f"{self.spec.user_id}_q{i:04d}", type=d.type, capability=CAPABILITY[d.type],
                domain=d.domain, difficulty_user=self.level, question=d.question, answer=d.answer,
                answer_type=d.answer_type, accept=d.accept, evidence_days=sorted(d.evidence),
                recency_distance_days=max(0, self.V - latest), grading=d.grading,
                viewpoint_day=self.V, tags=d.tags,
            ))
        return items


def build_qa(spec: PersonaSpec, world: WorldState) -> list[QAItem]:
    return QABuilder(spec, world).build()


# ── Curation ────────────────────────────────────────────────────────

def _pick_diverse(pool: list[QAItem], k: int, rng: random.Random) -> list[QAItem]:
    """Greedy pick that spreads the choice over answer types, answers, domains and tags."""
    pool = list(pool)
    rng.shuffle(pool)
    chosen: list[QAItem] = []
    seen = {name: Counter() for name in ("atype", "answer", "domain", "tags")}
    keys = lambda q: {"atype": q.answer_type, "answer": q.accept[0].lower() if q.accept else q.answer,
                      "domain": q.domain, "tags": tuple(sorted(q.tags))}
    while pool and len(chosen) < k:
        best = min(pool, key=lambda q: tuple(seen[n][v] for n, v in keys(q).items()))
        pool.remove(best)
        chosen.append(best)
        for n, v in keys(best).items():
            seen[n][v] += 1
    return chosen


def curate(items: list[QAItem], n: int, seed: int) -> list[QAItem]:
    """A small, balanced subset: every type the user has, weighted towards the harder ones."""
    rng = random.Random(f"{seed}:curate")
    by_type: dict[str, list[QAItem]] = defaultdict(list)
    for q in items:
        by_type[q.type].append(q)
    present = [t for t in TYPE_ORDER if by_type[t]]
    quota = {t: 0 for t in present}
    for _ in range(min(n, len(items))):
        open_ = [t for t in present if quota[t] < len(by_type[t])]
        t = max(open_, key=lambda t: (config.CURATED_WEIGHTS.get(t, 1) / (quota[t] + 1), -TYPE_ORDER.index(t)))
        quota[t] += 1
    picked = {q.id for t in present for q in _pick_diverse(by_type[t], quota[t], rng)}
    return [q for q in items if q.id in picked]


# ── Answer-distribution guard ───────────────────────────────────────

def answer_distribution(items: list[QAItem]) -> dict[tuple[str, str], Counter]:
    groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for q in items:
        groups[(q.type, q.domain)][q.accept[0].strip().lower() if q.accept else q.answer.lower()] += 1
    return groups


def entropy(c: Counter) -> float:
    total = sum(c.values())
    return -sum(n / total * math.log2(n / total) for n in c.values()) if total else 0.0


def guard_errors(items: list[QAItem]) -> list[str]:
    errs = []
    small = defaultdict(list)
    for q in items:
        small[(q.type, q.domain)].append("small_answer_space" in q.tags)
    for (qtype, dom), c in sorted(answer_distribution(items).items()):
        total = sum(c.values())
        if total < config.GUARD_MIN_GROUP_SIZE or all(small[(qtype, dom)]):
            continue
        answer, n = c.most_common(1)[0]
        if n / total > config.MAX_SHARED_ANSWER_FRACTION:
            errs.append(f"{qtype}/{dom}: '{answer}' is {n}/{total} = {n / total:.0%} of answers")
    return errs


def qa_stats(items: list[QAItem]) -> dict:
    dist = answer_distribution(items)
    return {
        "total": len(items),
        "by_type": dict(Counter(q.type for q in items)),
        "by_capability": dict(Counter(q.capability for q in items)),
        "by_grading": dict(Counter(q.grading for q in items)),
        "tags": dict(Counter(t for q in items for t in q.tags)),
        "answer_distribution": {
            f"{t}/{d}": {"n": sum(c.values()), "top_answer": c.most_common(1)[0][0],
                         "top_share": round(c.most_common(1)[0][1] / sum(c.values()), 3),
                         "entropy_bits": round(entropy(c), 3), "distinct": len(c)}
            for (t, d), c in sorted(dist.items())
        },
    }
