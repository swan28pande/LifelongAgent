"""Observations that ground QA, including uncertainty at routine boundaries."""

from __future__ import annotations

from functools import lru_cache

from . import rules
from .schema import PersonaSpec, ResolvedRegime, WorldState
from .simulator import reference_registry


class Observations:
    def __init__(self, spec: PersonaSpec, world: WorldState):
        self.spec, self.world = spec, world
        self.causes = reference_registry(spec)

    def context(self, day: int):
        state = self.world.days[day - 1]
        return rules.context_for(state.date, (state.active_facts.get("city") or [None])[0])

    def fact_days(self, entity: str, target: int) -> list[int]:
        """Disclosures establishing the complete state at target, even if told later.

        Include removals and replacements, not just statements naming retained values.
        A caller must wait for every returned day before asking the question.
        """
        statements = [s for s in self.world.statements
                      if s.entity == entity and s.effective_day <= target
                      and s.kind in ("background_fact", "fact_change")]
        return sorted({s.stated_day for s in statements if s.stated_day is not None})

    def pattern_days(self, reg: ResolvedRegime, cutoff: int) -> list[int]:
        """Cover every applicable phase and every value described by a rule.

        Empty means the full pattern has an unobserved part; it is not answerable.
        Exceptions report the usual base value too, so they can support a phase.
        """
        phases = {rules.phase_key(reg.rule, d.day, d.date, reg.anchor, self.context(d.day))
                  for d in self.world.days[reg.start - 1: min(reg.end, cutoff)]}
        seen, values = {}, set()
        for n in reg.mention_days:
            if n > cutoff:
                break
            day = self.world.days[n - 1]
            phase = rules.phase_key(reg.rule, n, day.date, reg.anchor, self.context(n))
            seen.setdefault(phase, n)
            values.add(day.preferences[reg.domain].base_value)
        if not phases or phases - seen.keys() or rules.values_of(reg.rule) - values:
            return []
        last = max(n for n in reg.mention_days if n <= cutoff)
        return sorted(set(seen.values()) | {last})

    def distinguishing_day(self, reg: ResolvedRegime) -> int | None:
        """First observed choice incompatible with the preceding routine."""
        previous = self.previous(reg)
        if previous is None:
            return reg.first_mention_day
        return next((n for n in reg.mention_days
                     if self.world.days[n - 1].preferences[reg.domain].base_value !=
                     rules.evaluate(previous.rule, n, self.spec.date_of(n),
                                    previous.anchor, self.context(n))), None)

    def previous(self, reg: ResolvedRegime) -> ResolvedRegime | None:
        regimes = self.world.regimes_for(reg.domain)
        i = regimes.index(reg)
        return regimes[i - 1] if i else None

    @lru_cache(maxsize=None)
    def change(self, regime_id: str, cutoff: int) -> tuple[list[int], list[int]]:
        """Compatible change dates and the observations constraining them.

        Implicit changes need not begin on their first mention: old/new rules can
        coincide that day, and weekly blocks may have an indistinguishable rotation.
        Keep explicit announcements and condition-ending reversions within the
        declared start-to-disclosure interval. Use only choices available by cutoff.
        """
        reg = next(r for r in self.world.regimes if r.id == regime_id)
        previous = self.previous(reg)
        first = self.distinguishing_day(reg)
        if previous is None or first is None or first > cutoff:
            return [], []
        lower = previous.start + 14
        upper = min(reg.end - 13, cutoff)
        if reg.visibility == "explicit" or reg.kind == "reversion":
            lower = max(lower, reg.start)
            upper = min(upper, reg.first_mention_day)
        elif reg.cause_day is not None:
            lower = max(lower, reg.cause_day + self.spec.difficulty.lag_days.min)
            upper = min(upper, reg.cause_day + self.spec.difficulty.lag_days.max)

        observed = [d for d in self.world.days[lower - 1:min(reg.end, cutoff)]
                    if d.preferences[reg.domain].mentioned]
        variants = [reg.rule]
        if reg.rule.type == "weekly_alternate" and reg.visibility != "explicit":
            variants += [reg.rule.model_copy(update={"values": reg.rule.values[i:] + reg.rule.values[:i]})
                         for i in range(1, len(reg.rule.values))]
        accepted = []
        for start in range(lower, upper + 1):
            for rule in variants:
                anchor = reg.anchor if reg.kind == "reversion" else start
                if all((rules.evaluate(previous.rule, d.day, d.date, previous.anchor, self.context(d.day))
                        if d.day < start else rules.evaluate(rule, d.day, d.date, anchor, self.context(d.day)))
                       == d.preferences[reg.domain].base_value for d in observed):
                    accepted.append(start)
                    break
        # Declared transition timing can add information beyond the daily choices.
        # Always retain the true start-to-disclosure interval used by the benchmark.
        accepted = sorted(set(accepted) | set(range(reg.start, min(reg.first_mention_day, cutoff) + 1)))
        before = [n for n in previous.mention_days if n < lower]
        evidence = {d.day for d in observed} | {first}
        if before:
            evidence.add(before[-1])
        if reg.cause_ref:
            cause_id = self.causes[reg.cause_ref].id
            cause = next(s for s in self.world.statements if s.id == cause_id)
            if cause.stated_day is not None and cause.stated_day <= cutoff:
                evidence.add(cause.stated_day)
        return accepted, sorted(evidence)
