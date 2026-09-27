"""Preference rules: pure functions of (day, date, anchor, world context).

Every cyclic rule is anchored to its regime's start day (`anchor`), never to the dataset
start. A reverted regime keeps the anchor of the regime it reverts to, so its phase
continues as if the calendar had kept running.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

from .schema import (
    ConditionalRule, ConstantRule, DayOfWeekRule, NDayCycleRule, NestedRule, Rule,
    WeeklyAlternateRule, WEEKDAYS,
)

SEASON_BY_MONTH = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}

# Month → weather bucket for known cities; anything else falls back to the season default.
CLIMATE: dict[str, list[str]] = {
    "austin":     ["mild", "mild", "mild", "warm", "hot", "hot", "hot", "hot", "hot", "warm", "mild", "mild"],
    "chicago":    ["snowy", "snowy", "cold", "mild", "mild", "warm", "hot", "hot", "warm", "mild", "cold", "snowy"],
    "denver":     ["snowy", "snowy", "cold", "cold", "mild", "warm", "hot", "warm", "warm", "mild", "cold", "snowy"],
    "pittsburgh": ["snowy", "snowy", "cold", "rainy", "rainy", "warm", "hot", "warm", "mild", "mild", "cold", "snowy"],
    "seattle":    ["rainy", "rainy", "rainy", "rainy", "mild", "mild", "warm", "warm", "mild", "rainy", "rainy", "rainy"],
}
SEASON_DEFAULT_WEATHER = {"winter": "cold", "spring": "mild", "summer": "hot", "fall": "mild"}

WEEKDAY_LABEL = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri", "sat": "Sat", "sun": "Sun"}
CONDITION_LABEL = {("is_workday", "true"): "on workdays (Mon-Fri)", ("is_workday", "false"): "on weekends"}


@dataclass(frozen=True)
class Context:
    is_workday: bool
    season: str
    weather: str

    def value_of(self, condition: str) -> str:
        if condition == "is_workday":
            return "true" if self.is_workday else "false"
        return getattr(self, condition)


def weather_for(city: Optional[str], date: dt.date) -> str:
    key = (city or "").split(",")[0].strip().lower()
    if key in CLIMATE:
        return CLIMATE[key][date.month - 1]
    return SEASON_DEFAULT_WEATHER[SEASON_BY_MONTH[date.month]]


def context_for(date: dt.date, city: Optional[str] = None) -> Context:
    return Context(
        is_workday=date.weekday() < 5,
        season=SEASON_BY_MONTH[date.month],
        weather=weather_for(city, date),
    )


def weekday_key(date: dt.date) -> str:
    return WEEKDAYS[date.weekday()]


def _offset(day: int, anchor: int) -> int:
    if day < anchor:
        raise ValueError(f"day {day} is before rule anchor {anchor}")
    return day - anchor


def _case(rule: ConditionalRule, ctx: Context) -> "str | Rule":
    key = ctx.value_of(rule.condition)
    if key in rule.cases:
        return rule.cases[key]
    if "default" in rule.cases:
        return rule.cases["default"]
    raise ValueError(f"conditional rule has no case for {rule.condition}={key} and no default")


def evaluate(rule: Rule, day: int, date: dt.date, anchor: int, ctx: Context) -> str:
    off = _offset(day, anchor)
    if isinstance(rule, ConstantRule):
        return rule.value
    if isinstance(rule, WeeklyAlternateRule):
        return rule.values[(off // 7) % len(rule.values)]
    if isinstance(rule, NDayCycleRule):
        pos = off % sum(s.days for s in rule.segments)
        for s in rule.segments:
            if pos < s.days:
                return s.value
            pos -= s.days
    if isinstance(rule, DayOfWeekRule):
        return rule.map[weekday_key(date)]
    if isinstance(rule, NestedRule):
        block = rule.blocks[(off // rule.block_days) % len(rule.blocks)]
        return evaluate(block, day, date, anchor, ctx)
    if isinstance(rule, ConditionalRule):
        case = _case(rule, ctx)
        return case if isinstance(case, str) else evaluate(case, day, date, anchor, ctx)
    raise TypeError(f"unknown rule {rule!r}")


def phase_key(rule: Rule, day: int, date: dt.date, anchor: int, ctx: Context) -> tuple:
    """Which part of the cycle a day falls in, for stratified sampling."""
    off = _offset(day, anchor)
    if isinstance(rule, ConstantRule):
        return ()
    if isinstance(rule, WeeklyAlternateRule):
        return ((off // 7) % len(rule.values),)
    if isinstance(rule, NDayCycleRule):
        pos = off % sum(s.days for s in rule.segments)
        for i, s in enumerate(rule.segments):
            if pos < s.days:
                return (i,)
            pos -= s.days
    if isinstance(rule, DayOfWeekRule):
        return (weekday_key(date),)
    if isinstance(rule, NestedRule):
        idx = (off // rule.block_days) % len(rule.blocks)
        return (idx,) + phase_key(rule.blocks[idx], day, date, anchor, ctx)
    if isinstance(rule, ConditionalRule):
        key = ctx.value_of(rule.condition)
        case = _case(rule, ctx)
        sub = () if isinstance(case, str) else phase_key(case, day, date, anchor, ctx)
        return (key,) + sub
    raise TypeError(f"unknown rule {rule!r}")


def values_of(rule: "Rule | str") -> set[str]:
    if isinstance(rule, str):
        return {rule}
    if isinstance(rule, ConstantRule):
        return {rule.value}
    if isinstance(rule, WeeklyAlternateRule):
        return set(rule.values)
    if isinstance(rule, NDayCycleRule):
        return {s.value for s in rule.segments}
    if isinstance(rule, DayOfWeekRule):
        return set(rule.map.values())
    if isinstance(rule, NestedRule):
        return set().union(*(values_of(b) for b in rule.blocks))
    if isinstance(rule, ConditionalRule):
        return set().union(*(values_of(c) for c in rule.cases.values()))
    raise TypeError(f"unknown rule {rule!r}")


def _fmt_date(d: dt.date) -> str:
    return f"{d.strftime('%a')} {d.isoformat()}"


def _describe_dow(m: dict[str, str]) -> str:
    groups: dict[str, list[str]] = {}
    for wd in WEEKDAYS:
        groups.setdefault(m[wd], []).append(WEEKDAY_LABEL[wd])
    return "; ".join(f"{v} on {'/'.join(days)}" for v, days in groups.items())


def _condition_label(condition: str, key: str) -> str:
    if (condition, key) in CONDITION_LABEL:
        return CONDITION_LABEL[(condition, key)]
    if key == "default":
        return "otherwise"
    return f"in {key}" if condition == "season" else f"when the weather is {key}"


def _situation(condition: str, key: str) -> str:
    if condition == "is_workday":
        return "a workday" if key == "true" else "a weekend day"
    if condition == "season":
        return f"in {key}"
    return f"a {key}-weather day"


def describe(rule: "Rule | str", anchor_date: dt.date) -> str:
    """Natural-language description of the full rule, used as the pattern answer."""
    if isinstance(rule, str):
        return rule
    if isinstance(rule, ConstantRule):
        return f"{rule.value} every day"
    if isinstance(rule, WeeklyAlternateRule):
        order = ", then ".join(rule.values)
        return (f"alternates week by week: {order}, repeating "
                f"(7-day blocks starting {_fmt_date(anchor_date)})")
    if isinstance(rule, NDayCycleRule):
        total = sum(s.days for s in rule.segments)
        parts = ", then ".join(f"{s.value} for {s.days} days" for s in rule.segments)
        return f"a {total}-day cycle starting {_fmt_date(anchor_date)}: {parts}, repeating"
    if isinstance(rule, DayOfWeekRule):
        return f"by day of week: {_describe_dow(rule.map)}"
    if isinstance(rule, NestedRule):
        blocks = " | ".join(
            f"block {i + 1}: {describe(b, anchor_date)}" for i, b in enumerate(rule.blocks)
        )
        return (f"a {len(rule.blocks)}-block rotation of {rule.block_days}-day blocks starting "
                f"{_fmt_date(anchor_date)} - {blocks}")
    if isinstance(rule, ConditionalRule):
        parts = "; ".join(
            f"{_condition_label(rule.condition, k)}: {describe(c, anchor_date)}"
            for k, c in rule.cases.items()
        )
        return {"is_workday": "split by workday vs weekend", "season": "split by season",
                "weather": "split by weather"}[rule.condition] + f" - {parts}"
    raise TypeError(f"unknown rule {rule!r}")


def explain(rule: "Rule | str", day: int, date: dt.date, anchor: int, ctx: Context) -> str:
    """Why this rule gives this value on this day, used in prediction reasoning."""
    if isinstance(rule, str):
        return f"so it is {rule}"
    off = _offset(day, anchor)
    value = evaluate(rule, day, date, anchor, ctx)
    if isinstance(rule, ConstantRule):
        return f"it is {value} every day"
    if isinstance(rule, WeeklyAlternateRule):
        block = off // 7
        return (f"{date.isoformat()} is in 7-day block {block + 1} since the anchor "
                f"(block index {block} mod {len(rule.values)} = {block % len(rule.values)}), so {value}")
    if isinstance(rule, NDayCycleRule):
        total = sum(s.days for s in rule.segments)
        return f"it is day {off % total + 1} of the {total}-day cycle, so {value}"
    if isinstance(rule, DayOfWeekRule):
        return f"{date.isoformat()} is a {date.strftime('%A')}, so {value}"
    if isinstance(rule, NestedRule):
        idx = (off // rule.block_days) % len(rule.blocks)
        inner = explain(rule.blocks[idx], day, date, anchor, ctx)
        return f"it falls in block {idx + 1} of the rotation; {inner}"
    if isinstance(rule, ConditionalRule):
        key = ctx.value_of(rule.condition)
        case = _case(rule, ctx)
        inner = explain(case, day, date, anchor, ctx)
        return f"{date.isoformat()} is {_situation(rule.condition, key)}; {inner}"
    raise TypeError(f"unknown rule {rule!r}")


def short_label(rule: "Rule | str") -> str:
    """Compact label when a regime has no hand-written label."""
    if isinstance(rule, str):
        return rule
    if isinstance(rule, ConstantRule):
        return rule.value
    vals = sorted(values_of(rule))
    return f"a {rule.type.replace('_', ' ')} routine of " + ", ".join(vals)
