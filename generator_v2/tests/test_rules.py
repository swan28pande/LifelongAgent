import datetime as dt

import pytest
from pydantic import TypeAdapter

from generator_v2 import rules
from generator_v2.schema import Rule

START = dt.date(2026, 3, 1)  # a Sunday
R = TypeAdapter(Rule)


def date_of(day: int) -> dt.date:
    return START + dt.timedelta(days=day - 1)


def ev(rule: dict, day: int, anchor: int = 1, city: str | None = None) -> str:
    d = date_of(day)
    return rules.evaluate(R.validate_python(rule), day, d, anchor, rules.context_for(d, city))


def test_constant():
    assert ev({"type": "constant", "value": "tea"}, 400) == "tea"


def test_weekly_alternate_blocks_of_seven():
    rule = {"type": "weekly_alternate", "values": ["a", "b"]}
    assert [ev(rule, d) for d in (1, 7, 8, 14, 15)] == ["a", "a", "b", "b", "a"]


def test_weekly_alternate_anchors_to_regime_start_not_dataset_start():
    rule = {"type": "weekly_alternate", "values": ["a", "b"]}
    # Regime starting on day 12 (a Thursday): blocks run 12–18, 19–25, …
    assert ev(rule, 12, anchor=12) == "a"
    assert ev(rule, 18, anchor=12) == "a"
    assert ev(rule, 19, anchor=12) == "b"


def test_n_day_cycle_segments_and_anchor():
    rule = {"type": "n_day_cycle", "segments": [{"value": "x", "days": 3}, {"value": "y", "days": 2}]}
    assert [ev(rule, d, anchor=10) for d in range(10, 21)] == list("xxxyyxxxyyx")


def test_day_of_week_uses_calendar_weekday():
    m = {"mon": "M", "tue": "T", "wed": "W", "thu": "R", "fri": "F", "sat": "S", "sun": "U"}
    rule = {"type": "day_of_week", "map": m}
    assert ev(rule, 1) == "U"      # 2026-03-01 is a Sunday
    assert ev(rule, 2) == "M"
    assert ev(rule, 2, anchor=2) == "M"


def test_day_of_week_requires_all_days():
    with pytest.raises(Exception):
        R.validate_python({"type": "day_of_week", "map": {"mon": "a"}})


def test_nested_week_parity_by_day_of_week():
    a = {d: "A-" + d for d in rules.WEEKDAYS}
    b = {d: "B-" + d for d in rules.WEEKDAYS}
    rule = {"type": "nested", "blocks": [{"type": "day_of_week", "map": a},
                                         {"type": "day_of_week", "map": b}]}
    assert ev(rule, 2) == "A-mon"
    assert ev(rule, 9) == "B-mon"
    assert ev(rule, 16) == "A-mon"
    # Anchored at day 5: block 1 covers 5–11, block 2 covers 12–18.
    assert ev(rule, 9, anchor=5) == "A-mon"
    assert ev(rule, 16, anchor=5) == "B-mon"


def test_conditional_on_workday_and_nested_season():
    rule = {"type": "conditional", "condition": "is_workday", "cases": {
        True: {"type": "conditional", "condition": "season",
               "cases": {"winter": "bus", "default": "bike"}},
        False: "none"}}
    assert ev(rule, 1) == "none"              # Sunday
    assert ev(rule, 2) == "bike"              # Monday in March (spring)
    assert ev(rule, 2 + 7 * 40) == "bus"      # a Monday in December


def test_conditional_on_weather_uses_city_climate():
    rule = {"type": "conditional", "condition": "weather",
            "cases": {"snowy": "gym", "default": "outdoor run"}}
    jan = (dt.date(2027, 1, 12) - START).days + 1
    assert ev(rule, jan, city="Denver, Colorado") == "gym"
    assert ev(rule, jan, city="Austin, Texas") == "outdoor run"


def test_day_before_anchor_is_an_error():
    with pytest.raises(ValueError):
        ev({"type": "weekly_alternate", "values": ["a", "b"]}, 3, anchor=5)


def test_phase_key_and_describe_are_consistent():
    rule = R.validate_python({"type": "n_day_cycle",
                              "segments": [{"value": "x", "days": 2}, {"value": "y", "days": 1}]})
    ctx = rules.context_for(date_of(4))
    assert rules.phase_key(rule, 4, date_of(4), 1, ctx) == (0,)
    assert "3-day cycle" in rules.describe(rule, START)
    assert "day 1 of the 3-day cycle" in rules.explain(rule, 4, date_of(4), 1, ctx)
