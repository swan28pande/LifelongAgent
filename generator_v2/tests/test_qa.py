from collections import Counter

import pytest

from generator_v2 import qa, rules
from generator_v2.schema import QAItem


@pytest.fixture(scope="module")
def q1(u1, w1):
    return qa.build_qa(u1, w1)


@pytest.fixture(scope="module")
def q5(u5, w5):
    return qa.build_qa(u5, w5)


def test_qa_is_deterministic(u5, w5, q5):
    assert [q.model_dump() for q in qa.build_qa(u5, w5)] == [q.model_dump() for q in q5]


def test_counts_in_target_range(q1, q5):
    assert 300 <= len(q1) <= 400
    assert 300 <= len(q5) <= 400


def test_guard_passes_and_ids_and_questions_are_unique(q1, q5):
    for items in (q1, q5):
        assert qa.guard_errors(items) == []
        assert len({q.id for q in items}) == len(items)
        assert len({q.question for q in items}) == len(items)


def test_guard_catches_a_dominant_answer():
    items = [QAItem(id=f"x{i}", type="recall", capability="c", domain="coffee", difficulty_user=1,
                    question=f"q{i}", answer="tea", answer_type="value", accept=["tea" if i < 4 else "milk"],
                    evidence_days=[1], recency_distance_days=1, grading="exact", viewpoint_day=731)
             for i in range(5)]
    assert qa.guard_errors(items)


def test_recall_answers_match_world_state(w5, q5):
    for q in (q for q in q5 if q.type == "recall"):
        date = q.question.rsplit(" ", 1)[1].rstrip("?")
        day = next(d for d in w5.days if d.date.isoformat() == date)
        assert q.answer == day.preferences[q.domain].value
        assert ("inferred" in q.tags) == (not day.preferences[q.domain].mentioned)


def test_recall_and_prediction_spread_over_weekdays(q1, q5):
    """v1 always asked on Thursdays; v2 must cover the week."""
    for items in (q1, q5):
        for qtype in ("recall", "prediction"):
            by_domain: dict[str, list[str]] = {}
            for q in items:
                if q.type == qtype:
                    by_domain.setdefault(q.domain, []).append(q.question.split(" on ")[-1].split(",")[0])
            for dom, days in by_domain.items():
                if len(days) >= 7:
                    assert len(set(days)) >= 5, (qtype, dom, Counter(days))


def test_prediction_answers_follow_the_final_regime(u5, w5, q5):
    for q in (q for q in q5 if q.type == "prediction"):
        date = q.question.split(", ")[1].split("?")[0]
        day = (__import__("datetime").date.fromisoformat(date) - u5.start_date).days + 1
        assert day > u5.num_days
        reg = next(r for r in w5.regimes_for(q.domain) if r.end == u5.num_days)
        d = u5.date_of(day)
        final_city = w5.days[-1].active_facts["city"][0]
        assert q.accept[0] == rules.evaluate(reg.rule, day, d, reg.anchor, rules.context_for(d, final_city))


def test_change_detection_accepts_the_effective_date(u5, w5, q5):
    starts = {r.domain: set() for r in w5.regimes}
    for r in w5.regimes:
        starts[r.domain].add(u5.date_of(r.start).isoformat())
    for q in (q for q in q5 if q.type == "change_detection"):
        assert q.answer[:10] in q.accept and q.answer[:10] in starts[q.domain]


def test_uncaused_shifts_have_no_reason(w5, q5):
    uncaused = {(r.domain, r.start) for r in w5.regimes if r.visibility == "uncaused" and r.kind != "reversion"}
    answers = [q for q in q5 if q.type == "attribution" and q.answer_type == "abstain"]
    assert len(answers) == len(uncaused)
    assert all(q.answer == "No reason was stated." for q in answers)


def test_yes_no_types_have_positive_controls(q5):
    for qtype in ("exception_vs_shift", "distractor_probe", "reversion", "other_person"):
        keys = Counter(q.accept[0] for q in q5 if q.type == qtype and q.answer_type == "yes_no")
        assert keys["yes"] > 0 and keys["no"] > 0, qtype


def test_fact_current_reflects_latest_true_value(q1):
    cur = {q.domain: q.answer for q in q1 if q.type == "fact_current"}
    assert cur["job"] == "account manager at the logistics firm"
    assert cur["city"] == "Chicago, Illinois"


def test_viewpoint_and_recency(u5, q5):
    for q in q5:
        assert q.viewpoint_day == u5.num_days + 1
        assert q.recency_distance_days >= 0
        assert all(1 <= d <= u5.num_days for d in q.evidence_days)


def test_u1_has_no_types_its_ladder_excludes(q1):
    types = {q.type for q in q1}
    assert not types & {"exception_vs_shift", "reversion", "distractor_probe", "other_person"}
