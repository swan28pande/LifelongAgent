import datetime as dt
from functools import lru_cache

import pytest

from generator_v2 import config, probing, qa, simulator
from generator_v2.schema import load_ladder, load_spec


@lru_cache
def generated(uid):
    spec = load_spec(config.PERSONA_DIR / f"{uid}.yaml", load_ladder(config.LADDER_PATH))
    world = simulator.simulate(spec)
    return spec, world, qa.build_qa(spec, world)


@pytest.fixture(scope="module")
def monthly():
    return probing.build_probes(*generated("u5"))


def test_months_include_the_final_partial_month():
    months = probing._month_boundaries(dt.date(2026, 3, 1), 730)
    assert len(months) == 24
    assert months[-1] == {"day": 730, "date": "2028-02-28", "month": "2028-02"}
    assert probing._month_boundaries(dt.date(2028, 2, 20), 3) == [
        {"day": 3, "date": "2028-02-22", "month": "2028-02"}]


@pytest.mark.parametrize("kind", ["fact_current", "abstention"])
def test_current_hobbies_include_piano_in_january(kind):
    spec, world, pool = generated("u5")
    item = next(q for q in pool if q.type == kind and (
        q.domain == "hobbies" or "hobbies" in q.tags))
    builder = qa.QABuilder(spec, world)
    january = builder.at_viewpoint(item, 337)
    may = builder.at_viewpoint(item, 457)
    assert january.answer == "bouldering, piano, trail running"
    assert may.answer == "bouldering, trail running"
    assert 420 not in january.evidence_days
    assert 420 in may.evidence_days
    assert builder.at_viewpoint(item, 337) == january  # later grounding never mutates earlier gold


def test_delayed_historical_disclosures_and_removals_are_cited():
    spec, world, _ = generated("u5")
    builder = qa.QABuilder(spec, world)
    assert world.days[471].active_facts["diet"] == ["vegetarian"]
    assert 476 in builder.observations.fact_days("diet", 472)
    assert 476 > 472  # retrospective evidence is allowed, but a probe must wait for it
    assert 420 not in builder._fact_evidence("hobbies", "piano")
    assert 420 in builder.observations.fact_days("hobbies", 457)


def test_unsupported_recall_is_not_generated():
    for uid, date, domain in (("u5", "2026-05-09", "exercise"),
                              ("u4", "2027-12-19", "music")):
        _, _, pool = generated(uid)
        assert not any(q.type == "recall" and q.domain == domain and date in q.question for q in pool)


@pytest.mark.parametrize("uid,regime,cutoff,alternatives", [
    ("u3", "lunch.r1", 122, [91, 92]),
    ("u3", "commute.r3", 671, [640]),
    ("u4", "workout.r3", 365, [331, 332, 333, 335]),
    ("u4", "breakfast.r3", 487, [459]),
    ("u4", "music.r3", 610, [603, 604]),
    ("u5", "bedtime.r1", 122, [99]),
    ("u5", "music.r1", 122, [117, 118]),
    ("u5", "dinner.r1", 184, [151, 152, 153, 154, 155, 156]),
    ("u5", "music.r2", 214, [197, 198, 199]),
    ("u5", "exercise.r4", 365, [334]),
    ("u5", "music.r3", 365, [346, 347, 348, 349, 350]),
    ("u5", "commute.r3", 426, [391, 392, 393, 394, 395, 396, 397, 398, 399]),
    ("u5", "reading.r4", 457, [453, 454]),
    ("u5", "commute.r4", 549, [524]),
    ("u5", "bedtime.r3", 549, [521, 522]),
    ("u5", "exercise.r5", 579, [556, 557, 558, 559]),
])
def test_observation_compatible_date_alternatives_are_accepted(uid, regime, cutoff, alternatives):
    spec, world, _ = generated(uid)
    builder = qa.QABuilder(spec, world)
    dates, evidence = builder.observations.change(regime, cutoff)
    assert set(alternatives) <= set(dates)
    assert max(dates) <= cutoff
    assert max(evidence) <= cutoff
    assert builder.observations.distinguishing_day(next(r for r in world.regimes if r.id == regime)) in evidence


@pytest.mark.parametrize("uid,rid", [("u4", "workout.r3"), ("u5", "exercise.r4")])
def test_change_waits_for_a_distinguishing_observation(uid, rid):
    spec, world, _ = generated(uid)
    builder = qa.QABuilder(spec, world)
    assert builder.observations.change(rid, 337) == ([], [])
    reg = next(r for r in world.regimes if r.id == rid)
    first = builder.observations.distinguishing_day(reg)
    dates, evidence = builder.observations.change(rid, first)
    assert max(dates) <= first
    assert max(evidence) <= first


def test_pattern_questions_cover_every_described_value():
    for uid in ("u1", "u2", "u3", "u4", "u5"):
        _, world, pool = generated(uid)
        for q in pool:
            if q.type.startswith("pattern"):
                reg = next(r for r in world.regimes if r.start <= min(q.evidence_days) <= r.end
                           and r.domain == q.domain)
                values = {world.days[n - 1].preferences[q.domain].base_value for n in q.evidence_days}
                assert qa.rules.values_of(reg.rule) <= values


def test_completed_durations_include_closure_observations():
    for uid in ("u3", "u4", "u5"):
        spec, world, pool = generated(uid)
        builder = qa.QABuilder(spec, world)
        for q in pool:
            if q.type != "duration" or "fact" in q.tags:
                continue
            rid = next(t.split(":", 1)[1] for t in q.tags if t.startswith("regime:"))
            reg = next(r for r in world.regimes if r.id == rid)
            if reg.end < world.num_days:
                after = builder.regime_at(reg.domain, reg.end + 1)
                assert builder.observations.distinguishing_day(after) in q.evidence_days


def test_monthly_gold_is_grounded_at_its_own_probe(monthly):
    spec, world, pool = generated("u5")
    assert probing.build_probes(spec, world, pool) == monthly
    builder = qa.QABuilder(spec, world)
    original = {q.id: q for q in pool}
    all_questions = []
    for probe in monthly["probes"]:
        assert probe["questions"]
        for q in probe["questions"]:
            all_questions.append(q)
            assert q["viewpoint_day"] == probe["viewpoint_day"]
            grounded = builder.at_viewpoint(original[q["original_id"]], q["viewpoint_day"])
            assert grounded.answer == q["answer"]
            assert grounded.accept == q["accept"]
            assert grounded.evidence_days == q["evidence_days"]
            assert max(q["evidence_days"], default=1) <= q["viewpoint_day"]
            assert probing._available_day(grounded, world.start_date) <= q["viewpoint_day"]
    assert len(all_questions) == monthly["total_questions"] == 200
    assert len({q["id"] for q in all_questions}) == 200
    assert "pattern_current" in {q["type"] for q in all_questions}
    assert "prediction" not in {q["type"] for q in all_questions}
    abstentions = [q for q in all_questions if q["type"] == "abstention"]
    assert any(q["answer_type"] == "abstain" and not q["evidence_days"] for q in abstentions)
    assert any(q["answer_type"] != "abstain" and q["evidence_days"] for q in abstentions)
