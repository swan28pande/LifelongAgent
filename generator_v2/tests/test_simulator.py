from generator_v2 import rules, simulator

from .conftest import make_spec


def test_same_seed_same_world(u5, w5):
    again = simulator.simulate(u5)
    assert again.model_dump_json() == w5.model_dump_json()


def test_different_seed_different_sessions():
    a = simulator.simulate(make_spec("u5"))
    b = simulator.simulate(make_spec("u5", lambda d: d.update(seed=999)))
    assert [d.has_session for d in a.days] != [d.has_session for d in b.days]


def test_statements_land_on_session_days_after_they_take_effect(w5):
    sessions = {d.day for d in w5.days if d.has_session}
    for s in w5.statements:
        assert s.stated_day in sessions
        assert s.stated_day >= s.effective_day
        assert s.retrospective == (s.stated_day > s.effective_day and s.kind != "background_fact")
    assert any(s.retrospective for s in w5.statements)


def test_retraction_is_true_from_the_original_day_but_known_only_after_correction(w5):
    biscuit = next(s for s in w5.statements if s.id == "pet.biscuit")
    fix = next(s for s in w5.statements if s.kind == "retraction" and s.entity == "pets"
               and s.true_value == "labrador named Biscuit")
    assert w5.days[95 - 1].active_facts["pets"] == ["labrador named Biscuit"]
    assert "golden retriever named Biscuit" in w5.days[biscuit.stated_day - 1].known_facts["pets"]
    assert w5.days[fix.stated_day - 1].known_facts["pets"][0] == "labrador named Biscuit"


def test_corrections_come_in_a_later_session_than_the_mistake(u5, w5):
    stated = {s.id: s.stated_day for s in w5.statements}
    for rid, target in simulator.retraction_order(u5).items():
        assert stated[rid] > stated[target]


def test_exceptions_in_a_domain_are_spaced_out(w5):
    from generator_v2 import config
    for dom in w5.days[0].preferences:
        days = [d.day for d in w5.days if d.preferences[dom].is_exception]
        assert all(b - a > config.EXCEPTION_MIN_SPACING for a, b in zip(days, days[1:]))


def test_cause_text_uses_the_corrected_value_but_the_statement_uses_what_was_said(w5):
    biscuit = next(s for s in w5.statements if s.id == "pet.biscuit")
    assert biscuit.text == "adopted a golden retriever named Biscuit"
    exercise = next(r for r in w5.regimes if r.id == "exercise.r1")
    assert exercise.cause_text == "adopted a labrador named Biscuit"


def test_single_valued_facts_update_in_place(w1):
    assert w1.days[319 - 1].active_facts["job"] == ["account coordinator at a logistics firm"]
    assert w1.days[320 - 1].active_facts["job"] == ["account manager at the logistics firm"]


def test_reversion_resumes_the_base_regime_phase(u5, w5):
    rev = next(r for r in w5.regimes if r.id == "exercise.r2.revert")
    base = next(r for r in w5.regimes if r.id == "exercise.r1")
    assert (rev.start, rev.end, rev.anchor, rev.rule) == (215, 261, base.anchor, base.rule)
    for d in range(rev.start, rev.start + 14):
        day = w5.days[d - 1]
        expected = rules.evaluate(base.rule, d, day.date, base.anchor, rules.context_for(day.date))
        assert day.preferences["exercise"].base_value == expected


def test_values_follow_the_active_regime(w1):
    coffee = [d.preferences["coffee"].value for d in w1.days]
    assert coffee[0:7] == ["oat milk latte"] * 7
    assert coffee[7] == "black coffee"
    assert coffee[204] == "matcha latte"     # regime starts day 205, anchored there


def test_exceptions_are_observed_and_differ_from_the_rule(w5):
    exc = [(d, p) for d in w5.days for p in d.preferences.values() if p.is_exception]
    assert exc
    for d, p in exc:
        assert d.has_session and p.mentioned and p.value != p.base_value and p.exception_reason


def test_every_new_regime_gets_one_matching_instruction_on_first_observation(w5):
    kinds = {"explicit": "link_explicitly", "implicit": "mention_without_link", "uncaused": "do_not_mention"}
    for r in w5.regimes:
        if r.kind == "initial":
            continue
        day = w5.days[r.first_mention_day - 1]
        ins = [c for c in day.cause_instructions if c.regime_id == r.id]
        assert ins and ins[0].instruction == kinds[r.visibility]
        if r.visibility == "uncaused":
            assert ins[0].cause_text is None
        assert day.preferences[r.domain].mentioned


def test_every_regime_is_observed_enough(w5):
    from generator_v2 import config
    assert all(len(r.mention_days) >= config.MIN_REGIME_MENTIONS for r in w5.regimes)


def test_days_without_sessions_carry_no_instructions(w5):
    for d in w5.days:
        if not d.has_session:
            assert not (d.cause_instructions or d.facts_to_state_today or d.event_updates_today)
            assert not any(p.mentioned for p in d.preferences.values())
