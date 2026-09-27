from generator_v2 import checks, simulator

from .conftest import make_spec


def domain(data, name):
    return next(p for p in data["preferences"] if p["domain"] == name)


def spec_errors(mutate):
    return checks.check_spec(make_spec("u5", mutate))


def test_personas_pass_all_checks(u1, w1, u5, w5):
    for spec, world in ((u1, w1), (u5, w5)):
        assert checks.check_spec(spec) == []
        assert checks.check_world(spec, world) == []
        assert checks.knob_errors(checks.compare_knobs(spec, checks.measure_knobs(spec, world))) == []


def test_lag_must_match_cause_day():
    def m(d):
        domain(d, "exercise")["regimes"][1]["cause"]["lag_days"] = 4
    assert any("!= cause day" in e for e in spec_errors(m))


def test_cause_after_shift_is_rejected():
    def m(d):
        domain(d, "music")["regimes"][2]["start"] = 190
    assert any("after the shift" in e for e in spec_errors(m))


def test_remove_of_inactive_value_is_rejected():
    def m(d):
        hobbies = next(f for f in d["facts"] if f["entity"] == "hobbies")
        hobbies["changes"].append({"day": 400, "op": "remove", "value": "knitting"})
    assert any("not active" in e for e in spec_errors(m))


def test_single_valued_fact_cannot_add():
    def m(d):
        job = next(f for f in d["facts"] if f["entity"] == "job")
        job["changes"].append({"day": 400, "op": "add", "value": "second job"})
    assert any("only use op=set" in e for e in spec_errors(m))


def test_temporary_regime_needs_end():
    def m(d):
        del domain(d, "dinner")["regimes"][4]["end"]
    assert any("needs an end" in e for e in spec_errors(m))


def test_temporary_regime_must_leave_room_to_revert():
    def m(d):
        domain(d, "dinner")["regimes"][4]["end"] = 675
    assert any("reversion lasts" in e for e in spec_errors(m))


def test_regimes_must_be_ordered():
    def m(d):
        domain(d, "reading")["regimes"][2]["start"] = 100
    assert any("strictly increasing" in e for e in spec_errors(m))


def test_repeated_rule_is_not_a_shift():
    def m(d):
        regs = domain(d, "reading")["regimes"]
        regs[2]["rule"] = regs[1]["rule"]
    assert any("not a real shift" in e for e in spec_errors(m))


def test_distractor_near_uncaused_shift_needs_confounder_flag():
    def near(d):
        d["distractors"][0]["day"] = 258        # exercise uncaused shift on day 262
    assert any("uncaused exercise shift" in e for e in spec_errors(near))

    def flagged(d):
        near(d)
        d["distractors"][0]["confounder"] = True
    assert not any("uncaused" in e for e in spec_errors(flagged))


def test_unknown_event_ref_is_rejected():
    def m(d):
        domain(d, "bedtime")["regimes"][1]["cause"]["event_ref"] = "nonexistent"
    assert any("unknown event_ref" in e for e in spec_errors(m))


def test_conditional_rule_must_cover_every_case():
    def m(d):
        domain(d, "music")["regimes"][0]["rule"]["cases"] = {True: "lo-fi beats"}
    assert any("lacks cases" in e for e in spec_errors(m))


def test_knob_mismatch_is_reported(u5, w5):
    spec = make_spec("u5")
    spec.difficulty = spec.difficulty.model_copy(update={"distractors": 3})
    errs = checks.knob_errors(checks.compare_knobs(spec, checks.measure_knobs(spec, w5)))
    assert any("distractors" in e for e in errs)


def test_world_check_catches_an_unobserved_regime(u5):
    world = simulator.simulate(u5)
    world.regimes[3].mention_days = world.regimes[3].mention_days[:2]
    assert any("observed 2 times" in e for e in checks.check_world(u5, world))
