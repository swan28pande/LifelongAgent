import json

from generator_v2 import conversation, validator
from generator_v2.tests.conftest import make_spec
from generator_v2.simulator import simulate


def good_extraction(check):
    return {
        "preferences": {d: exp for d, (exp, _) in check.preferences.items()},
        "statements": {sid: True for sid in check.statements},
        "causes": {cid: {"event_mentioned": True, "linked": c["instruction"] == "link_explicitly",
                         "reason_given": c["instruction"] == "link_explicitly"}
                   for cid, c in check.causes.items()},
        "inventions": [],
    }


def test_matching_extraction_passes(u1, w1):
    check = validator.build_check(u1, w1, w1.days[204])
    assert validator.diff(check, good_extraction(check)) == []


def test_wrong_absent_and_leaked_values_fail(u5, w5):
    day = next(d for d in w5.days if d.has_session
               and any(p.mentioned for p in d.preferences.values())
               and any(not p.mentioned for p in d.preferences.values()))
    check = validator.build_check(u5, w5, day)
    on = next(d for d, (e, _) in check.preferences.items() if e != "not mentioned")
    off = next(d for d, (e, _) in check.preferences.items() if e == "not mentioned")
    ext = good_extraction(check)
    ext["preferences"][on] = "other"
    ext["preferences"][off] = check.preferences[off][1][0]
    kinds = {f.split(":")[0] for f in validator.diff(check, ext)}
    assert {"pref_wrong", "pref_leak"} <= kinds


def test_cause_instructions_are_enforced(u1, w1):
    check = validator.build_check(u1, w1, w1.days[204])     # explicit coffee link on day 205
    ext = good_extraction(check)
    ext["causes"]["c0"]["linked"] = False
    assert any(f.startswith("cause_not_linked") for f in validator.diff(check, ext))


def test_uncaused_shift_must_not_get_a_reason(u5, w5):
    reg = next(r for r in w5.regimes if r.visibility == "uncaused" and r.kind == "shift")
    day = w5.days[reg.first_mention_day - 1]
    check = validator.build_check(u5, w5, day)
    cid = next(c for c, v in check.causes.items() if v["instruction"] == "do_not_mention")
    ext = good_extraction(check)
    ext["causes"][cid]["reason_given"] = True
    assert any(f.startswith("reason_given") for f in validator.diff(check, ext))


def test_missing_statement_and_invention_fail(u1, w1):
    check = validator.build_check(u1, w1, w1.days[0])
    ext = good_extraction(check)
    sid = next(iter(check.statements))
    ext["statements"][sid] = False
    ext["inventions"] = ["has a brother named Max"]
    kinds = {f.split(":")[0] for f in validator.diff(check, ext)}
    assert {"statement_missing", "invention"} <= kinds


def test_writer_prompt_lists_absent_topics_and_hides_the_rule(u5, w5):
    day = next(d for d in w5.days if d.has_session and not all(p.mentioned for p in d.preferences.values()))
    prompt = conversation.writer_prompt(u5, w5, day, [])
    absent = [p.noun for p in u5.preferences if not day.preferences[p.domain].mentioned]
    assert all(n in prompt.split("MUST NOT COME UP:")[1].split("\n")[0] for n in absent)
    for r in w5.regimes:
        assert r.label not in prompt


def test_writer_prompt_uses_only_facts_known_before_the_day(u5, w5):
    s = next(s for s in w5.statements if s.id == "pet.biscuit")
    prompt = conversation.writer_prompt(u5, w5, w5.days[s.stated_day - 1], [])
    known = prompt.split("ALREADY KNOWS")[1].split("TODAY'S CHOICES")[0]
    assert "Biscuit" not in known
    assert "Biscuit" in prompt.split("REQUIRED UPDATES")[1]


def test_shape_errors():
    assert conversation.shape_errors([]) == ["format: no turns returned"]
    turns = [{"speaker": "user", "text": "hi"}] * 5
    assert any("5 turns" in e for e in conversation.shape_errors(turns))
    turns = [{"speaker": "bot", "text": "hi"}] * 14
    assert any("speaker" in e for e in conversation.shape_errors(turns))


def test_validator_has_the_session_date_and_disclosed_fact_change_history():
    spec = make_spec("u2")
    world = simulate(spec)
    day = world.days[120]  # June 29, following the June 28 tennis-club update
    check = validator.build_check(spec, world, day)
    prompt = validator.validator_prompt(spec, check, [])
    assert "DATE: 2026-06-29" in prompt
    assert "joined a tennis club (happened on 2026-06-28)" in prompt


def test_fidelity_preserves_known_usage_when_the_log_is_an_lfs_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(conversation.config, "OUTPUT_DIR", tmp_path)
    base = tmp_path / "u1"
    (base / "sessions").mkdir(parents=True)
    (base / "sessions/2026-03-01.json").write_text(json.dumps({
        "date": "2026-03-01", "turns": [], "passed_validation": True,
        "attempts": 5, "failures": [], "first_try_failures": ["pref_wrong: old failure"],
    }))
    (base / "llm_log.jsonl").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n")
    (base / "fidelity.json").write_text(json.dumps({"usage": {"gemini.calls": 42}}))
    result = conversation.fidelity("u1")
    assert result["flagged"] == []
    assert result["passed_after_retries"] == 1
    assert result["passed_first_try"] == 0
    assert result["usage"] == {"gemini.calls": 42}
    assert "unavailable LFS" in result["usage_source"]
