#!/usr/bin/env python3
"""Offline, read-only audit of non-recall gold answers at monthly viewpoints.

Run from anywhere with the repository's Python dependencies available:
  PYTHONDONTWRITEBYTECODE=1 python .research/v2-probing-audit/gold/audit_gold.py

Only audit artifacts in this directory are written. This does not regenerate data,
invoke an LLM, use a network, or interpret validator flags as evidence failures.
"""
from __future__ import annotations

import bisect
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from generator_v2 import rules
from generator_v2.schema import WorldState, load_ladder, load_spec


CAPABILITIES = {
    "recall": "episodic_recall", "pattern_current": "pattern_abstraction",
    "pattern_at_time": "pattern_abstraction", "prediction": "pattern_extrapolation",
    "change_detection": "temporal_change", "attribution": "causal_reasoning",
    "exception_vs_shift": "exception_handling", "reversion": "temporal_change",
    "fact_current": "knowledge_update", "fact_at_time": "temporal_fact",
    "fact_history": "knowledge_update", "event_status": "event_tracking",
    "duration": "temporal_arithmetic", "distractor_probe": "causal_reasoning",
    "other_person": "entity_disambiguation", "abstention": "abstention",
}
ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MONTH = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December) (\d{4})\b")


class Source:
    """Index JSON pointers to exact value lines, independently of repeated values."""
    def __init__(self, path: Path):
        self.path = path
        self.text = path.read_text()
        self.data = json.loads(self.text)
        self.positions = {}
        self.newlines = [-1] + [m.start() for m in re.finditer("\n", self.text)]
        decoder = json.JSONDecoder()

        def ws(i):
            while i < len(self.text) and self.text[i].isspace():
                i += 1
            return i

        def parse(i, pointer):
            i = ws(i)
            self.positions[pointer] = i
            if self.text[i] == "{":
                i = ws(i + 1)
                if self.text[i] == "}":
                    return i + 1
                while True:
                    key, i = decoder.raw_decode(self.text, i)
                    i = ws(i)
                    assert self.text[i] == ":"
                    key = str(key).replace("~", "~0").replace("/", "~1")
                    i = ws(parse(i + 1, pointer + "/" + key))
                    if self.text[i] == "}":
                        return i + 1
                    assert self.text[i] == ","
                    i = ws(i + 1)
            if self.text[i] == "[":
                i = ws(i + 1)
                if self.text[i] == "]":
                    return i + 1
                index = 0
                while True:
                    i = ws(parse(i, pointer + "/" + str(index)))
                    if self.text[i] == "]":
                        return i + 1
                    assert self.text[i] == ","
                    i, index = ws(i + 1), index + 1
            _, end = decoder.raw_decode(self.text, i)
            return end

        parse(0, "")

    def loc(self, pointer=""):
        offset = self.positions[pointer]
        return {"file": str(self.path.relative_to(ROOT)), "line": bisect.bisect_left(self.newlines, offset), "pointer": pointer}


def joined(values):
    return ", ".join(values) if values else "none"


def fmt(date):
    return f"{date.strftime('%A')}, {date.isoformat()}"


def rule_items(reg):
    vals = sorted(rules.values_of(reg.rule))
    if len(vals) == 1:
        return vals[0]
    shown = " / ".join(vals) if len(vals) <= 5 else " / ".join(vals[:4]) + " etc."
    return f"the {shown} routine"


def intervals(world, entity, cutoff):
    opened, out = {}, []
    for day in world.days[:cutoff]:
        values = day.active_facts[entity]
        for value in list(opened):
            if value not in values:
                out.append((value, opened.pop(value), day.day - 1))
        for value in values:
            opened.setdefault(value, day.day)
    out += [(value, start, cutoff) for value, start in opened.items()]
    return sorted(out, key=lambda x: (x[1], x[0]))


def future_dates(text, viewpoint):
    out = []
    for m in ISO.finditer(text):
        date = dt.date.fromisoformat(m.group())
        if date > viewpoint:
            out.append({"text": m.group(), "date": date.isoformat(), "offset": m.start(), "format": "iso_date"})
    for m in MONTH.finditer(text):
        date = dt.datetime.strptime(m.group(), "%B %Y").date()
        if date > viewpoint:
            out.append({"text": m.group(), "date": date.isoformat(), "offset": m.start(), "format": "named_month"})
    return out


def main():
    all_coverage, issues, uncertainties, observation_limits, diagnostics, selection, fingerprints = [], [], [], [], [], [], {}
    type_totals, user_totals = collections.Counter(), collections.Counter()
    total_rows = 0
    ladder_path = ROOT / "generator_v2/personas/difficulty_ladder.yaml"
    ladder = load_ladder(ladder_path)
    input_paths = [ladder_path] + [ROOT / p for p in (
        "generator_v2/qa.py", "generator_v2/probing.py", "generator_v2/rules.py",
        "generator_v2/simulator.py", "generator_v2/DECISIONS.md", "generator_v2/schema.py",
        "generator_v2/conversation.py", "experiments/core/grading.py",
    )]

    for user_number in range(1, 6):
        uid = f"u{user_number}"
        spec_path = ROOT / f"generator_v2/personas/{uid}.yaml"
        spec = load_spec(spec_path, ladder)
        base = ROOT / f"datasets/v2/{uid}"
        world_src, probe_src, pool_src = [Source(base / f) for f in ("world_state.json", "probing_questions.json", "qa_pool.json")]
        world = WorldState.model_validate(world_src.data)
        pool = {q["id"]: (i, q) for i, q in enumerate(pool_src.data)}
        statements = {s.id: s for s in world.statements}
        statement_indexes = {s.id: i for i, s in enumerate(world.statements)}
        regime_indexes = {r.id: i for i, r in enumerate(world.regimes)}
        facts = {f.entity: f for f in spec.facts}
        nouns = {p.domain: p.noun for p in spec.preferences}
        rows = [(ip, iq, q) for ip, probe in enumerate(probe_src.data["probes"]) for iq, q in enumerate(probe["questions"])]
        input_paths += [spec_path, world_src.path, probe_src.path, pool_src.path]

        def date_of(day):
            return world.start_date + dt.timedelta(days=day - 1)

        def day_of(date):
            return (date - world.start_date).days + 1

        def describe(reg):
            return rules.describe(reg.rule, date_of(reg.anchor))

        def reg_at(domain, day):
            return next(r for r in world.regimes_for(domain) if r.start <= day <= r.end)

        def reg_locs(reg):
            pointer = f"/regimes/{regime_indexes[reg.id]}"
            return [world_src.loc(pointer + "/" + field) for field in ("id", "start", "end", "rule", "first_mention_day")]

        def stat_locs(ids):
            return [world_src.loc(f"/statements/{statement_indexes[sid]}") for sid in ids]

        def stated(sid):
            if sid in statements:
                return statements[sid].stated_day
            event = next(e for e in spec.events if e.chain_id == sid)
            return statements[event.states[0].id].stated_day

        def evaluate(reg, day):
            d = world.days[day - 1]
            return rules.evaluate(reg.rule, day, d.date, reg.anchor, rules.context_for(d.date, (d.active_facts.get("city") or [None])[0]))

        for ip, iq, q in rows:
            total_rows += 1
            if q["type"] == "recall":
                continue
            type_totals[q["type"]] += 1
            user_totals[uid] += 1
            viewpoint = q["viewpoint_day"]
            probe_date = dt.date.fromisoformat(q["probe_date"])
            pointer = f"/probes/{ip}/questions/{iq}"
            source = {field: probe_src.loc(pointer + "/" + field) for field in ("id", "question", "answer", "answer_type", "accept", "evidence_days", "viewpoint_day")}
            pool_index, pool_q = pool[q["original_id"]]
            row = {
                "id": q["id"], "user_id": uid, "original_id": q["original_id"],
                "type": q["type"], "domain": q["domain"], "probe_date": q["probe_date"],
                "viewpoint_day": viewpoint, "is_retrospective": q["is_retrospective"],
                "question": q["question"], "observed_answer": q["answer"], "observed_accept": q["accept"],
                "answer_type": q["answer_type"], "grading": q["grading"], "evidence_days": q["evidence_days"],
                "source": source, "pool_source": pool_src.loc(f"/{pool_index}"),
                "checks": [], "issue_categories": [], "uncertainty_categories": [], "observation_limit_categories": [], "evidence_diagnostics": [],
                "interpretation_limits": [], "authoritative_sources": [],
                "future_text": {field: future_dates(q[field] if field != "accept" else json.dumps(q[field]), probe_date) for field in ("question", "answer", "accept")},
            }

            def finding(category, kind, expected, rationale, authority, observed=None):
                target = uncertainties if kind == "uncertainty" else observation_limits if kind == "observation_limit" else issues
                target.append({
                    "id": q["id"], "user_id": uid, "original_id": q["original_id"], "type": q["type"],
                    "category": category, "kind": kind, "probe_date": q["probe_date"], "viewpoint_day": viewpoint,
                    "source": source, "question": q["question"],
                    "observed": observed if observed is not None else {"answer": q["answer"], "accept": q["accept"], "evidence_days": q["evidence_days"]},
                    "expected": expected, "rationale": rationale, "authoritative_sources": authority,
                    "full_prefix_unanswerable": None,
                    "historical_viewpoint_compatible": kind != "temporal_eligibility_error",
                    "inferential_predictability": "Not ruled out by this eligibility check; historical observation and prediction are separate." if kind == "temporal_eligibility_error" else "not assessed",
                })
                row["uncertainty_categories" if kind == "uncertainty" else "observation_limit_categories" if kind == "observation_limit" else "issue_categories"].append(category)

            expected_answer = expected_accept = expected_type = expected_grade = None
            typ, dom = q["type"], q["domain"]
            row["checks"].append("pool_provenance_compared_but_not_used_as_truth")
            row["pool_fields_equal"] = all(q[f] == pool_q[f] for f in ("question", "answer", "answer_type", "accept", "evidence_days", "grading", "type", "domain"))
            row["checks"].append("all_evidence_days_within_probe_and_on_session_days")
            row["evidence_calendar_valid"] = all(1 <= d <= viewpoint and world.days[d - 1].has_session for d in q["evidence_days"])
            assert row["evidence_calendar_valid"], q["id"]
            assert q["capability"] == CAPABILITIES[typ], q["id"]

            if row["future_text"]["question"]:
                finding("explicit_future_target", "temporal_eligibility_error", {"latest_target_date": probe_date.isoformat()},
                        "A retrospective target date/month is beyond the available prefix. The rule or plan may already be known, but that does not establish historical status at a future date.",
                        [{"file": "generator_v2/probing.py", "line": 78}, source["question"], source["viewpoint_day"]])

            if typ in ("fact_current", "fact_at_time"):
                target = viewpoint if typ == "fact_current" else day_of(dt.date.fromisoformat(ISO.search(q["question"]).group()))
                vals = world.days[target - 1].active_facts[dom]
                f = facts[dom]
                expected_answer = joined(vals)
                expected_type, expected_grade = ("value", "exact") if f.cardinality == "single" else ("free_text", "llm_judge")
                expected_accept = [expected_answer, expected_answer.split(",")[0]] if f.cardinality == "single" else [expected_answer]
                row.update(target_day=target, target_date=date_of(target).isoformat(), expected_values=vals,
                           known_values_at_probe=world.days[viewpoint - 1].known_facts[dom])
                row["authoritative_sources"] = [world_src.loc(f"/days/{target - 1}/active_facts/{dom}")]
                row["checks"] += ["gold_against_daily_active_facts_at_semantic_target", "multi_value_cardinality_and_acceptance"]
                relevant = [s for s in world.statements if s.entity == dom and s.kind in ("background_fact", "fact_change") and s.effective_day <= target]
                removals = [s for s in relevant if s.op == "remove" and s.stated_day not in q["evidence_days"]]
                if removals:
                    row["evidence_diagnostics"].append({"category": "uncited_fact_removals", "statement_ids": [s.id for s in removals], "locations": stat_locs([s.id for s in removals]),
                                                       "scope": "citation completeness only; full prefix can contain these updates"})
                later_value_updates = [s for s in world.statements if s.entity == dom and s.kind in ("background_fact", "fact_change") and s.value in vals
                                       and s.effective_day > target and s.stated_day in q["evidence_days"]]
                if later_value_updates:
                    row["evidence_diagnostics"].append({"category": "fact_cites_later_value_update", "target_day": target,
                                                       "statement_ids": [s.id for s in later_value_updates], "locations": stat_locs([s.id for s in later_value_updates]),
                                                       "scope": "The update was known by the probe, but does not directly establish membership at the earlier target; can delay selection unnecessarily."})
                if not vals:
                    relevant_cited = [s.id for s in relevant if s.stated_day in q["evidence_days"]]
                    assert not relevant_cited, (q["id"], relevant_cited)
                    finding("empty_fact_has_no_negative_statement", "observation_limit", {"evidence": "An explicit pet-free statement, or an explicit closed-world task convention."},
                            "The empty set is correct synthetic truth, but evidence [1] is a fallback. No background/fact-change statement asserts the empty set, so absence of mention is not observational proof of no pets.",
                            [world_src.loc(f"/days/{target - 1}/active_facts/{dom}"), {"file": "generator_v2/qa.py", "line": 417}])
                    row["interpretation_limits"].append("Open-world conversational answerability of 'none' is unresolved; synthetic closed-world truth agrees with gold.")

            elif typ == "fact_history":
                ints = intervals(world, dom, viewpoint)
                expected_answer = "; ".join(f"{v} ({date_of(s).isoformat()} to {'now' if e == viewpoint else date_of(e).isoformat()})" for v, s, e in ints)
                expected_type, expected_grade, expected_accept = "free_text", "llm_judge", [ints[-1][0]]
                row["expected_intervals"] = [{"value": v, "start_day": s, "end_day": e, "start_date": date_of(s).isoformat(), "end_date": date_of(e).isoformat(), "ongoing_at_probe": e == viewpoint} for v, s, e in ints]
                row["authoritative_sources"] = stat_locs([s.id for s in world.statements if s.entity == dom and s.kind in ("background_fact", "fact_change") and s.effective_day <= viewpoint])
                row["checks"] += ["prefix_only_fact_intervals", "current_now_interpreted_at_probe", "effective_versus_stated_dates"]
                row["interpretation_limits"].append("A short accept hint is not the complete history contract; llm_judge uses the full reference and requires every history item.")

            elif typ in ("pattern_at_time", "pattern_current"):
                if typ == "pattern_current":
                    start_target = end_target = viewpoint
                elif MONTH.search(q["question"]):
                    month = dt.datetime.strptime(MONTH.search(q["question"]).group(), "%B %Y").date()
                    nxt = dt.date(month.year + month.month // 12, month.month % 12 + 1, 1)
                    start_target, end_target = day_of(month), day_of(nxt - dt.timedelta(days=1))
                else:
                    start_target = end_target = day_of(dt.date.fromisoformat(ISO.search(q["question"]).group()))
                matching = [r for r in world.regimes_for(dom) if r.start <= start_target and r.end >= end_target]
                assert len(matching) == 1, (q["id"], matching)
                reg = matching[0]
                expected_answer = describe(reg)
                if typ == "pattern_current":
                    expected_answer += f" (since {date_of(reg.start).isoformat()})"
                expected_accept, expected_type, expected_grade = [reg.label], "free_text", "llm_judge"
                row.update(target_start_day=start_target, target_end_day=end_target, regime_id=reg.id)
                row["authoritative_sources"] = reg_locs(reg)
                cited_values = {world.days[d - 1].preferences[dom].base_value for d in q["evidence_days"]}
                missing_values = sorted(rules.values_of(reg.rule) - cited_values)
                row["evidence_diagnostics"].append({"category": "pattern_value_coverage", "cited_base_values": sorted(cited_values), "uncited_rule_values": missing_values,
                                                   "scope": "Not proof of unanswerability: more observations occur in the full prefix, and rules can be inferred."})
                row["checks"] += ["entire_named_month_or_point_inside_target_regime", "exact_rule_description_including_reversion_anchor", "accept_hint_matches_regime_label"]
                for target_day in range(start_target, end_target + 1):
                    assert evaluate(reg, target_day) == world.days[target_day - 1].preferences[dom].base_value, (q["id"], target_day)
                row["interpretation_limits"].append("No exhaustive natural-language or information-theoretic proof that three cited observations identify the entire rule. A known rule does not establish its persistence into an unreached target month.")

            elif typ in ("change_detection", "attribution"):
                date = dt.date.fromisoformat(ISO.search(q["question"] if typ == "attribution" else q["answer"]).group())
                reg = next(r for r in world.regimes_for(dom) if r.start == day_of(date))
                first = reg.first_mention_day
                row.update(regime_id=reg.id, effective_start_day=reg.start, first_mention_day=first, visibility=reg.visibility)
                row["authoritative_sources"] = reg_locs(reg)
                if typ == "change_detection":
                    expected_answer = date.isoformat() + (f" (first observed {date_of(first).isoformat()})" if first > reg.start else "")
                    expected_accept = [date_of(d).isoformat() for d in range(reg.start, first + 1)]
                    expected_type, expected_grade = "date", "date_exact"
                    regs = world.regimes_for(dom)
                    prev = regs[regs.index(reg) - 1]
                    old = evaluate(prev, first)
                    new = world.days[first - 1].preferences[dom].value
                    first_difference = next((d for d in reg.mention_days if evaluate(prev, d) != world.days[d - 1].preferences[dom].value), None)
                    row.update(previous_regime_id=prev.id, old_rule_value_at_first_mention=old, new_value_at_first_mention=new,
                               first_distinguishing_mention_day=first_difference)
                    row["checks"] += ["effective_to_first_mention_accepted_date_policy", "previous_and_new_regime_comparison", "first_mention_distinguishability"]
                    if old == new and reg.visibility != "explicit":
                        finding("change_citations_do_not_observe_change", "evidence_gap", {"first_distinguishing_mention_day": first_difference, "first_distinguishing_date": date_of(first_difference).isoformat() if first_difference else None},
                                "The first new-regime mention has the same value the old rule predicts, and the writer is instructed not to announce the change. The cited first mention alone cannot justify 'first observed' as a change boundary. Later prefix observations can support a rule inference; the gold effective start still matches synthetic truth.",
                                reg_locs(reg) + [{"file": "generator_v2/conversation.py", "line": 88}, {"file": "generator_v2/qa.py", "line": 256}])
                        # Test all declared observations, not just the two cited days.
                        # Keep subsequent regimes/facts/exceptions fixed; shift the old/new
                        # boundary. Anchor each new cyclic rule at its candidate start.
                        lower = reg.start
                        lag_min, lag_max = spec.difficulty.lag_days.min, spec.difficulty.lag_days.max
                        if reg.cause_day is not None:
                            lower = max(prev.start + 14, reg.start - 14, reg.cause_day + lag_min)
                        observed_days = [d for d in world.days[lower - 1:min(viewpoint, reg.end)] if d.preferences[dom].mentioned]
                        alternatives = []
                        for candidate in range(lower, first_difference + 1):
                            if reg.cause_day is not None and not lag_min <= candidate - reg.cause_day <= lag_max:
                                continue
                            if reg.end - candidate + 1 < 14:
                                continue
                            variants = [("same_rule", reg.rule)]
                            if reg.rule.type == "weekly_alternate":
                                variants += [("rotated_weekly_value_order", reg.rule.model_copy(update={"values": reg.rule.values[1:] + reg.rule.values[:1]}))]
                            for variant, candidate_rule in variants:
                                valid = True
                                for observed in observed_days:
                                    ctx = rules.context_for(observed.date, (observed.active_facts.get("city") or [None])[0])
                                    predicted = (rules.evaluate(prev.rule, observed.day, observed.date, prev.anchor, ctx) if observed.day < candidate else
                                                 rules.evaluate(candidate_rule, observed.day, observed.date, candidate, ctx))
                                    if predicted != observed.preferences[dom].base_value:
                                        valid = False
                                        break
                                if valid:
                                    alternatives.append({"start_day": candidate, "start_date": date_of(candidate).isoformat(), "rule_variant": variant,
                                                         "cause_lag_days": candidate - reg.cause_day if reg.cause_day is not None else None,
                                                         "accepted_by_date_exact": date_of(candidate).isoformat() in q["accept"]})
                        rejected = [a for a in alternatives if not a["accepted_by_date_exact"]]
                        assert rejected, q["id"]
                        row["counterfactual_change_starts"] = {"observed_choice_count": len(observed_days), "alternatives": alternatives,
                                                              "exceptions": "Actual exception values unchanged; their stated usual/base values must also match.",
                                                              "scope": "All declared choices through the actual probe; later regimes fixed; generator lag range respected. First-mention transcripts manually inspected; no exhaustive transcript entailment re-judging."}
                        row["checks"].append("bounded_counterfactual_start_with_anchor_and_cause_lag")
                        finding("non_unique_change_date_rejected_by_grader", "observation_limit",
                                {"conversation_compatible_rejected_starts": rejected, "observed_choice_count": len(observed_days)},
                                "Alternative boundaries preserve every declared pre-probe choice, including exception base values, but lie outside date_exact acceptance. For weekly reading, rotating the value order while anchoring at the later start preserves the identical observed cycle. This demonstrates an observability/grading limit, not incorrect hidden-world gold; no uniquely inferable boundary is proved.",
                                reg_locs(reg) + reg_locs(prev) + [{"file": "generator_v2/DECISIONS.md", "line": 46}, {"file": "experiments/core/grading.py", "line": 50}])
                elif reg.visibility == "uncaused":
                    expected_answer, expected_accept, expected_type, expected_grade = "No reason was stated.", ["no reason was stated"], "abstain", "llm_judge"
                    row["checks"].append("uncaused_semantics_no_reason_stated")
                else:
                    expected_answer = reg.cause_text
                    expected_accept, expected_type, expected_grade = [expected_answer], "free_text", "llm_judge"
                    row["checks"].append("cause_ref_text_and_first_observation_by_probe")
                    assert first <= viewpoint and stated(reg.cause_ref) <= viewpoint, q["id"]
                    if reg.visibility == "implicit":
                        row["interpretation_limits"].append("Cause is defined by synthetic cause_ref; the conversation intentionally omits a causal link. This is intended inference, not a direct quotation or proof of causation from text alone.")

            elif typ == "exception_vs_shift":
                target = day_of(dt.date.fromisoformat(ISO.search(q["question"]).group()))
                day = world.days[target - 1]
                pref = day.preferences[dom]
                reg = reg_at(dom, target)
                noun = nouns[dom]
                if pref.is_exception:
                    expected_answer = f"No - it was a one-off ({pref.exception_reason}); the usual routine continued."
                    expected_accept = ["no"]
                    old = pref.base_value
                else:
                    assert reg.kind == "shift", q["id"]
                    prev = world.regimes_for(dom)[world.regimes_for(dom).index(reg) - 1]
                    old = evaluate(prev, target)
                    expected_answer = f"Yes - from {date_of(reg.start).isoformat()} the routine changed to {reg.label}."
                    expected_accept = ["yes"]
                assert q["question"] == f"On {fmt(day.date)}, {spec.name}'s {noun} was {pref.value} instead of the usual {old}. Was that a lasting change?", q["id"]
                expected_type, expected_grade = "yes_no", "llm_judge"
                row.update(target_day=target, regime_id=reg.id, exception=pref.is_exception)
                row["authoritative_sources"] = [world_src.loc(f"/days/{target - 1}/preferences/{dom}")] + reg_locs(reg)
                row["checks"] += ["question_values_against_exception_or_previous_rule", "one_off_versus_shift_truth", "lasting_means_regime_shift_not_forever_unchanging"]

            elif typ == "reversion":
                matches = []
                for reg in world.regimes_for(dom):
                    if reg.kind != "temporary":
                        continue
                    back = next(r for r in world.regimes_for(dom) if r.start == reg.end + 1)
                    noun = nouns[dom]
                    if q["question"] == f"Is {spec.name}'s {noun} still {rule_items(reg)}?":
                        active = reg_at(dom, viewpoint)
                        assert active.id != reg.id, q["id"]
                        matches.append((reg, back, f"No - that lasted from {date_of(reg.start).isoformat()} to {date_of(reg.end).isoformat()}; then {spec.name} went back to {back.label}.", ["no"], "yes_no"))
                    elif q["question"] == f"What did {spec.name}'s {noun} go back to after the period of {rule_items(reg)}?":
                        matches.append((reg, back, f"{back.label} - {describe(back)}", [back.label], "free_text"))
                assert len(matches) == 1, (q["id"], matches)
                reg, back, expected_answer, expected_accept, expected_type = matches[0]
                expected_grade = "llm_judge"
                assert back.first_mention_day <= viewpoint, q["id"]
                row.update(temporary_regime_id=reg.id, return_regime_id=back.id, active_regime_at_probe=reg_at(dom, viewpoint).id)
                row["authoritative_sources"] = reg_locs(reg) + reg_locs(back)
                row["checks"] += ["temporary_inclusive_interval_and_return", "resumed_original_anchor", "still_question_reinterpreted_at_probe"]

            elif typ == "duration":
                matches = []
                for reg in world.regimes_for(dom):
                    if reg.kind == "reversion":
                        continue
                    question = (f"How long did {spec.name}'s {nouns[dom]} stay {rule_items(reg)}?" if reg.end < world.num_days else
                                f"As of {date_of(world.num_days).isoformat()}, how long had {spec.name}'s {nouns[dom]} been {rule_items(reg)}?")
                    if question == q["question"]:
                        matches.append((reg.start, reg.end, reg))
                if dom in facts:
                    f = facts[dom]
                    for value, start, end in intervals(world, dom, world.num_days):
                        if start == 1 or end == world.num_days:
                            continue
                        question = (f"How long was {value} one of {spec.name}'s {f.label}?" if f.cardinality == "multi" else f"How long was {spec.name}'s {f.label} {value}?")
                        if question == q["question"]:
                            matches.append((start, end, None))
                assert len(matches) == 1, (q["id"], matches)
                start, end, reg = matches[0]
                n = end - start + 1
                expected_answer = (f"{n} days ({date_of(start).isoformat()} to {date_of(end).isoformat()})" if end < world.num_days else f"{n} days (since {date_of(start).isoformat()})")
                expected_accept, expected_type, expected_grade = [str(n)], "free_text", "llm_judge"
                row.update(interval_start_day=start, interval_end_day=end, inclusive_days=n, elapsed_days_at_probe=min(end, viewpoint) - start + 1)
                row["checks"] += ["inclusive_calendar_arithmetic", "interval_derived_from_regime_or_daily_fact_runs", "completed_versus_ongoing_at_probe", "gold_and_accept_future_date_scan"]
                if reg:
                    row["authoritative_sources"] = reg_locs(reg)
                    nxt = next((r for r in world.regimes_for(dom) if r.start == end + 1), None)
                    closed = nxt is not None and nxt.first_mention_day <= viewpoint
                    row.update(regime_id=reg.id, closing_regime_id=nxt.id if nxt else None,
                               closing_first_mention_day=nxt.first_mention_day if nxt else None, closure_available_by_probe=closed)
                    if nxt and nxt.first_mention_day not in q["evidence_days"]:
                        diagnostic = {"id": q["id"], "type": typ, "category": "duration_closure_not_cited", "source": source,
                                      "cited_days": q["evidence_days"], "closing_first_mention_day": nxt.first_mention_day,
                                      "closing_available_by_probe": closed, "authority": reg_locs(nxt),
                                      "interpretation": "The metadata cites the old routine's first/last observations, not cessation. This is citation incompleteness, not by itself a wrong gold or unavailable full-prefix answer."}
                        diagnostics.append(diagnostic)
                        row["evidence_diagnostics"].append(diagnostic)
                    if end > viewpoint:
                        finding("future_completed_duration_in_gold", "temporal_eligibility_error", {"status": "routine has not ended by the probe", "elapsed_days": viewpoint - start + 1, "final_duration": "not established as a completed historical total at the probe", "interval_end_date": date_of(end).isoformat()},
                                "The gold asserts a completed total and ending date after the probe, while the question contains no future date. Correct synthetic eventual duration is not an available historical answer at this probe.",
                                reg_locs(reg) + reg_locs(nxt) + [{"file": "generator_v2/qa.py", "line": 441}, {"file": "generator_v2/probing.py", "line": 89}])
                    elif not closed:
                        finding("duration_completion_not_observed", "uncertainty", {"synthetic_interval_days": n, "elapsed_if_assumed_ongoing_at_probe": viewpoint - start + 1, "closing_first_mention_date": date_of(nxt.first_mention_day).isoformat() if nxt else None},
                                "The next regime has not been observed by the probe. The gold interval is correct synthetic truth, but its completed framing is not established by the cited old-routine observations. For two rows the endpoint is the probe itself, making the number also the correct elapsed duration; the third ends one day earlier with its transition first mentioned in the next month.",
                                reg_locs(reg) + (reg_locs(nxt) if nxt else []))
                else:
                    row["authoritative_sources"] = stat_locs([s.id for s in world.statements if s.entity == dom and s.kind in ("background_fact", "fact_change") and s.effective_day in (start, end + 1)])

            elif typ == "event_status":
                matches = []
                for event in spec.events:
                    if q["question"] == f"What happened with {event.title}?":
                        states = [s for s in event.states if s.day <= viewpoint]
                        assert states
                        state = states[-1]
                        matches.append((event, state, f"{state.text} ({state.status}, {date_of(state.day).isoformat()})", viewpoint))
                    for a, b in zip(event.states, event.states[1:]):
                        mid = (a.day + b.day) // 2
                        if q["question"] == f"What was the status of {event.title} on {date_of(mid).isoformat()}?":
                            state = next(s for s in reversed(event.states) if s.day <= mid)
                            matches.append((event, state, f"{state.text} ({state.status})", mid))
                assert len(matches) == 1, (q["id"], matches)
                event, state, expected_answer, target = matches[0]
                assert statements[state.id].text == f"[{event.title}] {state.text} ({state.status})" and statements[state.id].effective_day == state.day, q["id"]
                expected_accept, expected_type, expected_grade = [state.text], "free_text", "llm_judge"
                row.update(event_chain_id=event.chain_id, state_id=state.id, state_status=state.status, target_day=target,
                           effective_day=state.day, stated_day=statements[state.id].stated_day)
                row["authoritative_sources"] = stat_locs([s.id for s in event.states])
                row["checks"] += ["latest_effective_event_state_at_semantic_target", "planned_happened_cancelled_status_distinguished", "effective_versus_stated_event_dates"]
                if state.status == "planned":
                    row["interpretation_limits"].append("A future event discussed as a plan is legitimate knowledge. Only an explicit historical status query at an unreached date is ineligible; the planned status text itself is not an invention.")

            elif typ == "distractor_probe":
                matches = []
                for dis in spec.distractors:
                    question = f"Around {date_of(dis.day).isoformat()}, {dis.text}. Did that lead to any change in {spec.name}'s {nouns[dis.domain]}?"
                    if question != q["question"]:
                        continue
                    answer = f"No - nothing about {spec.name}'s {nouns[dis.domain]} changed because of it."
                    sources = stat_locs([dis.id])
                    if dis.confounder:
                        near = [r for r in world.regimes_for(dis.domain) if abs(r.start - dis.day) <= 10 and r.kind != "initial"]
                        if near:
                            r = near[0]
                            answer = f"No - {spec.name}'s {nouns[dis.domain]} did change around then (to {r.label} from {date_of(r.start).isoformat()}), but no link to this was ever stated."
                            sources += reg_locs(r)
                            if r.first_mention_day not in q["evidence_days"]:
                                finding("confounder_change_not_cited", "evidence_gap", {"change_observation_day": r.first_mention_day},
                                        "Gold's explanation correctly describes a nearby change, but the evidence list contains only the distractor statement. The actual change is available elsewhere in the prefix.", sources)
                    matches.append((answer, ["no"], sources))
                for reg in world.regimes_for(dom):
                    if reg.kind not in ("shift", "temporary") or not reg.cause_text:
                        continue
                    question = f"Around {date_of(reg.cause_day).isoformat()}, {reg.cause_text}. Did that lead to any change in {spec.name}'s {nouns[dom]}?"
                    if question == q["question"]:
                        matches.append((f"Yes - from {date_of(reg.start).isoformat()} {spec.name}'s {nouns[dom]} changed to {reg.label}.", ["yes"], reg_locs(reg)))
                        assert reg.first_mention_day <= viewpoint and stated(reg.cause_ref) <= viewpoint, q["id"]
                        if reg.visibility == "implicit":
                            row["interpretation_limits"].append("Positive causal control is inferred from synthetic cause_ref and temporally nearby observations; an explicit conversational causal link is intentionally absent.")
                assert len(matches) == 1, (q["id"], matches)
                expected_answer, expected_accept, sources = matches[0]
                expected_type, expected_grade = "yes_no", "llm_judge"
                row["authoritative_sources"] = sources
                row["checks"] += ["distractor_versus_causal_control", "confounder_change_distinguished_from_attribution", "cause_and_change_available_at_probe"]

            elif typ == "other_person":
                matches = []
                for person in spec.other_people:
                    for i, fact in enumerate(person.facts):
                        sid = f"other.{person.person.lower()}.{i}"
                        question = fact.question or f"What is {spec.name}'s {person.relation} {person.person}'s {fact.noun}?"
                        if question == q["question"]:
                            matches.append((fact.value, [fact.value], "value", stat_locs([sid])))
                        if fact.confusable_with:
                            f = facts[fact.confusable_with]
                            vals = world.days[viewpoint - 1].active_facts[f.entity]
                            ask = lambda value: (f"Is {value} one of {spec.name}'s {f.label}?" if f.cardinality == "multi" else f"Is {spec.name}'s {f.label} {value}?")
                            if q["question"] == ask(fact.value):
                                assert fact.value not in vals, q["id"]
                                matches.append((f"No - that is {person.person} ({spec.name}'s {person.relation}), not {spec.name}.", ["no"], "yes_no", stat_locs([sid])))
                            for v in world.days[-1].active_facts[f.entity][:1]:
                                if q["question"] == ask(v):
                                    yes = v in vals
                                    matches.append(("Yes." if yes else "No.", ["yes" if yes else "no"], "yes_no", [world_src.loc(f"/days/{viewpoint - 1}/active_facts/{f.entity}")]))
                # Same personal control can be generated by multiple confusable facts.
                unique = {(a, tuple(ac), at): (a, ac, at, src) for a, ac, at, src in matches}
                assert len(unique) == 1, (q["id"], matches)
                expected_answer, expected_accept, expected_type, sources = next(iter(unique.values()))
                expected_grade = "llm_judge"
                row["authoritative_sources"] = sources
                row["checks"] += ["person_identity_and_fact_provenance", "personal_yes_no_control_at_changed_viewpoint"]

            elif typ == "abstention":
                assert "control" in q["tags"], q["id"]
                entity = next(tag for tag in q["tags"] if tag in facts)
                values = world.days[viewpoint - 1].active_facts[entity]
                expected_answer, expected_accept, expected_type, expected_grade = joined(values), [joined(values)], "value", "llm_judge"
                row.update(control=True, semantic_entity=entity, expected_values=values)
                row["authoritative_sources"] = [world_src.loc(f"/days/{viewpoint - 1}/active_facts/{entity}")]
                row["checks"] += ["answerable_control_not_unknown", "control_facts_at_changed_viewpoint"]

            else:
                raise AssertionError((q["id"], "Unhandled non-recall family", typ))

            assert expected_answer is not None and expected_accept is not None, q["id"]
            row.update(expected_answer=expected_answer, expected_accept=expected_accept,
                       expected_answer_type=expected_type, expected_grading=expected_grade,
                       gold_matches_semantic_truth=q["answer"] == expected_answer,
                       acceptance_matches_semantic_truth=q["accept"] == expected_accept,
                       answer_metadata_valid=q["answer_type"] == expected_type and q["grading"] == expected_grade)
            if q["answer"] != expected_answer:
                finding("gold_answer_mismatch_at_probe", "reference_error", {"answer": expected_answer, "accept": expected_accept},
                        "Gold was generated for the final world but this relative/current question is asked at an earlier probe. The per-day active facts define the correct list at that viewpoint.",
                        row["authoritative_sources"] + [{"file": "generator_v2/qa.py", "line": 370}, {"file": "generator_v2/probing.py", "line": 145}])
            if q["accept"] != expected_accept:
                finding("accepted_answer_mismatch_at_probe", "reference_error", {"accept": expected_accept},
                        "The acceptance list encodes the same stale/mismatched reference at this changed viewpoint.", row["authoritative_sources"], observed=q["accept"])
            if not row["answer_metadata_valid"]:
                finding("wrong_answer_type_or_grading", "reference_error", {"answer_type": expected_type, "grading": expected_grade},
                        "Metadata does not match the semantic family/cardinality.", row["authoritative_sources"])
            row["reference_verdict"] = "mismatch" if not row["gold_matches_semantic_truth"] else "matches_synthetic_truth"
            row["eligibility_verdict"] = "future_target_or_completed_endpoint" if any(c in row["issue_categories"] for c in ("explicit_future_target", "future_completed_duration_in_gold")) else "no_confirmed_future_semantics"
            row["observational_status"] = ("requires_closed_world_convention" if "empty_fact_has_no_negative_statement" in row["observation_limit_categories"] else
                                           "specific_evidence_gap" if any(i["id"] == q["id"] and i["kind"] == "evidence_gap" for i in issues) else "no_specific_gap_proven")
            row["interpretation_limits"].append("Transcript entailment for every paraphrase was not exhaustively re-judged; no saved validation failure is automatically treated as an evidence failure.")
            for diagnostic in row["evidence_diagnostics"]:
                if diagnostic["category"] != "duration_closure_not_cited":
                    diagnostics.append({"id": q["id"], "user_id": uid, "type": typ, "source": source, **diagnostic})
            all_coverage.append(row)

        probing_abstention = [q for _, _, q in rows if q["type"] == "abstention"]
        unknown_pool = [(i, q) for i, q in enumerate(pool_src.data) if q["type"] == "abstention" and q["answer_type"] == "abstain"]
        selection.append({
            "user_id": uid, "category": "unknown_abstention_excluded_by_empty_evidence", "kind": "selection_gap",
            "unknown_pool_count": len(unknown_pool), "unknown_probing_count": sum(q["answer_type"] == "abstain" for q in probing_abstention),
            "answerable_probing_controls": len(probing_abstention),
            "omitted_pool_items": [{"id": q["id"], "source": pool_src.loc(f"/{i}"), "evidence_days": q["evidence_days"], "question": q["question"]} for i, q in unknown_pool],
            "rationale": "The probing allocator immediately skips every pool item with no evidence. All genuinely never-mentioned abstention questions have empty evidence.",
            "authority": [{"file": "generator_v2/probing.py", "line": 83}, {"file": "generator_v2/qa.py", "line": 527}],
        })

    for path in input_paths:
        fingerprints[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert total_rows == 1000 and len(all_coverage) == 511
    assert len({r["id"] for r in all_coverage}) == 511
    distinct_by_kind = {kind: sorted({i["id"] for i in issues if i["kind"] == kind}) for kind in sorted({i["kind"] for i in issues})}
    summary = {
        "total_probing_rows_enumerated": total_rows, "non_recall_rows_checked": len(all_coverage), "recall_rows_delegated": total_rows - len(all_coverage),
        "by_user": dict(user_totals), "by_type": dict(type_totals),
        "gold_matches_synthetic_truth": sum(r["gold_matches_semantic_truth"] for r in all_coverage),
        "gold_mismatch_rows": sum(not r["gold_matches_semantic_truth"] for r in all_coverage),
        "metadata_mismatch_rows": sum(not r["answer_metadata_valid"] for r in all_coverage),
        "acceptance_mismatch_rows": sum(not r["acceptance_matches_semantic_truth"] for r in all_coverage),
        "issue_record_count": len(issues), "issue_counts_by_category": dict(collections.Counter(i["category"] for i in issues)),
        "issue_distinct_question_count": len({i["id"] for i in issues}),
        "distinct_questions_with_issue_or_observation_limit": len({i["id"] for i in issues + observation_limits}),
        "distinct_ids_by_kind": distinct_by_kind,
        "uncertainty_counts_by_category": dict(collections.Counter(i["category"] for i in uncertainties)),
        "observation_limit_counts_by_category": dict(collections.Counter(i["category"] for i in observation_limits)),
        "diagnostic_counts_by_category": dict(collections.Counter(i["category"] for i in diagnostics)),
        "unknown_abstention_pool_items_excluded": sum(r["unknown_pool_count"] for r in selection),
        "answerable_abstention_controls_in_probes": sum(r["answerable_probing_controls"] for r in selection),
        "future_dates_in_gold_or_accept": [{"id": r["id"], "source": r["source"], "future_text": {f: r["future_text"][f] for f in ("answer", "accept")}} for r in all_coverage if r["future_text"]["answer"] or r["future_text"]["accept"]],
        "patterns_with_uncited_rule_values": sum(any(d.get("category") == "pattern_value_coverage" and d["uncited_rule_values"] for d in r["evidence_diagnostics"]) for r in all_coverage),
        "missing_probe_families": sorted(set(CAPABILITIES) - {r["type"] for r in all_coverage} - {"recall"}),
        "limitations": ["Truth and temporal eligibility are separate verdicts.", "Evidence gaps do not by themselves prove a wrong reference or an unanswerable full prefix.",
                        "Implicit causation and rule abstraction are legitimate intended inference; a formal identifiability proof was not attempted.",
                        "Pool provenance equality is checked separately from synthetic truth at the probe.", "No network, paid model, or saved-session revalidation is used."],
    }
    for filename, obj in (("coverage.json", all_coverage), ("issues.json", issues), ("uncertainties.json", uncertainties),
                          ("observation_limits.json", observation_limits), ("evidence_diagnostics.json", diagnostics), ("selection_gaps.json", selection), ("summary.json", summary), ("input_sha256.json", fingerprints)):
        (OUT / filename).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    write_report(summary, issues, uncertainties, observation_limits, diagnostics, selection)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("future_dates_in_gold_or_accept", "distinct_ids_by_kind", "limitations")}, indent=2))


def source_label(location):
    return f"`{location['file']}:{location['line']}`"


def write_report(summary, issues, uncertainties, observation_limits, diagnostics, selection):
    lines = [
        "# Non-recall gold and implicit-time audit", "",
        "This is an offline audit of all 511 non-recall records in the 1,000 canonical monthly probing questions. All records were mapped to a semantic family and checked against the daily synthetic world at the appropriate target and probe viewpoint. Pool equality is a provenance check, not the truth oracle. No datasets, generator, agents, or evaluation implementation were edited.", "",
        "## Results", "",
        f"- {summary['gold_matches_synthetic_truth']}/511 gold references match the synthetic truth at their semantic target; {summary['gold_mismatch_rows']} current reference is wrong at its actual probe.",
        "- One acceptance list repeats that wrong current reference. All 511 answer-type/grading combinations match their semantic family/cardinality.",
        "- 47 non-recall questions have explicit future targets: 42 patterns and 5 event-status queries. These overlap the root scheduling audit and must not be added as disjoint findings.",
        "- Two additional questions have an unreached completed endpoint only in the gold answer. Their eventual synthetic durations are correct, but they are not completed historical totals at the primary probe.",
        "- Three duration rows have an unobserved closing transition: two end on the probe itself and one ends a day earlier but is first observed in the next month. They are recorded as closure/wording uncertainties.",
        "- Four empty-pet answers require an unstated closed-world convention; they are observation/answerability limits. Specific citation gaps affect 5 change-boundary rows and 2 confounder explanations. None is counted as wrong synthetic gold.",
        "- Those same 5 change-boundary rows also admit declared-choice-consistent alternative start dates rejected by date_exact. This is an overlapping observability/grading limit, not five additional hidden-world errors.",
        f"- {summary['diagnostic_counts_by_category'].get('duration_closure_not_cited', 0)} preference-duration records cite first/last old-routine observations but omit the closing transition. This is a diagnostic of citation incompleteness, not a claim that the full conversation prefix is missing the transition.",
        "- All 30 genuinely unknown abstention pool questions are excluded because they have empty evidence; the probes retain 13 answerable controls and zero unknown abstention cases. This is a selection gap, not erroneous gold on those 13 controls.", "",
        "| Family | Probing rows | Family-specific truth check |", "| --- | ---: | --- |",
    ]
    family_checks = {
        "fact_at_time": "Daily active fact list on the explicit target; cardinality and empty sets",
        "pattern_at_time": "Entire named month/point inside the target regime; full rule and correct anchor",
        "attribution": "Cause reference, uncaused semantics, first observation and statement date",
        "exception_vs_shift": "Exception value/reason versus previous rule; genuine regime shifts",
        "change_detection": "Effective start, first mention, accepted-date bounds and distinguishability",
        "duration": "Inclusive calendar arithmetic, world-derived intervals and observed closure",
        "event_status": "Effective state at target/viewpoint, separate planned/happened/cancelled status",
        "distractor_probe": "Distractor versus real cause; near confounder change and availability",
        "fact_current": "Complete daily fact set at the actual probe, including retained values",
        "abstention": "Answerable controls rechecked at actual probe; exclusion of unknown cases",
        "reversion": "Inclusive temporary period, actual return and resumed base anchor; current 'still'",
        "fact_history": "Maximal value intervals recomputed using only the prefix; 'now' at actual probe",
        "other_person": "Person/fact identity and all personal yes/no controls at actual probe",
    }
    for family, count in summary["by_type"].items():
        lines.append(f"| {family} | {count} | {family_checks[family]} |")
    lines += ["", "No `pattern_current` or `prediction` items occur in these probes. Historical pattern copies remain anchored to their explicit target; none of the selected current fact/history/other-person/abstention/reversion copies produced a new retrospective-only stale reference.", "",
              "## Confirmed reference mismatch", ""]
    for issue in issues:
        if issue["category"] != "gold_answer_mismatch_at_probe":
            continue
        lines += [f"### {issue['id']}", "", f"Source: {source_label(issue['source']['question'])}; gold: {source_label(issue['source']['answer'])}; accept: {source_label(issue['source']['accept'])}.",
                  f"Question: {issue['question']}", f"Probe: {issue['probe_date']}. Observed gold: `{issue['observed']['answer']}`. Expected: `{issue['expected']['answer']}`.",
                  "Piano is still an active and known hobby on the January probe. It is removed only on day 420 (2027-04-24). The final-world current-hobby generator keeps bouldering/trail running, and the probing allocator moves that unchanged final answer to January using only the remaining hobbies' evidence [1, 331]. The full January transcript also retains piano, including the January 13 user report of playing and January 22 intention to practice.",
                  "Authoritative daily truth: " + ", ".join(source_label(s) for s in issue["authoritative_sources"]), ""]
    lines += ["## Future completed endpoints in the gold", "", "These are additional to the 51 explicitly future-dated questions handled by the root audit.", "",
              "| ID | Question source | Probe | Gold completed endpoint/total | Prefix truth |", "| --- | --- | --- | --- | --- |"]
    for issue in issues:
        if issue["category"] == "future_completed_duration_in_gold":
            lines.append(f"| {issue['id']} | {source_label(issue['source']['question'])} | {issue['probe_date']} | {issue['observed']['answer']} | Still ongoing; {issue['expected']['elapsed_days']} days elapsed; final total not yet established as completed |")
    lines += ["", "## Specific evidence gaps", "", "Gold truth and conversation observations are deliberately separated here. These findings show what cited metadata does not establish; they do not invalidate a reference that agrees with the synthetic world.", "",
              "| ID | Category | Question source | Observation |", "| --- | --- | --- | --- |"]
    for issue in issues + observation_limits:
        if issue["kind"] == "evidence_gap" or issue["category"] == "empty_fact_has_no_negative_statement":
            lines.append(f"| {issue['id']} | {issue['category']} | {source_label(issue['source']['question'])} | {issue['rationale']} |")
    lines += ["", "The three primary empty-pet queries have no user pet vocabulary anywhere in their available prefixes (u2 through August 2026, u3 through April 2026). The u2 retrospective empty-pet query has subsequently learned the January 2027 adoption, but neither its day-1 citation nor the recorded fact statements establish that this was the person's first pet. Empty-list gold therefore depends on treating unmentioned synthetic facts as absent.", "",
              "For the five non-explicit change rows above, the first mention is compatible with the old rule. The first differing values occur later: u4 music on day 606, u5 reading on day 197, and u5 music on day 351. Explicitly announced shifts with unchanged daily values are not flagged: their declaration can supply the missing boundary evidence.", "",
              "## Date grading observability limits", "", "These overlap the five change-citation gaps above. Every alternative respects the declared pre-probe choice values, exception bases, and the persona's cause-lag range. Subsequent regimes remain fixed. Actual first-mention transcripts contain daily choices without a dated switch announcement.", "",
              "| ID | Question source | Accepted dates | Rejected compatible boundaries |", "| --- | --- | --- | --- |"]
    for issue in observation_limits:
        if issue["category"] != "non_unique_change_date_rejected_by_grader":
            continue
        alternatives = "; ".join(f"{a['start_date']} ({a['rule_variant']})" for a in issue["expected"]["conversation_compatible_rejected_starts"])
        lines.append(f"| {issue['id']} | {source_label(issue['source']['question'])} | {', '.join(issue['observed']['accept'])} | {alternatives} |")
    lines += ["", "For u4 music, moving the same day-of-week rule to days 603/604 keeps cause lags 3/4 within the U4 1–5 range. Day 606 would also fit the observations but exceeds that lag range, so it is excluded from the recorded admissible alternatives. For u5 music, days 346–350 remain within U5's 1–10 range; day 351 would fit the choices but has an excessive lag of 11. For u5 reading, fixed ordered weekly values constrain the original anchor; reversing the two values and beginning on the first poetry day (197) yields the same continuing cycle without exposing any earlier behavioral difference. The question supplies neither cycle order nor anchor. These are bounded counterexamples, not a census of every compatible alternative world.", "",
              "## Timing-specific citation diagnostics", "", "Eight current/at-time fact rows omit an earlier removal relevant to establishing a complete list. Twenty-two at-time fact rows cite a later update to the same value; three of these hobby rows cite the much later piano removal (day 420) for targets in October/December 2026. Those updates are known by the actual probes, so this is not future leakage; it is temporally unsuitable direct support for the earlier membership and can delay question selection. The exact IDs, statements, and source lines are in evidence_diagnostics.json.", "",
              "## Closure uncertainty", "", "| ID | Question source | Probe | Synthetic interval days | First next-regime mention |", "| --- | --- | --- | ---: | --- |"]
    for issue in uncertainties:
        lines.append(f"| {issue['id']} | {source_label(issue['source']['question'])} | {issue['probe_date']} | {issue['expected']['synthetic_interval_days']} | {issue['expected']['closing_first_mention_date']} |")
    lines += ["", "## Counter-evidence and limits", "",
              "- All 11 fact-history answers were recomputed from the prefix and remain valid; no future completed fact change or stale 'now' interval was found.",
              "- All 31 event references match their intended state and effective date. Plans for later months are legitimate recorded knowledge. Five historical status queries nevertheless target dates beyond their probe; knowing a current plan does not establish status at a later date.",
              "- All selected exception, reversion, other-person, and answerable abstention controls match their world truth at the probe. Intended causal inference is not flagged merely because a cause link was implicit.",
              "- Free-text `accept` hints can be incomplete labels (especially history); they are not treated as strict alternative answers. The evaluator's judge uses the complete reference, while `date_exact` uses its accepted dates.",
              f"- {summary['patterns_with_uncited_rule_values']} of 59 pattern citations omit one or more values in their full rule. This is an evidence diagnostic, not an error count: additional prefix observations and intended inference can supply them.",
              "- Rule descriptions, per-day truth, statement dates, and metadata were checked reproducibly for every non-recall row. Exhaustive entailment of every natural-language paraphrase and formal uniqueness of every inferred routine/cause remain outside this script's proof. Saved validator flags were not assumed correct.", "",
              "## Reproduction and artifacts", "", "Run `PYTHONDONTWRITEBYTECODE=1 python .research/v2-probing-audit/gold/audit_gold.py`.", "",
              "- `coverage.json`: 511 per-question records with target/viewpoint truth, exact source lines, checks, caveats, and verdicts.",
              "- `issues.json`: per-ID reference, temporal eligibility, and specific evidence findings; categories can overlap.",
              "- `uncertainties.json`: completed-versus-elapsed wording cases.",
              "- `observation_limits.json`: four empty-set answerability limits and five bounded counterfactual date-grading limits, the latter overlapping the corresponding citation findings.",
              "- `evidence_diagnostics.json`: missing duration closing citations, fact removal/timing diagnostics, and pattern value coverage.",
              "- `selection_gaps.json`: omitted unknown abstention pool IDs and exact locations.",
              "- `summary.json`: counts and ID unions by finding kind; future-date scan across question, gold, and acceptance.",
              "- `input_sha256.json`: hashes of audited canonical inputs and authoritative generator/evaluator sources.",
              "- `sources.md` and `manual_evidence.md`: source-role inventory and the narrow transcript checks kept separate from interpretation.", ""]
    (OUT / "findings.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
