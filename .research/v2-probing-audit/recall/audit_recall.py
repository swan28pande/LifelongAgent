#!/usr/bin/env python3
"""Offline, read-only audit of every monthly recall instance in canonical v2.

Run from any directory: python .research/v2-probing-audit/recall/audit_recall.py
Only audit artifacts beside this script are written. No generator imports, network,
model calls, dataset edits, or temporary files are used. PyYAML is already installed
in this workspace and is used only to check exception-pool membership.
"""

from __future__ import annotations

import bisect
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

import yaml


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
WEEKDAYS = ('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun')
SEASONS = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring',
           5: 'spring', 6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall',
           10: 'fall', 11: 'fall'}

# Independent manual transcript reviews by the audit agent of ALL direct target citations in which no
# user turn contains the reference value literally. Indexed by source QA ID so a
# retrospective copy reuses the same reviewed transcript, not a second judgment.
# These are supporting paraphrases, not new closed-option model extractions.
REVIEW_TURNS = {
    'u3_q0004': [1], 'u3_q0122': [5], 'u3_q0118': [5],
    'u3_q0072': [7], 'u3_q0078': [5], 'u3_q0061': [3],
    'u3_q0075': [5], 'u3_q0029': [1], 'u3_q0015': [1],
    'u3_q0089': [5], 'u3_q0056': [3], 'u3_q0000': [1],
    'u3_q0003': [1], 'u3_q0134': [7], 'u3_q0044': [3],
    'u3_q0103': [7], 'u3_q0141': [7, 9], 'u3_q0110': [3],
    'u3_q0069': [5], 'u3_q0101': [1], 'u3_q0126': [7, 8, 9],
    'u3_q0058': [3], 'u3_q0010': [1], 'u3_q0093': [1],
    'u3_q0148': [3, 5], 'u3_q0083': [3], 'u3_q0131': [3],
    'u3_q0030': [1], 'u3_q0145': [5], 'u3_q0039': [5],
    'u3_q0129': [5], 'u3_q0095': [3], 'u3_q0024': [1],
    'u3_q0037': [3], 'u3_q0059': [5, 7], 'u3_q0142': [5, 7],
    'u3_q0132': [7], 'u3_q0023': [3], 'u3_q0128': [5],
    'u3_q0090': [1],
    'u4_q0047': [1], 'u4_q0038': [1], 'u4_q0009': [3],
    'u4_q0093': [7], 'u4_q0095': [7], 'u4_q0037': [1],
    'u4_q0006': [1], 'u4_q0105': [7, 9], 'u4_q0059': [3],
    'u4_q0100': [9], 'u4_q0007': [1], 'u4_q0049': [1],
    'u4_q0104': [5], 'u4_q0019': [1], 'u4_q0032': [1, 2],
    'u4_q0010': [1], 'u4_q0008': [1], 'u4_q0056': [5],
    'u4_q0109': [9], 'u4_q0000': [1], 'u4_q0112': [9],
    'u4_q0088': [3],
    'u5_q0062': [7], 'u5_q0069': [7], 'u5_q0020': [1],
    'u5_q0024': [3], 'u5_q0047': [3], 'u5_q0073': [11],
    'u5_q0036': [5], 'u5_q0042': [5], 'u5_q0002': [1],
    'u5_q0027': [1], 'u5_q0031': [3],
}


class JSONSource:
    """Parse JSON and index exact one-based source lines by JSON path."""

    def __init__(self, path):
        self.path = Path(path)
        self.text = self.path.read_text()
        self.data = json.loads(self.text)
        self.lines = self.text.splitlines()
        self.newlines = [i for i, ch in enumerate(self.text) if ch == '\n']
        self.offsets = {}
        self.decoder = json.JSONDecoder()
        self._parse(0, ())

    def _skip(self, p):
        while p < len(self.text) and self.text[p].isspace():
            p += 1
        return p

    def _parse(self, p, path):
        p = self._skip(p)
        start = p
        ch = self.text[p]
        if ch == '{':
            p = self._skip(p + 1)
            while self.text[p] != '}':
                key, p = self.decoder.raw_decode(self.text, p)
                p = self._skip(p)
                assert self.text[p] == ':'
                p = self._parse(p + 1, path + (key,))
                p = self._skip(p)
                if self.text[p] == ',':
                    p = self._skip(p + 1)
                else:
                    assert self.text[p] == '}'
            p += 1
        elif ch == '[':
            p = self._skip(p + 1)
            n = 0
            while self.text[p] != ']':
                p = self._parse(p, path + (n,))
                n += 1
                p = self._skip(p)
                if self.text[p] == ',':
                    p = self._skip(p + 1)
                else:
                    assert self.text[p] == ']'
            p += 1
        else:
            _, p = self.decoder.raw_decode(self.text, p)
        self.offsets[path] = (start, p)
        return p

    def ref(self, path=(), excerpt=True):
        a, b = self.offsets[path]
        lo = bisect.bisect_left(self.newlines, a) + 1
        hi = bisect.bisect_left(self.newlines, b - 1) + 1
        result = {'file': str(self.path.relative_to(ROOT)), 'line_start': lo,
                  'line_end': hi, 'json_pointer': '/' + '/'.join(
                      str(x).replace('~', '~0').replace('/', '~1') for x in path)}
        if excerpt:
            result['excerpt'] = '\n'.join(self.lines[lo - 1:hi])
        return result


def code_ref(filename, phrase, n=1):
    path = ROOT / filename
    lines = path.read_text().splitlines()
    lo = next(i for i, line in enumerate(lines) if phrase in line)
    return {'file': filename, 'line_start': lo + 1, 'line_end': lo + n,
            'excerpt': '\n'.join(lines[lo:lo + n])}


def context(date):
    # Current persona rules use only workday and season, not weather. An explicit
    # error prevents this audit from silently accepting a future weather rule.
    return {'is_workday': 'true' if date.weekday() < 5 else 'false',
            'season': SEASONS[date.month]}


def rule_eval(rule, day, date, anchor):
    """Independent raw-JSON interpreter; never calls generator_v2/rules.py."""
    if isinstance(rule, str):
        return rule
    off = day - anchor
    assert off >= 0
    t = rule['type']
    if t == 'constant':
        return rule['value']
    if t == 'weekly_alternate':
        return rule['values'][(off // 7) % len(rule['values'])]
    if t == 'n_day_cycle':
        pos = off % sum(s['days'] for s in rule['segments'])
        for seg in rule['segments']:
            if pos < seg['days']:
                return seg['value']
            pos -= seg['days']
    if t == 'day_of_week':
        return rule['map'][WEEKDAYS[date.weekday()]]
    if t == 'nested':
        block = rule['blocks'][(off // rule.get('block_days', 7)) % len(rule['blocks'])]
        return rule_eval(block, day, date, anchor)
    if t == 'conditional':
        key = context(date)[rule['condition']]
        case = rule['cases'].get(key, rule['cases'].get('default'))
        assert case is not None
        return rule_eval(case, day, date, anchor)
    raise ValueError(rule)


def phase(rule, day, date, anchor):
    if isinstance(rule, str) or rule['type'] == 'constant':
        return ()
    off = day - anchor
    t = rule['type']
    if t == 'weekly_alternate':
        return ((off // 7) % len(rule['values']),)
    if t == 'n_day_cycle':
        pos = off % sum(s['days'] for s in rule['segments'])
        for i, seg in enumerate(rule['segments']):
            if pos < seg['days']:
                return (i,)
            pos -= seg['days']
    if t == 'day_of_week':
        return (WEEKDAYS[date.weekday()],)
    if t == 'nested':
        i = (off // rule.get('block_days', 7)) % len(rule['blocks'])
        return (i,) + phase(rule['blocks'][i], day, date, anchor)
    if t == 'conditional':
        key = context(date)[rule['condition']]
        case = rule['cases'].get(key, rule['cases'].get('default'))
        assert case is not None
        return (key,) + phase(case, day, date, anchor)
    raise ValueError(rule)


def write_jsonl(filename, rows):
    with (OUT / filename).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    coverage, evidence, issues, reviews, boundary_reviews, phase_reviews = [], [], [], [], [], []
    sources = set()
    by_user = {}
    manual_seen = set()
    for user in (f'u{i}' for i in range(1, 6)):
        base = ROOT / 'datasets/v2' / user
        world = JSONSource(base / 'world_state.json')
        probes = JSONSource(base / 'probing_questions.json')
        pool = JSONSource(base / 'qa_pool.json')
        conversations = JSONSource(base / 'conversations.json')
        persona_path = ROOT / 'generator_v2/personas' / f'{user}.yaml'
        persona = yaml.safe_load(persona_path.read_text())
        sources.update([world.path, probes.path, pool.path, conversations.path, persona_path])
        days = {d['day']: (i, d) for i, d in enumerate(world.data['days'])}
        regimes = {r['id']: (i, r) for i, r in enumerate(world.data['regimes'])}
        original = {q['id']: (i, q) for i, q in enumerate(pool.data)}
        prefspec = {p['domain']: p for p in persona['preferences']}
        start = dt.date.fromisoformat(world.data['start_date'])
        sessions = {}
        counters = collections.Counter()

        def session(day):
            date = days[day][1]['date']
            if date not in sessions:
                path = base / 'sessions' / f'{date}.json'
                sessions[date] = JSONSource(path) if path.exists() else None
                if path.exists():
                    sources.add(path)
            return sessions[date]

        def pref_ref(day, dom, key):
            return world.ref(('days', days[day][0], 'preferences', dom, key))

        for pi, probe in enumerate(probes.data['probes']):
            for qi, q in enumerate(probe['questions']):
                if q['type'] != 'recall':
                    continue
                qp = ('probes', pi, 'questions', qi)
                dates = re.findall(r'\d{4}-\d{2}-\d{2}', q['question'])
                assert len(dates) == 1, q['id']
                date = dt.date.fromisoformat(dates[0])
                target = (date - start).days + 1
                ti, day = days[target]
                dom, viewpoint = q['domain'], q['viewpoint_day']
                p = day['preferences'][dom]
                ri, reg = regimes[p['regime_id']]
                oi, sourceq = original[q['original_id']]
                target_phase = phase(reg['rule'], target, date, reg['anchor'])
                expected_base = rule_eval(reg['rule'], target, date, reg['anchor'])
                exception_options = prefspec[dom].get('exception_pool', [])
                explicit = prefspec[dom].get('exceptions', [])
                same_phase = [d for d in reg['mention_days'] if
                              phase(reg['rule'], d, dt.date.fromisoformat(days[d][1]['date']),
                                    reg['anchor']) == target_phase]
                nearest_pool = same_phase or reg['mention_days']
                expected_evidence = [target] if p['mentioned'] else sorted(
                    sorted(nearest_pool, key=lambda d: abs(d - target))[:3])
                boundary = any(abs(target - r['start']) <= 7 for _, r in regimes.values()
                               if r['domain'] == dom and r['start'] > 1)
                expected_tags = []
                if not p['mentioned']:
                    expected_tags.append('inferred')
                if p['is_exception']:
                    expected_tags.append('exception')
                if not day['has_session']:
                    expected_tags.append('no_session')
                if boundary:
                    expected_tags.append('boundary')
                copied = ('type', 'capability', 'domain', 'question', 'answer',
                          'answer_type', 'accept', 'evidence_days', 'grading', 'tags')
                checks = {
                    'one_iso_target_date': len(dates) == 1,
                    'target_date_matches_world_day': day['date'] == dates[0],
                    'weekday_matches_target_date': date.strftime('%A') in q['question'],
                    'domain_exists_in_world': dom in day['preferences'],
                    'regime_contains_target': reg['start'] <= target <= reg['end'],
                    'independent_rule_equals_world_base': expected_base == p['base_value'],
                    'nonexception_value_equals_rule': p['is_exception'] or p['value'] == expected_base,
                    'exception_differs_from_base': not p['is_exception'] or p['value'] != expected_base,
                    'exception_in_persona_pool_or_explicit': not p['is_exception'] or
                        any(x['value'] == p['value'] and x['reason'] == p['exception_reason']
                            for x in exception_options) or
                        any(x['day'] == target and x['value'] == p['value'] for x in explicit),
                    'gold_answer_equals_target_truth': q['answer'] == p['value'],
                    'accepted_values_equal_target_truth': q['accept'] == [p['value']],
                    'source_original_id_exists': q['original_id'] in original,
                    'copied_fields_equal_original': all(q[k] == sourceq[k] for k in copied),
                    'inferred_no_session_exception_boundary_tags_correct': q['tags'] == expected_tags,
                    'evidence_equals_generator_selection': q['evidence_days'] == expected_evidence,
                    'viewpoint_matches_probe': viewpoint == probe['viewpoint_day'],
                    'viewpoint_matches_probe_date': viewpoint == (
                        dt.date.fromisoformat(q['probe_date']) - start).days + 1,
                    'target_not_after_viewpoint': target <= viewpoint,
                    'evidence_not_after_viewpoint': all(e <= viewpoint for e in q['evidence_days']),
                }
                citations = []
                for e in q['evidence_days']:
                    ei, eday = days[e]
                    ep = eday['preferences'][dom]
                    es = session(e)
                    cs = conversations.data['sessions'].get(eday['date'])
                    extracted = es.data.get('extracted', {}).get('preferences', {}).get(dom) if es else None
                    user_turns = [(i, t['text']) for i, t in enumerate(es.data['turns'])
                                  if t['speaker'] == 'user'] if es else []
                    literal = [i for i, text in user_turns if ep['value'].casefold() in text.casefold()]
                    ephase = phase(reg['rule'], e, dt.date.fromisoformat(eday['date']), reg['anchor'])
                    ec = {
                        'world_day_exists': e in days,
                        'world_has_session': eday['has_session'],
                        'topic_marked_mentioned': ep['mentioned'],
                        'individual_session_exists': es is not None,
                        'aggregate_session_exists': cs is not None,
                        'aggregate_and_individual_turns_equal': bool(es and cs and cs['turns'] == es.data['turns']),
                        'same_regime_as_target': ep['regime_id'] == reg['id'],
                        'nonexception_unless_target_exception': e == target or not ep['is_exception'],
                        'saved_extracted_value_equals_day_truth': extracted == ep['value'],
                        'independent_rule_equals_evidence_base': rule_eval(
                            reg['rule'], e, dt.date.fromisoformat(eday['date']), reg['anchor']) == ep['base_value'],
                        'same_phase_as_target': ephase == target_phase,
                        'evidence_value_equals_gold': ep['value'] == q['answer'],
                        'not_after_viewpoint': e <= viewpoint,
                    }
                    er = {'question_id': q['id'], 'original_id': q['original_id'], 'user_id': user,
                          'domain': dom, 'day': e, 'date': eday['date'], 'truth': ep['value'],
                          'base': ep['base_value'], 'phase': list(ephase), 'checks': ec,
                          'saved_validator_value': extracted, 'literal_user_turns': literal,
                          'session_failures': es.data['failures'] if es else None,
                          'world_value_source': pref_ref(e, dom, 'value'),
                          'session_source': es.ref(excerpt=False) if es else None,
                          'saved_extraction_source': es.ref(('extracted', 'preferences', dom))
                              if es and dom in es.data.get('extracted', {}).get('preferences', {}) else None,
                          'literal_user_excerpts': [es.ref(('turns', i, 'text')) for i in literal],
                          'aggregate_turns_source': conversations.ref(
                              ('sessions', eday['date'], 'turns'), excerpt=False) if cs else None}
                    citations.append(er)
                    evidence.append(er)
                direct_support = None
                if p['mentioned']:
                    es = session(target)
                    literal = citations[0]['literal_user_turns']
                    direct_support = 'literal_user_value_plus_saved_extraction' if literal else 'agent_reviewed_paraphrase'
                    if not literal:
                        turns = REVIEW_TURNS[q['original_id']]
                        if q['original_id'] not in manual_seen:
                            manual_seen.add(q['original_id'])
                            reviews.append({'original_id': q['original_id'], 'user_id': user,
                                            'domain': dom, 'target_date': dates[0], 'gold': q['answer'],
                                            'status': 'supports_intended_value', 'turn_indexes': turns,
                                            'excerpts': [es.ref(('turns', i, 'text')) for i in turns],
                                            'limitation': 'Manual transcript review by audit agent; not a new independent model validation. '
                                               + ('"Pounded the pavement on foot" is an idiomatic running description; '
                                                  'it is less explicit than saying "ran".' if q['original_id'] == 'u4_q0032' else '')})
                result = {'question_id': q['id'], 'original_id': q['original_id'], 'user_id': user,
                          'probe_date': q['probe_date'], 'viewpoint_day': viewpoint,
                          'target_date': dates[0], 'target_day': target, 'domain': dom,
                          'question': q['question'], 'reference_answer': q['answer'],
                          'accept': q['accept'], 'target_truth': p['value'], 'independent_base': expected_base,
                          'target_phase': list(target_phase), 'regime_id': reg['id'],
                          'regime_start_day': reg['start'], 'first_observation_day': reg['first_mention_day'],
                          'target_mentioned': p['mentioned'], 'target_has_session': day['has_session'],
                          'exception': p['is_exception'], 'tags': q['tags'], 'expected_tags': expected_tags,
                          'is_retrospective': q['is_retrospective'], 'evidence_days': q['evidence_days'],
                          'expected_evidence_days': expected_evidence, 'same_phase_observed_days': same_phase,
                          'evidence_method': 'direct_target' if p['mentioned'] else
                             ('same_phase' if same_phase else 'fallback_other_phase'),
                          'direct_transcript_support': direct_support, 'checks': checks,
                          'evidence_checks': [{'day': c['day'], 'checks': c['checks']} for c in citations],
                          'question_source': probes.ref(qp + ('question',)),
                          'record_source': probes.ref(qp, excerpt=False),
                          'answer_source': probes.ref(qp + ('answer',)),
                          'accept_source': probes.ref(qp + ('accept',)),
                          'viewpoint_source': probes.ref(qp + ('viewpoint_day',)),
                          'evidence_source': probes.ref(qp + ('evidence_days',)),
                          'original_source': pool.ref((oi,), excerpt=False),
                          'target_value_source': pref_ref(target, dom, 'value'),
                          'target_mentioned_source': pref_ref(target, dom, 'mentioned'),
                          'target_session_source': world.ref(('days', ti, 'has_session')),
                          'regime_rule_source': world.ref(('regimes', ri, 'rule')),
                          'regime_start_source': world.ref(('regimes', ri, 'start')),
                          'first_observation_source': world.ref(('regimes', ri, 'first_mention_day'))}
                coverage.append(result)
                counters['instances'] += 1
                counters['retrospective_instances'] += q['is_retrospective']
                counters['inferred_instances'] += not p['mentioned']
                counters['no_session_instances'] += not day['has_session']
                counters['exception_instances'] += p['is_exception']
                counters['evidence_references'] += len(citations)
                counters['direct_nonliteral_instances_reviewed'] += direct_support == 'agent_reviewed_paraphrase'
                common = {'question_id': q['id'], 'original_id': q['original_id'], 'user_id': user,
                          'question': q['question'], 'reference_answer': q['answer'],
                          'accepted_values': q['accept'], 'synthetic_gold_correct': checks['gold_answer_equals_target_truth'],
                          'target_date': dates[0], 'target_day': target, 'probe_date': q['probe_date'],
                          'viewpoint_day': viewpoint,
                          'sources': [result['question_source'], result['answer_source'], result['accept_source'],
                                      result['viewpoint_source'], result['target_value_source']]}
                if target > viewpoint:
                    issues.append({**common, 'category': 'future_target_in_recall', 'confidence': 'confirmed',
                                   'observed': f'Recall asks about day {target}, after probe day {viewpoint}.',
                                   'expected': 'A past-tense recall target must not exceed its probe viewpoint.',
                                   'days_after_probe': target - viewpoint,
                                   'source_interpretation': 'The reference is correct for the later synthetic day; '
                                      'available pattern observations may make it predictable, but this is a future recall target.',
                                   'generator_source': code_ref('generator_v2/probing.py', 'if q["type"] == "fact_at_time":', 9)})
                if not p['mentioned'] and not same_phase:
                    observed_values = sorted({days[d][1]['preferences'][dom]['value'] for d in reg['mention_days']
                                              if d <= viewpoint})
                    issues.append({**common, 'category': 'unobserved_answer_phase', 'confidence': 'confirmed',
                                   'observed': {'cited_values': [c['truth'] for c in citations],
                                                'same_phase_observations_in_entire_regime': same_phase,
                                                'regime_observed_values_by_probe': observed_values},
                                   'expected': 'An inferred exact answer needs observations or an explicit stated rule '
                                               'that support its target phase; unrelated phases do not identify an unseen choice.',
                                   'source_interpretation': 'No bike-ride workout was reported in the entire initial regime. '
                                      'Pre-probe cycling language is attributed to commuting or other people, not this workout phase.',
                                   'sources': common['sources'] + [result['evidence_source'], result['regime_rule_source']]
                                              + [c['world_value_source'] for c in citations],
                                   'transcript_sources': [session(d).ref(('turns', i, 'text')) for d, i in
                                       [(68, 9), (77, 5), (79, 5), (68, 1), (79, 1), (85, 5), (91, 1)]],
                                   'generator_source': code_ref('generator_v2/qa.py', 'pool = same or reg.mention_days', 2),
                                   'limitations': 'Absence of an observed exact phrase alone was not used. '
                                      'The full pre-probe transcript was reviewed for bicycle/pedaling/rotation language and attribution.'})
                    # Keep all observed exercise-day text, and every pre-probe
                    # bicycle/cycle/phase cue including incidental mentions.
                    cue = re.compile(r'bik|cycl|pedal|two.wheel|saturday|weekend|alternat|rotation|every other|two.week', re.I)
                    phase_reviews.append({'question_id': q['id'], 'original_id': q['original_id'],
                        'all_monthly_copies': [{'question_id': candidate['id'], 'probe_date': candidate['probe_date']}
                            for pp in probes.data['probes'] for candidate in pp['questions']
                            if candidate['original_id'] == q['original_id']],
                        'rule': reg['rule'], 'regime_start': reg['start'], 'regime_end': reg['end'],
                        'all_same_phase_days': [{'day': d, 'date': days[d][1]['date'],
                            'has_session': days[d][1]['has_session'],
                            'preference_mentioned': days[d][1]['preferences'][dom]['mentioned']}
                            for d in range(reg['start'], reg['end'] + 1)
                            if phase(reg['rule'], d, dt.date.fromisoformat(days[d][1]['date']), reg['anchor']) == target_phase],
                        'observed_domain_days_by_probe': [{'day': d, 'truth': days[d][1]['preferences'][dom]['value'],
                            'user_excerpts': [session(d).ref(('turns', i, 'text')) for i, t in
                                             enumerate(session(d).data['turns']) if t['speaker'] == 'user']}
                            for d in reg['mention_days'] if d <= viewpoint],
                        'all_pre_probe_cue_excerpts': [{'day': d, 'date': days[d][1]['date'],
                            'speaker': t['speaker'], 'text_source': session(d).ref(('turns', i, 'text'))}
                            for d in range(1, viewpoint + 1) if days[d][1]['has_session']
                            for i, t in enumerate(session(d).data['turns']) if cue.search(t['text'])],
                        'agent_attribution_review': 'User cycling cues are work commutes; workout cycling cues are '
                            'about coworker Dana. No target-phase workout or stated exercise rotation specifies bike rides.'})
                if not p['mentioned'] and target < reg['first_mention_day']:
                    prevreg = max((r for _, r in regimes.values() if r['domain'] == dom and r['end'] < reg['start']),
                                  key=lambda r: r['end'])
                    alternative = rule_eval(prevreg['rule'], target, date, prevreg['anchor'])
                    lastold = max(prevreg['mention_days'])
                    first = reg['first_mention_day']
                    # Challenge backward inference using the ENTIRE available
                    # observed sequence. Constant and weekday rules can move
                    # their start to the first observation without shifting any
                    # later phase. An anchor-sensitive cyclic rule would have to
                    # pass this same check before receiving this finding.
                    counterfactuals = []
                    observable_days = [d for d in range(prevreg['start'], viewpoint + 1)
                                       if days[d][1]['preferences'][dom]['mentioned']
                                       and d <= reg['end']]
                    for candidate_start in range(reg['start'], first + 1):
                        mismatches = []
                        for d in observable_days:
                            actual = days[d][1]['preferences'][dom]
                            # One-off exceptions retain the recorded value/reason;
                            # compare their base too because the text may state it.
                            rr = prevreg if d < candidate_start else reg
                            anchor = prevreg['anchor'] if d < candidate_start else candidate_start
                            predicted = rule_eval(rr['rule'], d,
                                dt.date.fromisoformat(days[d][1]['date']), anchor)
                            if predicted != actual['base_value']:
                                mismatches.append(d)
                        target_rule = prevreg if target < candidate_start else reg
                        target_anchor = prevreg['anchor'] if target < candidate_start else candidate_start
                        counterfactuals.append({'candidate_start_day': candidate_start,
                            'candidate_start_date': days[candidate_start][1]['date'],
                            'observed_days_checked': observable_days,
                            'observed_base_mismatch_days': mismatches,
                            'target_answer': rule_eval(target_rule['rule'], target, date, target_anchor),
                            'compatible_with_all_available_observed_choices': not mismatches})
                    compatible_alternatives = [x for x in counterfactuals
                                              if not x['observed_base_mismatch_days']
                                              and x['target_answer'] != q['answer']]
                    assert compatible_alternatives, q['id']
                    checks['exact_new_regime_start_observed'] = False
                    issue = {**common, 'category': 'unobserved_shift_boundary', 'confidence': 'confirmed_answerability_limitation',
                             'observed': {'target': target, 'world_regime_start': reg['start'],
                                          'last_old_observation': lastold, 'first_new_observation': first,
                                          'generator_change_date_acceptance_days': list(range(reg['start'], first + 1)),
                                          'alternative_if_change_occurred_after_target': alternative,
                                          'rule_type': reg['rule']['type'],
                                          'counterfactual_start_checks': counterfactuals},
                             'expected': 'Either establish that the new regime began by the target day, '
                                         'or permit the old-regime value / uncertainty at the unresolved boundary.',
                             'source_interpretation': 'The gold is correct in hidden truth, but actual observations '
                                'do not date the shift precisely. Choosing a later change within the generator\'s '
                                'own accepted date interval changes this target\'s answer.',
                             'sources': common['sources'] + [result['target_mentioned_source'], result['regime_start_source'],
                                                            result['first_observation_source'],
                                                            pref_ref(lastold, dom, 'value'), pref_ref(first, dom, 'value')],
                             'generator_source': code_ref('generator_v2/qa.py', 'accept = [self.date(d).isoformat()', 3),
                             'transcript_sources': [session(lastold).ref(('turns', i, 'text')) for i in
                                                    ([5] if user == 'u3' else [7])]
                                                  + [session(first).ref(('turns', 5, 'text'))],
                             'limitations': 'The hidden effective change day is known to the audit, not the evaluated agent. '
                                'A plausible inferential guess may still match the gold. This is not a wrong synthetic answer.'}
                    issues.append(issue)
                    # Capture ALL actual user turns over the observational boundary;
                    # these preserve the negative evidence against exact date claims.
                    boundary_reviews.append({'question_id': q['id'], 'domain': dom,
                        'target': target, 'start': reg['start'], 'first_observation': first,
                        'counterfactual_start_checks': counterfactuals,
                        'by_day': [{'day': d, 'date': days[d][1]['date'],
                                    'user_excerpts': [session(d).ref(('turns', i, 'text'))
                                                      for i, t in enumerate(session(d).data['turns'])
                                                      if t['speaker'] == 'user']}
                                   for d in range(lastold, viewpoint + 1)
                                   if days[d][1]['has_session'] and
                                   (d <= first or days[d][1]['preferences'][dom]['mentioned'])]})
                if q['original_id'] == 'u5_q0006':
                    # Preserve a complete pre-probe transcript for this confirmed
                    # missing-phase finding, so absence and attribution can be reviewed.
                    for d in range(1, viewpoint + 1):
                        if days[d][1]['has_session']:
                            session(d)
        by_user[user] = dict(counters)

    assert len(coverage) == 489
    allowed_failed = {'target_not_after_viewpoint', 'exact_new_regime_start_observed'}
    unexpected = [(r['question_id'], k) for r in coverage for k, v in r['checks'].items()
                  if not v and k not in allowed_failed]
    assert not unexpected, unexpected
    allowed_evidence_failed = {'same_phase_as_target', 'evidence_value_equals_gold'}
    unexpected_evidence = [(r['question_id'], r['day'], k) for r in evidence
                           for k, v in r['checks'].items() if not v and k not in allowed_evidence_failed]
    assert not unexpected_evidence, unexpected_evidence
    summary = {
        'scope': 'All monthly recall question instances in canonical datasets/v2/u1..u5',
        'instances': len(coverage), 'unique_source_questions': len({r['original_id'] for r in coverage}),
        'by_user': by_user,
        'check_failure_counts': dict(collections.Counter(k for r in coverage for k, v in r['checks'].items() if not v)),
        'evidence_check_failure_counts': dict(collections.Counter(k for r in evidence for k, v in r['checks'].items() if not v)),
        'issue_category_counts': dict(collections.Counter(i['category'] for i in issues)),
        'distinct_flagged_instances': len({i['question_id'] for i in issues}),
        'evidence_references': len(evidence),
        'unique_evidence_domain_days': len({(r['user_id'], r['domain'], r['day']) for r in evidence}),
        'direct_nonliteral_unique_sources_agent_reviewed': len(reviews),
        'wrong_reference_answer_instances': sum(not r['checks']['gold_answer_equals_target_truth'] for r in coverage),
        'wrong_accepted_values_instances': sum(not r['checks']['accepted_values_equal_target_truth'] for r in coverage),
        'direct_instances': sum(r['target_mentioned'] for r in coverage),
        'inferred_instances': sum(not r['target_mentioned'] for r in coverage),
        'no_session_instances': sum(not r['target_has_session'] for r in coverage),
        'ordinary_same_phase_inference_instances_without_flagged_issue': sum(
            r['evidence_method'] == 'same_phase' and r['question_id'] not in
            {i['question_id'] for i in issues} for r in coverage),
        'limitations': [
            'Existing saved validator extractions are source evidence, not new independent semantic validation.',
            'All direct citations without literal gold wording received manual transcript review by audit agent, preserved as excerpts.',
            'No-session/inferred status alone is not treated as an error; ordinary same-phase inference is accepted.',
            'Inferred recall assumes stable routine patterns despite one-off exceptions; this audit isolates '
            'specific missing phases and shift ambiguities rather than demanding logical certainty for every inference.',
            'The four future recall targets also belong to the broader previously known future-date category.',
        ],
    }
    write_jsonl('coverage.jsonl', coverage)
    write_jsonl('evidence.jsonl', evidence)
    (OUT / 'issues.json').write_text(json.dumps(issues, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'manual_paraphrase_reviews.json').write_text(json.dumps(reviews, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'boundary_transcript_reviews.json').write_text(json.dumps(boundary_reviews, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'phase_transcript_review.json').write_text(json.dumps(phase_reviews, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    for filename in ['generator_v2/qa.py', 'generator_v2/probing.py', 'generator_v2/rules.py',
                     'generator_v2/simulator.py', 'generator_v2/conversation.py', 'generator_v2/DECISIONS.md']:
        sources.add(ROOT / filename)
    manifest = [{'file': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(sources)]
    (OUT / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
