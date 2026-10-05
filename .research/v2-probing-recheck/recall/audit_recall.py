#!/usr/bin/env python3
"""Offline recheck of all current monthly recall instances; writes here only."""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import runpy

import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / '.research/v2-probing-audit/recall'
helpers = runpy.run_path(str(OLD / 'audit_recall.py'))
for name in ('JSONSource', 'rule_eval', 'phase'):
    obj = helpers[name]
    if name == 'JSONSource':
        obj.ref.__globals__['ROOT'] = ROOT
    else:
        obj.__globals__['ROOT'] = ROOT
JSONSource, rule_eval, phase = (helpers[n] for n in ('JSONSource', 'rule_eval', 'phase'))


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def jsonl(name, data):
    (OUT / name).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in data))


def main():
    oldrows = [json.loads(x) for x in (OLD / 'coverage.jsonl').read_text().splitlines()]
    old_by_target = {(r['user_id'], r['domain'], r['target_date']): r for r in oldrows}
    old_reviews = {r['original_id']: r for r in json.loads((OLD / 'manual_paraphrase_reviews.json').read_text())}
    manual_path = OUT / 'new_manual_turns.json'
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else {}
    rows, citations, reviews, phases, boundary_candidates = [], [], [], [], []
    sources = set()
    current_pool, current_monthly = {}, {}

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
        regs = {r['id']: (i, r) for i, r in enumerate(world.data['regimes'])}
        originals = {q['id']: (i, q) for i, q in enumerate(pool.data)}
        current_pool[user] = pool
        current_monthly[user] = probes
        specs = {p['domain']: p for p in persona['preferences']}
        start = dt.date.fromisoformat(world.data['start_date'])
        sessions = {}

        def session(day):
            date = days[day][1]['date']
            if date not in sessions:
                path = base / 'sessions' / f'{date}.json'
                sessions[date] = JSONSource(path) if path.exists() else None
                if path.exists():
                    sources.add(path)
            return sessions[date]

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
                ri, reg = regs[p['regime_id']]
                oi, sourceq = originals[q['original_id']]
                base_value = rule_eval(reg['rule'], target, date, reg['anchor'])
                target_phase = phase(reg['rule'], target, date, reg['anchor'])
                same = [d for d in reg['mention_days'] if phase(
                    reg['rule'], d, dt.date.fromisoformat(days[d][1]['date']), reg['anchor']) == target_phase]
                nearest = same or reg['mention_days']
                expected = [target] if p['mentioned'] else [d for d in sorted(
                    sorted(nearest, key=lambda d: abs(d - target))[:3]) if d <= target]
                near = any(abs(target-r['start']) <= 7 for _, r in regs.values()
                           if r['domain'] == dom and r['start'] > 1)
                tags = ([] if p['mentioned'] else ['inferred']) + (['exception'] if p['is_exception'] else [])
                tags += (['no_session'] if not day['has_session'] else []) + (['boundary'] if near else [])
                copied = ('type', 'capability', 'domain', 'question', 'answer',
                          'answer_type', 'accept', 'evidence_days', 'grading', 'tags')
                checks = {
                    'weekday_matches_date': date.strftime('%A') in q['question'],
                    'independent_rule_equals_world_base': base_value == p['base_value'],
                    'nonexception_value_equals_rule': p['is_exception'] or p['value'] == base_value,
                    'exception_differs_from_base': not p['is_exception'] or p['value'] != base_value,
                    'exception_in_persona': not p['is_exception'] or any(
                        x['value'] == p['value'] and x['reason'] == p['exception_reason']
                        for x in specs[dom].get('exception_pool', [])) or any(
                        x['day'] == target and x['value'] == p['value']
                        for x in specs[dom].get('exceptions', [])),
                    'gold_equals_world_truth': q['answer'] == p['value'],
                    'accepted_values_equal_truth': q['accept'] == [p['value']],
                    'copied_fields_match_original': all(q[k] == sourceq[k] for k in copied),
                    'tags_match_world': q['tags'] == tags,
                    'evidence_matches_updated_generator': q['evidence_days'] == expected,
                    'nonempty_evidence': bool(q['evidence_days']),
                    'viewpoint_matches_probe': viewpoint == probe['viewpoint_day'],
                    'viewpoint_matches_date': viewpoint == (dt.date.fromisoformat(q['probe_date'])-start).days+1,
                    'target_not_after_viewpoint': target <= viewpoint,
                    'evidence_not_after_target': all(e <= target for e in q['evidence_days']),
                    'evidence_not_after_viewpoint': all(e <= viewpoint for e in q['evidence_days']),
                }
                cited = []
                for e in q['evidence_days']:
                    ei, ed = days[e]; ep = ed['preferences'][dom]; es = session(e)
                    agg = conversations.data['sessions'].get(ed['date'])
                    literal = [i for i,t in enumerate(es.data['turns'])
                               if t['speaker']=='user' and ep['value'].casefold() in t['text'].casefold()] if es else []
                    ephase = phase(reg['rule'], e, dt.date.fromisoformat(ed['date']), reg['anchor'])
                    ec = {
                        'mentioned': ep['mentioned'], 'has_session': ed['has_session'],
                        'individual_session_exists': es is not None,
                        'aggregate_session_exists': agg is not None,
                        'aggregate_turns_match': bool(es and agg and agg['turns']==es.data['turns']),
                        'same_regime': ep['regime_id']==reg['id'],
                        'nonexception_unless_target': e==target or not ep['is_exception'],
                        'saved_validator_matches_world': bool(es and es.data.get('extracted',{}).get('preferences',{}).get(dom)==ep['value']),
                        'independent_rule_matches_base': rule_eval(reg['rule'],e,dt.date.fromisoformat(ed['date']),reg['anchor'])==ep['base_value'],
                        'same_phase': ephase==target_phase,
                        'value_equals_gold': ep['value']==q['answer'],
                        'within_probe': e<=viewpoint,
                    }
                    er={'question_id':q['id'],'original_id':q['original_id'],'user_id':user,
                        'domain':dom,'day':e,'date':ed['date'],'truth':ep['value'],'phase':list(ephase),
                        'checks':ec,'literal_user_turns':literal,
                        'value_source':world.ref(('days',ei,'preferences',dom,'value')),
                        'session_source':es.ref(excerpt=False) if es else None}
                    citations.append(er);cited.append(er)
                direct = None
                if p['mentioned']:
                    es = session(target)
                    if cited[0]['literal_user_turns']:
                        direct='literal_user_value_plus_saved_extraction'
                    else:
                        old=old_by_target.get((user,dom,dates[0]));review=old_reviews.get(old['original_id']) if old else None
                        reuse=bool(review and old['question']==q['question'] and old['reference_answer']==q['answer'])
                        turn_indexes = review['turn_indexes'] if reuse else manual.get(q['original_id'])
                        direct='manual_reviewed_paraphrase' if turn_indexes is not None else 'pending_manual_review'
                        reviews.append({'question_id':q['id'],'original_id':q['original_id'],'user_id':user,
                            'question':q['question'],'domain':dom,'target_date':dates[0],'gold':q['answer'],
                            'status':direct,'reused_verified_target_and_payload':reuse,
                            'baseline_original_id':old['original_id'] if reuse else None,
                            'turn_indexes':turn_indexes,
                            'excerpts':[es.ref(('turns',i,'text')) for i in turn_indexes] if turn_indexes is not None else [],
                            'all_user_turns':[{'turn_index':i,'text_source':es.ref(('turns',i,'text'))}
                                for i,t in enumerate(es.data['turns']) if t['speaker']=='user']})
                row={'question_id':q['id'],'original_id':q['original_id'],'user_id':user,
                    'probe_date':q['probe_date'],'viewpoint_day':viewpoint,
                    'target_date':dates[0],'target_day':target,'domain':dom,
                    'question':q['question'],'reference_answer':q['answer'],'accept':q['accept'],
                    'target_truth':p['value'],'independent_base':base_value,
                    'target_phase':list(target_phase),'regime_id':reg['id'],
                    'regime_start_day':reg['start'],'regime_end_day':reg['end'],
                    'first_observation_day':reg['first_mention_day'],
                    'target_mentioned':p['mentioned'],'target_has_session':day['has_session'],
                    'exception':p['is_exception'],'tags':q['tags'],'is_retrospective':q['is_retrospective'],
                    'evidence_days':q['evidence_days'],'expected_evidence_days':expected,
                    'same_phase_observed_days':same,
                    'same_phase_observed_by_probe':[d for d in same if d<=viewpoint],
                    'evidence_method':'direct_target' if p['mentioned'] else ('same_phase' if same else 'fallback_other_phase'),
                    'direct_transcript_support':direct,'checks':checks,
                    'record_source':probes.ref(qp,excerpt=False),
                    'question_source':probes.ref(qp+('question',)),
                    'answer_source':probes.ref(qp+('answer',)),
                    'evidence_source':probes.ref(qp+('evidence_days',)),
                    'target_source':world.ref(('days',ti,'preferences',dom)),
                    'rule_source':world.ref(('regimes',ri,'rule'))}
                rows.append(row)
                if not p['mentioned'] and not row['same_phase_observed_by_probe']:
                    cue=re.compile(r'bik|cycl|pedal|two.wheel|saturday|weekend|alternat|rotation|every other|two.week',re.I)
                    phases.append({'question_id':q['id'],'original_id':q['original_id'],'user_id':user,
                        'question':q['question'],'probe_date':q['probe_date'],'answer':q['answer'],
                        'cited_values':[c['truth'] for c in cited], 'same_phase_observed_by_probe':row['same_phase_observed_by_probe'],
                        'all_phase_days':[{'day':d,'date':days[d][1]['date'],'has_session':days[d][1]['has_session'],
                            'mentioned':days[d][1]['preferences'][dom]['mentioned']}
                            for d in range(reg['start'],reg['end']+1)
                            if phase(reg['rule'],d,dt.date.fromisoformat(days[d][1]['date']),reg['anchor'])==target_phase],
                        'all_domain_observations_by_probe':[{'day':d,'truth':days[d][1]['preferences'][dom]['value'],
                            'user_turns':[session(d).ref(('turns',i,'text')) for i,t in enumerate(session(d).data['turns']) if t['speaker']=='user']}
                            for d in reg['mention_days'] if d<=viewpoint],
                        'all_pre_probe_cue_excerpts':[{'day':d,'speaker':t['speaker'],'source':session(d).ref(('turns',i,'text'))}
                            for d in range(1,viewpoint+1) if days[d][1]['has_session']
                            for i,t in enumerate(session(d).data['turns']) if cue.search(t['text'])],
                        'record_source':row['record_source'],'question_source':row['question_source'],
                        'evidence_source':row['evidence_source'],'rule_source':row['rule_source']})
                if not p['mentioned']:
                    domain_regs=sorted((r for _,r in regs.values() if r['domain']==dom),key=lambda r:r['start'])
                    for prev,new in zip(domain_regs,domain_regs[1:]):
                        if not prev['mention_days'] or not new['mention_days']:
                            continue
                        last=max(prev['mention_days']);first=min(new['mention_days'])
                        if not last<target<first or first>viewpoint:
                            continue
                        observations=[d for d in range(prev['start'],min(new['end'],viewpoint)+1)
                                      if days[d][1]['preferences'][dom]['mentioned']]
                        alternatives=[]
                        for candidate in range(last+1,first+1):
                            mismatch=[]
                            for d in observations:
                                r=prev if d<candidate else new
                                anchor=prev['anchor'] if d<candidate else (candidate if new['anchor']==new['start'] else new['anchor'])
                                predicted=rule_eval(r['rule'],d,dt.date.fromisoformat(days[d][1]['date']),anchor)
                                if predicted!=days[d][1]['preferences'][dom]['base_value']:
                                    mismatch.append(d)
                            r=prev if target<candidate else new
                            anchor=prev['anchor'] if target<candidate else (candidate if new['anchor']==new['start'] else new['anchor'])
                            val=rule_eval(r['rule'],target,date,anchor)
                            alternatives.append({'candidate_start_day':candidate,'target_answer':val,'mismatch_days':mismatch})
                        boundary_candidates.append({'question_id':q['id'],'user_id':user,'domain':dom,
                            'target_day':target,'answer':q['answer'],'old_regime_id':prev['id'],'new_regime_id':new['id'],
                            'last_old_observation':last,'hidden_new_start':new['start'],'first_new_observation':first,
                            'observed_days_checked':observations,'counterfactual_starts':alternatives,
                            'different_compatible_answer':any(not a['mismatch_days'] and a['target_answer']!=q['answer'] for a in alternatives),
                            'question_source':row['question_source'],
                            'boundary_user_turns':[{'day':d,'turn_index':i,'source':session(d).ref(('turns',i,'text'))}
                                for d in range(last,first+1) if days[d][1]['has_session']
                                for i,t in enumerate(session(d).data['turns']) if t['speaker']=='user']})

    baseline=[]
    for old in json.loads((OLD/'issues.json').read_text()):
        user=old['user_id'];pp=current_pool[user];monthly=current_monthly[user]
        matches=[(i,q) for i,q in enumerate(pp.data) if q['question']==old['question'] and q['domain']==old.get('domain',next(
            r['domain'] for r in oldrows if r['question_id']==old['question_id']))]
        instances=[q for p in monthly.data['probes'] for q in p['questions'] if q['question']==old['question']]
        baseline.append({'baseline_question_id':old['question_id'],'baseline_original_id':old['original_id'],
            'category':old['category'],'question':old['question'],'target_date':old['target_date'],
            'current_pool_records':[{'id':q['id'],'answer':q['answer'],'accept':q['accept'],'evidence_days':q['evidence_days'],
                'source':pp.ref((i,),excerpt=False)} for i,q in matches],
            'current_monthly_records':[{'id':q['id'],'original_id':q['original_id'],'probe_date':q['probe_date'],
                'answer':q['answer'],'accept':q['accept'],'evidence_days':q['evidence_days']} for q in instances]})
    failures=collections.Counter(k for r in rows for k,v in r['checks'].items() if not v)
    evidence_failures=collections.Counter(k for r in citations for k,v in r['checks'].items() if not v)
    summary={'instances':len(rows),'unique_source_questions':len({r['original_id'] for r in rows}),
        'by_user':{u:{'instances':sum(r['user_id']==u for r in rows),
            'inferred':sum(r['user_id']==u and not r['target_mentioned'] for r in rows),
            'direct':sum(r['user_id']==u and r['target_mentioned'] for r in rows),
            'no_session':sum(r['user_id']==u and not r['target_has_session'] for r in rows),
            'exception':sum(r['user_id']==u and r['exception'] for r in rows),
            'retrospective':sum(r['user_id']==u and r['is_retrospective'] for r in rows)} for u in current_pool},
        'direct_instances':sum(r['target_mentioned'] for r in rows),
        'inferred_instances':sum(not r['target_mentioned'] for r in rows),
        'evidence_references':len(citations),'check_failure_counts':dict(failures),
        'evidence_check_failure_counts':dict(evidence_failures),
        'direct_nonliteral_unique':len({r['original_id'] for r in reviews}),
        'pending_manual_review_unique':len({r['original_id'] for r in reviews if r['status']=='pending_manual_review'}),
        'missing_phase_instances':len(phases),'boundary_ambiguity_candidates':len(boundary_candidates),
        'different_compatible_answer_instances':len({r['question_id'] for r in boundary_candidates if r['different_compatible_answer']})}
    jsonl('coverage.jsonl',rows);jsonl('evidence.jsonl',citations)
    write('manual_paraphrase_reviews.json',reviews);write('phase_reviews.json',phases)
    write('boundary_candidates.json',boundary_candidates);write('baseline_matches.json',baseline);write('summary.json',summary)
    sources.update(ROOT/p for p in ['generator_v2/qa.py','generator_v2/probing.py'])
    write('sources.json',[{'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(sources)])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
