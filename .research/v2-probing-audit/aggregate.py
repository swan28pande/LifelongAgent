#!/usr/bin/env python3
"""Consolidate audit evidence without editing any source dataset or code."""
import collections
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(name):
    path = OUT / name
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    return json.loads(path.read_text())


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def main():
    gold = {r['id']: r for r in read('gold/coverage.json')}
    recall = {r['question_id']: r for r in read('recall/coverage.jsonl')}
    source = {r['id']: r for r in read('source/probe_coverage.jsonl')}
    assert not gold.keys() & recall.keys()
    assert len(gold) == 511 and len(recall) == 489
    assert gold.keys() | recall.keys() == source.keys()
    questions = {}
    for uid in ('u1', 'u2', 'u3', 'u4', 'u5'):
        dataset = json.loads((ROOT / f'datasets/v2/{uid}/probing_questions.json').read_text())
        for probe in dataset['probes']:
            for q in probe['questions']:
                assert q['id'] not in questions
                questions[q['id']] = q
    assert questions.keys() == source.keys()

    findings = []
    for issue in read('calendar_issues.json'):
        findings.append({**issue, 'question_id': issue['id'], 'class': 'confirmed_error',
                         'evidence_artifact': 'calendar_issues.json'})
    for issue in read('gold/issues.json'):
        category = issue['category']
        if category == 'explicit_future_target':
            continue  # Already represented by the independent calendar audit.
        finding_class = ('confirmed_error' if issue['kind'] in
                         ('reference_error', 'temporal_eligibility_error') else
                         'observation_limit' if category == 'empty_fact_has_no_negative_statement' else
                         'citation_limit')
        findings.append({**issue, 'question_id': issue['id'], 'class': finding_class,
                         'evidence_artifact': 'gold/issues.json'})
    for issue in read('recall/issues.json'):
        if issue['category'] == 'future_target_in_recall':
            continue  # Four recall rows also occur in calendar_issues.json.
        findings.append({**issue, 'class': 'observation_limit',
                         'evidence_artifact': 'recall/issues.json'})
    for issue in read('gold/observation_limits.json'):
        findings.append({**issue, 'question_id': issue['id'], 'class': 'observation_limit',
                         'evidence_artifact': 'gold/observation_limits.json'})
    for issue in read('gold/uncertainties.json'):
        findings.append({**issue, 'question_id': issue['id'], 'class': 'uncertainty',
                         'evidence_artifact': 'gold/uncertainties.json'})

    by_question = collections.defaultdict(list)
    for finding in findings:
        assert finding['question_id'] in questions
        by_question[finding['question_id']].append(finding)
    diagnostics = collections.defaultdict(list)
    for record in gold.values():
        diagnostics[record['id']].extend(record['evidence_diagnostics'])
    flag_review = read('source/flag_reviews.json')['summary']
    exposure = set(flag_review['confirmed_source_contract_citation_exposure_probe_ids'])
    coverage = []
    for qid, q in questions.items():
        record = gold.get(qid, recall.get(qid))
        truth_correct = (record['gold_matches_semantic_truth'] if qid in gold else
                         record['checks']['gold_answer_equals_target_truth'])
        loc = record['source']['question'] if qid in gold else record['question_source']
        coverage.append({
            'id': qid, 'original_id': q['original_id'], 'type': q['type'],
            'question': q['question'], 'answer': q['answer'], 'accept': q['accept'],
            'probe_date': q['probe_date'], 'viewpoint_day': q['viewpoint_day'],
            'is_retrospective': q['is_retrospective'],
            'file': loc['file'], 'line': loc.get('line', loc.get('line_start')),
            'structural_checks_passed': True,
            'reference_matches_synthetic_truth_at_semantic_target': truth_correct,
            'finding_categories': sorted({r['category'] for r in by_question[qid]}),
            'finding_classes': sorted({r['class'] for r in by_question[qid]}),
            'evidence_diagnostics': diagnostics[qid],
            'cites_saved_flagged_session': bool(source[qid]['flagged_evidence_days']),
            'cites_session_with_confirmed_source_contract_issue': qid in exposure,
            'family_audit_artifact': ('gold/coverage.json' if qid in gold else 'recall/coverage.jsonl'),
            'semantic_coverage_limit': 'Synthetic truth checked; no exhaustive proof of all natural-language entailment.',
        })

    category_ids = collections.defaultdict(set)
    class_ids = collections.defaultdict(set)
    for f in findings:
        category_ids[f['category']].add(f['question_id'])
        class_ids[f['class']].add(f['question_id'])
    confirmed = class_ids['confirmed_error']
    limits = class_ids['observation_limit'] | class_ids['citation_limit']
    summary = {
        'audited_question_instances': len(coverage), 'non_recall_instances': len(gold),
        'recall_instances': len(recall),
        'reference_matches_synthetic_truth': sum(r['reference_matches_synthetic_truth_at_semantic_target'] for r in coverage),
        'structural_checks': read('structure_summary.json')['checks'],
        'structural_failures': read('structure_summary.json')['failed_checks'],
        'distinct_questions_by_class': {k: len(v) for k, v in sorted(class_ids.items())},
        'distinct_questions_by_category': {k: len(v) for k, v in sorted(category_ids.items())},
        'distinct_confirmed_error_questions': len(confirmed),
        'distinct_additional_observation_or_citation_limit_questions': len(limits - confirmed),
        'distinct_uncertain_questions': len(class_ids['uncertainty']),
        'all_narrow_question_finding_union': len(set(by_question) & set().union(*class_ids.values())),
        'question_ids_by_class': {k: sorted(v) for k, v in sorted(class_ids.items())},
        'confirmed_errors_by_user': dict(collections.Counter(qid.split('_')[0] for qid in sorted(confirmed))),
        'category_overlap_note': 'Wrong gold and wrong acceptance refer to the same question. Calendar findings include the four recall and 47 non-recall temporal findings; these were deduplicated.',
        'source_quality': flag_review,
        'native_world': read('source/native_world_checks.json')['totals'],
        'integration': {
            'normal_loader_monthly_questions_loaded': len(read('source/integration.json')['loader']['monthly_ids_loaded']),
            'normal_loader_legacy_questions_loaded': read('source/integration.json')['loader']['loaded_questions'],
            'runner_enforces_monthly_cutoff': False,
            'offline_reproduction': 'source/integration.json',
        },
        'selection_gaps': {
            'unknown_abstention_questions_excluded': sum(r['pool_abstention_unanswerable'] for r in read('selection.json')),
            'unknown_abstention_questions_selected': sum(r['probe_abstention_unanswerable'] for r in read('selection.json')),
            'selected_answerable_abstention_controls': sum(q['type'] == 'abstention' for q in questions.values()),
            'prediction_questions_selected': sum(q['type'] == 'prediction' for q in questions.values()),
            'current_pattern_questions_selected': sum(q['type'] == 'pattern_current' for q in questions.values()),
            'final_unprobed_days_per_user': 28,
        },
        'limits': [
            'Wrong synthetic truth, temporal eligibility, actual observability, citation completeness, and source-contract problems are separate findings.',
            'Saved fidelity flags were manually interpreted against source transcripts; replay is not a new semantic validation.',
            'Source-session exposure does not establish an incorrect question answer and is excluded from question-error totals.',
            'Calendar targets beyond the probe violate historical scheduling even when known routines make a future value predictable.',
            'No paid API calls or full agent evaluation were run; offline runner stub scores are not performance results.',
            'These checks cover every record; they do not prove absence of all possible natural-language defects.',
        ],
    }
    save('findings.json', findings)
    save('summary.json', summary)
    (OUT / 'question_coverage.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in coverage))

    lines = ['# Per-question audit findings', '',
             'Only question-level findings are listed here. Source flags, coverage omissions, and evaluator integration are documented in the main report. Counts concern instances, including retrospective copies. Wrong gold and wrong acceptance are the same question.', '']
    labels = [('confirmed_error', 'Confirmed scheduling or reference errors'),
              ('observation_limit', 'Observed-answer limitations'),
              ('citation_limit', 'Specific citation limitations'),
              ('uncertainty', 'Residual closure or wording uncertainty')]
    for finding_class, label in labels:
        ids = class_ids[finding_class]
        lines += [f'## {label} ({len(ids)} distinct questions)', '',
                  '| ID | Probe | Type | Finding | Question location |',
                  '| --- | --- | --- | --- | --- |']
        for qid in sorted(ids):
            record = next(r for r in coverage if r['id'] == qid)
            categories = sorted({f['category'] for f in by_question[qid] if f['class'] == finding_class})
            location = f"[{record['file']}:{record['line']}]({ROOT / record['file']}:{record['line']})"
            lines.append(f"| `{qid}` | {record['probe_date']} | {record['type']} | {', '.join(categories)} | {location} |")
        lines.append('')
    lines += ['Each record in `findings.json` preserves the source workstream’s observed value, expected behavior, source locations, and rationale. `question_coverage.jsonl` accounts for all 1,000 records, including those without a specific finding.', '']
    (OUT / 'question_findings.md').write_text('\n'.join(lines))
    print(json.dumps({k: summary[k] for k in ('audited_question_instances', 'reference_matches_synthetic_truth', 'distinct_questions_by_class', 'distinct_questions_by_category', 'distinct_confirmed_error_questions', 'distinct_additional_observation_or_citation_limit_questions', 'all_narrow_question_finding_union')}, indent=2))


if __name__ == '__main__':
    main()
