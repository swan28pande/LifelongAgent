"""Summarize saved observations without making additional model calls."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def select(events, **fields):
    return [event for event in events if all(event.get(key) == value for key, value in fields.items())]


def active_span(rows):
    ends = [datetime.fromisoformat(row['utc']).timestamp() for row in rows]
    return max(ends) - min(end-row['seconds'] for end, row in zip(ends, rows))


def summarize(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    run = json.loads((directory / 'run_summary.json').read_text())
    events = records(directory / 'events.jsonl')
    phases = json.loads((directory / 'phases.json').read_text())
    rows, totals = [], []
    stages = sorted({phase['stage'] for phase in phases})
    for stage in stages:
        for method in ('timem', 'naive_rag', 'full_context'):
            row = {'stage': stage, 'method': method}
            for condition in ('solitary', 'concurrent'):
                matching = select(events, stage=stage, method=method, condition=condition)
                logical = select(matching, kind='logical')
                latencies = [event['seconds'] for event in logical if event['ok']]
                tokens = select(matching, kind='tokens')
                if len(tokens) != len(latencies):
                    raise ValueError(f'Incomplete callback accounting: {stage}/{method}/{condition}')
                rounds = sorted({event['round'] for event in logical})
                elapsed = sum(active_span(select(logical, round=number)) for number in rounds)
                if not logical:
                    row[condition] = None
                    continue
                row[condition] = {
                    'n': len(logical), 'failed': sum(not event['ok'] for event in logical),
                    'mean_s': statistics.mean(latencies) if latencies else None,
                    'median_s': statistics.median(latencies) if latencies else None,
                    'max_s': max(latencies) if latencies else None, 'active_seconds': elapsed,
                    'requests_per_minute': 60*len(latencies)/elapsed,
                    'input_tokens': sum(event.get('input_tokens',0) for event in tokens),
                    'output_tokens': sum(event.get('output_tokens',0) for event in tokens),
                    'cached_input_tokens': sum(event.get('input_token_details',{}).get('cache_read',0)
                                               for event in tokens),
                    'http_statuses': dict(Counter(str(event['status']) for event in select(matching,kind='http'))),
                }
            if not all(row.get(condition) and row[condition]['mean_s'] is not None
                       for condition in ('solitary','concurrent')):
                continue
            row['mean_latency_change_pct'] = 100*(row['concurrent']['mean_s']/row['solitary']['mean_s']-1)
            rows.append(row)
        aggregate = {'stage': stage}
        rounds = sorted({phase['round'] for phase in phases if phase['stage']==stage})
        for condition in ('solitary', 'concurrent'):
            elapsed = 0
            for number in rounds:
                batches = select(phases, stage=stage, condition=condition, round=number)
                if not batches:
                    continue
                elapsed += (sum(batch['seconds'] for batch in batches) if condition=='solitary'
                            else max(batch['seconds'] for batch in batches))
            matching = select(events, stage=stage, condition=condition)
            inputs = sum(event.get('input_tokens',0) for event in select(matching,kind='tokens'))
            logical = select(matching,kind='logical')
            aggregate[condition] = {
                'wall_seconds': elapsed, 'requests': len(logical),
                'requests_per_minute': 60*len(logical)/elapsed if elapsed else None,
                'input_tokens': inputs, 'input_tokens_per_minute': 60*inputs/elapsed if elapsed else None,
            }
        if not aggregate['concurrent']['wall_seconds']:
            continue
        aggregate['speedup'] = aggregate['solitary']['wall_seconds']/aggregate['concurrent']['wall_seconds']
        totals.append(aggregate)
    return {'manifest': manifest, 'run': run, 'per_method': rows, 'aggregate': totals}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('first_month', type=Path)
    parser.add_argument('late_history', type=Path)
    parser.add_argument('--scaled', type=Path)
    args = parser.parse_args()
    report = {'first_month': summarize(args.first_month), 'late_history': summarize(args.late_history)}
    if args.scaled:
        report['scaled_api'] = summarize(args.scaled)
    initial = args.first_month.parent / 'u5_march_20261004_a' / 'run_summary.json'
    if initial.exists():
        report['initial_preparation_attempt'] = json.loads(initial.read_text())
    root = Path(__file__).resolve().parents[2]
    hashes = report['first_month']['manifest']['source_hashes']
    report['source_files_unchanged'] = all(hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
                                          for path,digest in hashes.items())
    from setup_2.loader import load
    report['dataset_unchanged'] = load(users=['u5'])[0].input_hashes==report['first_month']['manifest']['dataset_hashes']
    output = Path(__file__).parent / 'summary.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    lines = ['# Observed Gemini concurrency results', '',
             'These are live API throughput measurements. See METHODOLOGY.md for controls and limits.', '']
    profiles = [('First month: 18 concurrent calls','first_month'),
                ('Late-history stress: 12 concurrent calls','late_history')]
    if args.scaled:
        profiles.append(('API replay: 120 answers / 240 judgments','scaled_api'))
    for title,key in profiles:
        data = report[key]
        lines.extend(['## '+title,'',
            '| Stage | Method | Requests per condition | Alone mean (s) | Concurrent mean (s) | Change |',
            '| --- | --- | ---: | ---: | ---: | ---: |'])
        for row in data['per_method']:
            lines.append(f"| {row['stage']} | {row['method']} | {row['solitary']['n']} | "
                         f"{row['solitary']['mean_s']:.2f} | {row['concurrent']['mean_s']:.2f} | "
                         f"{row['mean_latency_change_pct']:+.1f}% |")
        lines.extend(['','| Stage | Alone wall (s) | Together wall (s) | Speedup | Input tokens/min together |',
                      '| --- | ---: | ---: | ---: | ---: |'])
        for row in data['aggregate']:
            lines.append(f"| {row['stage']} | {row['solitary']['wall_seconds']:.2f} | "
                         f"{row['concurrent']['wall_seconds']:.2f} | {row['speedup']:.2f}x | "
                         f"{row['concurrent']['input_tokens_per_minute']:,.0f} |")
        lines.extend(['',f"Total run time, including preparation/warmup: {data['run']['wall_seconds']:.2f} seconds.",
                      f"All HTTP statuses: `{data['run']['http_statuses']}`.",
                      f"Measurement completed: {data['run']['complete']}; failure: {data['run']['failure']}.",''])
    lines.extend([f"Protected source files unchanged: {report['source_files_unchanged']}.",
                  f"Canonical input hashes unchanged: {report['dataset_unchanged']}.",''])
    output.with_suffix('.md').write_text('\n'.join(lines))
    print(output.read_text())


if __name__ == '__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    main()
