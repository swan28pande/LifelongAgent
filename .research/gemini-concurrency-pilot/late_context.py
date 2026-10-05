"""API stress with real late-history full context and two existing retrieval streams."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from pilot import Recorder, Usage, config, create, Instance, load, save, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    for name in (args.source_run, args.run):
        if Path(name).name != name:
            parser.error('Run names must be directory names')
    source = Path(__file__).parent / args.source_run
    if not json.loads((source / 'run_summary.json').read_text())['complete']:
        raise ValueError('The source pilot must finish successfully first')
    output = Path(__file__).parent / args.run
    output.mkdir(exist_ok=True)
    if (output / 'events.jsonl').exists():
        raise ValueError('Use a fresh run name')
    recorder = Recorder(output)
    start = time.monotonic()
    timeline = load(users=['u5'])[0]
    instance = Instance(timeline.id, [], [], timeline.name)
    methods, phases = {}, []
    names = ('timem', 'naive_rag', 'full_context')
    questions = {name: timeline.checkpoints[0].questions[:4] for name in names}
    questions['full_context'] = timeline.checkpoints[-1].questions[-4:]
    for name in names:
        store = output / name / 'store' if name == 'full_context' else source / name / 'store'
        store.mkdir(parents=True, exist_ok=True)
        methods[name] = create(name, store, instance, config.MODEL, Usage(recorder))
        recorder.attach(methods[name].llm, name)
    full = methods['full_context']
    for checkpoint in timeline.checkpoints:
        for session in checkpoint.sessions:
            full.ingest(session)
        full.finalize()
    manifest = json.loads((source / 'manifest.json').read_text())
    manifest.update({
        'purpose': 'API-only stress: late-history full context overlaps two first-month retrieval streams',
        'source_run': args.source_run, 'questions_per_method': 4,
        'full_context_cutoff': timeline.checkpoints[-1].cutoff,
        'full_context_sessions': len(full.sessions), 'full_context_chars': len(full.context),
        'estimated_full_context_tokens': (len(full.context)+3)//4,
        'effective_answer_concurrency': 4, 'combined_concurrency': 12,
        'judge': 'not repeated here; covered by the first-month comparison',
        'limitation': 'Mixed history sizes test API contention; this is not an accuracy benchmark',
        'question_ids': {name: [q.id for q in questions[name]] for name in names},
    })
    save(output / 'manifest.json', manifest)
    print('LATE_HISTORY', manifest['full_context_cutoff'], len(full.sessions), len(full.context), flush=True)

    def call(name, q, condition, round_number):
        meta = {'method': name, 'stage': 'answer', 'condition': condition,
                'round': round_number, 'question_id': q.id}
        response = recorder.call(meta, lambda: methods[name].answer(q))
        recorder.emit('answer', **meta, response=response.text, response_meta=response.meta)

    def phase(selected, condition, round_number):
        before = time.monotonic()
        with ThreadPoolExecutor(max_workers=8*len(selected)) as pool:
            futures = [pool.submit(call, name, q, condition, round_number)
                       for name in selected for q in questions[name]]
            for future in futures:
                future.result()
        elapsed = time.monotonic()-before
        for name in selected:
            row = summarize(recorder.events, {'method': name, 'stage': 'answer',
                            'condition': condition, 'round': round_number}, elapsed)
            phases.append(row)
            print('PHASE', json.dumps(row), flush=True)
        save(output / 'phases.json', phases)

    failure = None
    try:
        for name in names:
            call(name, questions[name][0], 'warmup', 0)
        for name in names:
            phase([name], 'solitary', 1)
        phase(names, 'concurrent', 1)
        phase(names, 'concurrent', 2)
        for name in reversed(names):
            phase([name], 'solitary', 2)
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)[:500]}
        raise
    finally:
        save(output / 'run_summary.json', {
            'complete': failure is None, 'failure': failure,
            'wall_seconds': time.monotonic()-start,
            'http_statuses': dict(Counter(str(e['status']) for e in recorder.events if e['kind']=='http')),
            'input_tokens': sum(e.get('input_tokens',0) for e in recorder.events if e['kind']=='tokens'),
            'output_tokens': sum(e.get('output_tokens',0) for e in recorder.events if e['kind']=='tokens'),
        })
        for method in methods.values():
            if hasattr(method,'close'):
                method.close()
            method.llm.client.close()
        print('LATE_PILOT_FINISHED', output, flush=True)


if __name__ == '__main__':
    main()
