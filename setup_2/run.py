"""Separate entry point: python -m setup_2.run --users u1 --dry-run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from experiments import config
from .loader import DATA_DIR, describe, load
from .methods import METHODS


SETUP_DIR = Path(__file__).resolve().parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Monthly cumulative memory_v4 and baseline evaluation')
    parser.add_argument('--method', choices=METHODS, default='ours_v4d')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--users', nargs='+', help='Default: all v2 users')
    parser.add_argument('--run', default='seed0', help='Separate result directory name')
    parser.add_argument('--through', help='Stop after YYYY-MM; may extend on resume')
    parser.add_argument('--model', default=config.MODEL)
    parser.add_argument('--judge-model', default=config.JUDGE_MODEL)
    parser.add_argument('--workers', type=int, default=config.ANSWER_WORKERS)
    parser.add_argument('--user-workers', type=int, default=1,
                        help='Parallel user processes; months remain sequential within each user')
    parser.add_argument('--judge-concurrency', type=int, default=config.JUDGE_CONCURRENCY)
    parser.add_argument('--dry-run', action='store_true', help='Print schedule without models or writes')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run):
        parser.error('--run must be a simple directory name')
    if min(args.workers, args.user_workers, args.judge_concurrency) < 1:
        parser.error('Worker counts must be positive')
    if args.through:
        try:
            if not re.fullmatch(r'\d{4}-\d{2}', args.through):
                raise ValueError
            dt.date.fromisoformat(args.through + '-01')
        except ValueError:
            parser.error('--through must be YYYY-MM')
    try:
        timelines = load(args.data_dir, args.users)
        plan = describe(timelines, args.through)
        plan.update({'method': args.method, 'method_profile': METHODS[args.method][2]})
        if any(not months for months in plan['users'].values()):
            raise ValueError('--through precedes the selected history')
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        from .runner import run
        output = SETUP_DIR / 'results' / args.run
        if args.method != 'ours_v4d':
            folder = METHODS[args.method][0].split('.')[-2]
            output = SETUP_DIR / 'baselines' / folder / 'results' / args.run
        summary = run(timelines, output, model=args.model, judge_model=args.judge_model,
                      workers=args.workers, judge_concurrency=args.judge_concurrency,
                      through=args.through, method_name=args.method, user_workers=args.user_workers)
        print(json.dumps({'results': str(output), 'method': summary['method'], 'complete': summary['complete'],
                          'answer_instances': summary['n'], 'accuracy': summary['accuracy']}, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as e:
        print(f'{type(e).__name__}: {e}\nRerun the same command to resume.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
