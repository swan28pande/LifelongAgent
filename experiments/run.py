"""
Run memory methods on benchmarks with one shared harness.

    venv/bin/python -m experiments.run list
    venv/bin/python -m experiments.run run --method ours_v3 --benchmark v2 --users u1
    venv/bin/python -m experiments.run run --method naive_rag --benchmark locomo
    venv/bin/python -m experiments.run run --method mem0 --benchmark longmemeval --sample 100
    venv/bin/python -m experiments.run run --method full_context --benchmark v2 --users u1 --seed 1
    venv/bin/python -m experiments.run report

Runs resume: rerun the same command after an interruption.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from . import config, methods
from .benchmarks import locomo, longmemeval, v2
from .core import runner

BENCHMARKS = ("v2", "locomo", "longmemeval")


def load_instances(args):
    if args.benchmark == "v2":
        return v2.load(args.users)
    if args.benchmark == "locomo":
        return locomo.load(args.conversations)
    return longmemeval.load(args.sample, seed=0)


def cmd_run(args) -> int:
    instances = load_instances(args)
    if args.limit:
        for inst in instances:
            inst.questions = inst.questions[: args.limit]
    run_name = args.run or (f"seed{args.seed}" + (f"_sample{args.sample}" if args.sample else ""))
    n_q = sum(len(i.questions) for i in instances)
    n_s = sum(len(i.sessions) for i in instances)
    print(f"{args.method} on {args.benchmark}: {len(instances)} instance(s), {n_s} sessions, "
          f"{n_q} questions → results/experiments/{args.benchmark}/{args.method}/{run_name}/", flush=True)
    summary = runner.run(methods.get(args.method), args.benchmark, instances, run_name,
                         model=args.model, judge_model=args.judge_model, workers=args.workers)
    print(json.dumps({k: summary[k] for k in ("n", "accuracy", "usage")}, indent=2))
    for cat, v in summary["by_category"].items():
        print(f"  {cat:<28} n={v['n']:<4} judge {v['judge']:.1%}" + (f"  F1 {v['f1']:.3f}" if "f1" in v else ""))
    return 0


def cmd_report(_args) -> int:
    rows = {}
    for path in sorted(config.RESULTS_DIR.glob("*/*/*/summary.json")):
        s = json.loads(path.read_text())
        rows.setdefault(s["method"], {}).setdefault(s["benchmark"], []).append(s["accuracy"])
    if not rows:
        print("no finished runs yet")
        return 0
    benches = sorted({b for r in rows.values() for b in r})
    print(f"{'method':<16}" + "".join(f"{b:>22}" for b in benches))
    for m, r in sorted(rows.items()):
        cells = []
        for b in benches:
            accs = r.get(b, [])
            if not accs:
                cells.append(f"{'-':>22}")
            else:
                mean = sum(accs) / len(accs)
                sd = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
                cells.append(f"{mean:>14.1%} ±{sd:.1%} (n={len(accs)})"[-22:].rjust(22))
        print(f"{m:<16}" + "".join(cells))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("report")
    r = sub.add_parser("run")
    r.add_argument("--method", required=True, choices=sorted(methods.METHODS))
    r.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    r.add_argument("--users", nargs="+", help="v2 only, e.g. u1 u2 (default: all generated users)")
    r.add_argument("--conversations", type=int, help="locomo only: first N conversations")
    r.add_argument("--sample", type=int, help="longmemeval only: stratified sample of N questions")
    r.add_argument("--limit", type=int, help="first N questions per instance (smoke tests)")
    r.add_argument("--seed", type=int, default=0, help="repeat index; separate runs for variance")
    r.add_argument("--run", help="override the run folder name")
    r.add_argument("--model", default=config.MODEL)
    r.add_argument("--judge-model", default=config.JUDGE_MODEL)
    r.add_argument("--workers", type=int, default=config.ANSWER_WORKERS)
    args = ap.parse_args(argv)

    if args.cmd == "list":
        print("methods:    " + ", ".join(sorted(methods.METHODS)))
        print("benchmarks: " + ", ".join(BENCHMARKS))
        return 0
    return cmd_report(args) if args.cmd == "report" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
