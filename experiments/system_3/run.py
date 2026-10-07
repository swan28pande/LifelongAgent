"""
Run system_3: a grounded user agent chats live with a memory-backed assistant over the v2 timeline.

    python -m experiments.system_3.run --method naive_rag --users u1 --through 2026-03
    python -m experiments.system_3.run --method ours_v4d --users u1 --through 2026-03 --run pilot

Rerun the same command to resume from the last completed day.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from experiments import config
from experiments.system_3 import simulator
from experiments.system_3.methods import SYSTEM3_METHODS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.system_3.run")
    ap.add_argument("--method", required=True, choices=sorted(SYSTEM3_METHODS))
    ap.add_argument("--users", nargs="+", default=["u1", "u2", "u3", "u4", "u5"])
    ap.add_argument("--through", help="last month to simulate, e.g. 2026-03 (default: whole timeline)")
    ap.add_argument("--run", default="seed0", help="result folder name")
    ap.add_argument("--days", type=int, help="only the first N session days (smoke tests)")
    ap.add_argument("--model", default=config.MODEL)
    args = ap.parse_args(argv)
    summary = simulator.run(args.method, args.users, args.run, args.model, args.through, args.days)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
