"""
Play the assistant yourself against the grounded user agent, in the terminal.

    python -m experiments.system_3.play --user u1 --date 2026-03-17
    python -m experiments.system_3.play --user u5 --date 2026-03-04 --validate

Type the assistant's reply after each user message. Commands: /status shows the checklist,
/end finishes the day (and validates with --validate), /quit leaves without validating.
"""

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from generator_v2 import config as gen_config
from generator_v2.llm import LLM
from generator_v2.schema import load_ladder, load_spec
from generator_v2.simulator import simulate
from generator_v2.validator import build_check, validate

from experiments import config
from experiments.system_3.simulator import MAX_TURNS
from experiments.system_3.user_agent import GroundedUser


def show_status(items, covered):
    for i in items:
        print(f"   [{'x' if i.id in covered else ' '}] {i.id}: {i.text}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.system_3.play")
    ap.add_argument("--user", default="u1")
    ap.add_argument("--date", help="session day YYYY-MM-DD (default: the user's first session)")
    ap.add_argument("--model", default=config.MODEL)
    ap.add_argument("--validate", action="store_true", help="run the dataset validator at the end")
    args = ap.parse_args(argv)

    spec = load_spec(gen_config.PERSONA_DIR / f"{args.user}.yaml", load_ladder(gen_config.LADDER_PATH))
    world = simulate(spec)
    sessions = [d for d in world.days if d.has_session]
    day = next((d for d in sessions if d.date.isoformat() == args.date), None) if args.date else sessions[0]
    if day is None:
        print(f"{args.user} has no session on {args.date}")
        return 1
    logs = Path(tempfile.mkdtemp(prefix="system3_play_"))
    user = GroundedUser(spec, world, LLM(args.model, 0.7, logs / "user_agent_log.jsonl"))
    loop = asyncio.new_event_loop()
    items, covered, turns = user.checklist(day), set(), []

    print(f"\n=== {spec.name} ({args.user}), {day.weekday} {day.date} — you are the ASSISTANT ===")
    print("\n" + user.guardrails(day) + "\n\nCHECKLIST:")
    show_status(items, covered)
    print("\nCommands: /status, /end, /quit\n")

    for k in range(MAX_TURNS + 2):
        msg = loop.run_until_complete(user.next_message(
            day, items, covered, turns, f"play #{k}", insist=k >= MAX_TURNS - 1))
        covered |= set(msg["covered"])
        turns.append({"speaker": "user", "text": msg["message"]})
        print(f"\n{spec.name}: {msg['message']}")
        print(f"   (covered now: {', '.join(msg['covered']) or 'nothing'}; "
              f"pending: {len([i for i in items if i.id not in covered])}; done={msg['done']})")
        if msg["done"]:
            print("   (the user agent considers the day complete; /end to finish or keep chatting)")
        while True:
            reply = input("\nassistant> ").strip()
            if reply == "/status":
                show_status(items, covered)
                continue
            break
        if reply == "/quit":
            return 0
        if reply == "/end":
            break
        turns.append({"speaker": "assistant", "text": reply})

    print("\nFINAL CHECKLIST:")
    show_status(items, covered)
    if args.validate:
        validator = LLM(gen_config.VALIDATOR_MODEL, 0.0, logs / "validator_log.jsonl")
        fails, _ = loop.run_until_complete(validate(validator, spec, build_check(spec, world, day), turns, "play check"))
        print("\nVALIDATOR:", "passed" if not fails else "")
        for f in fails:
            print("  -", f)
    print(f"\n(LLM logs: {logs})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
