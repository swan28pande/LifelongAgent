# Implement and verify the monthly loader/runner

Type: task
Status: resolved

## Comments

- User approved cumulative repeats retaining the original probe date and references.
- All new setup artifacts stay in `setup_2/`; the user's explicit folder scope takes
  precedence over the root issue-tracker/documentation destinations.

## Answer

Implemented `loader.py`, `runner.py`, and `run.py` with cumulative monthly questions,
original probe dates/references, the unchanged OursV4D adapter and common judge,
per-month reporting, concurrent evaluation within a fixed checkpoint, persistent
per-user memory, rollback of unfinished monthly ingestion, and resumable evaluation.
Protocol decisions are in `../../DECISIONS.md`; commands and validation in `../../README.md`.

Validation: 21 offline tests passed without skips. The full current dataset schedule
contains 2,738 unique sessions and 12,084 answer instances over 24 checkpoints per
user. Existing tracked files have no diff; all 40 primary inputs/code hashes from
the earlier dataset audit remain unchanged. No live model or paid API was called.
