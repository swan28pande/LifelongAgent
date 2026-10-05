# Repair the updated v2 dataset

Status: complete

Repair the remaining findings in `.research/v2-probing-recheck/`, including the
question generator so regeneration retains the corrections. Preserve the world
state, sampled sessions, agent, grader, and monthly cumulative evaluation protocol.

- Ground current answers at each question's original probe date. Repeated questions
  keep that date and gold; newly scheduled retrospective records get their own gold.
- Cite the statements and observations that establish each answer, including delayed
  historical disclosures, removals, and completed-interval closure.
- Omit underdetermined recall and pattern questions rather than invent observations.
- Accept observable uncertainty in change dates instead of requiring hidden precision.
- Include unknown-answer controls, current patterns, and the final partial month.
- Keep monthly questions historical. The user explicitly confirmed this on October
  3, 2026; prediction remains a separate end-of-timeline QA capability.
- Repair the audited conversation defects minimally, retain provenance, and validate
  repaired sessions against the unchanged world with the existing Gemini validator.
- Synchronize the canonical and experiment dataset copies and verify their equality.

All temporary files and logs stay under the project directory. No root `/tmp` use.

## Acceptance

Generator regression tests and existing generator/monthly-loader tests pass. All
five users have 200 questions across 24 probe checkpoints, with no future historical
targets or evidence. The audited defects have explicit resolutions and source
repairs have genuine validation results. All duplicate dataset JSON files agree.

## Result

Completed: 1,000 references validated; 24 probes and 200 questions per user; all
30 unknown templates included; all 26 flagged sessions genuinely revalidated, with
21 text repairs; 40 output pairs synchronized. 111 regression tests pass. Worlds,
agent and shared judge remain unchanged. Evidence: `.research/v2-dataset-fixes/report.md`.
