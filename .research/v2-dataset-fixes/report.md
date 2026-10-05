# v2 dataset repairs

The remaining findings from the updated dataset audit are repaired in the generator
and regenerated canonical/experiment datasets. The hidden world, session dates and
monthly evaluation protocol are preserved. Monthly questions remain historical,
as explicitly confirmed by the user on October 3, 2026.

| Check | Result |
| --- | --- |
| Monthly questions checked against world truth | 1,000 / 1,000 valid |
| Current-reference errors | 0 |
| Future historical targets or future reference endpoints | 0 |
| Unsupported recall phases | 0 |
| Audited rejected change-date alternatives | All 20 finding instances resolved |
| Missing completed-duration closure citations | 0 |
| Uncited pattern values or required fact removals | 0 |
| Genuinely unknown questions | All 30 distinct templates selected; 41 scheduled instances |
| Current-pattern questions | 12 scheduled instances |
| Timeline coverage | 24 probes/user, including February 28, 2028 |
| Dataset size | 200 questions/user; 2,738 sampled sessions preserved |
| Canonical/experiment output parity | 40 / 40 file pairs identical |
| Source text repairs | 21 sessions |
| Previously flagged sessions revalidated with Gemini | 26 / 26 passed |
| Offline regression tests | 90 generator + 21 monthly setup = 111 passed |

The five world-state files are byte-for-byte unchanged. The agent, shared judge,
summaries/distillation, retrieval planning and five-tool limit were not edited.

## What changed

- Current facts and answerable controls use the complete state at their own probe
  date. Priya's two January hobby questions now both include piano: the regenerated
  records are `u5_p202701_0067` and `u5_p202701_0070`.
- Facts cite actual introductions, replacements and removals, including delayed
  historical disclosures. Priya's June 15 vegetarian state is supported by the
  June 19 retrospective statement; the generator no longer substitutes day 1.
- Change questions wait for an observation distinguishing the old and new routines.
  Their accepted dates include compatible boundaries and weekly rotations, using
  only information through the original probe. Two January cases move to February
  because the first distinguishing choices are disclosed then.
- Completed durations include a closing observation and are scheduled after it.
  Ongoing durations retain explicit "as of" wording. Pattern citations cover the
  applicable phases and all described values.
- Unobserved-phase recalls and unmentioned recalls near transitions are excluded.
  Both specifically flagged recall questions are absent from the regenerated pool.
- Sampling retains proportional month allocation while guaranteeing question-family
  coverage, genuine unknown-answer questions and answerable controls. The final
  partial month introduces new questions. Prediction remains a separate pool
  capability and is intentionally excluded from the historical monthly evaluation.
- Required updates can name an otherwise absent daily topic without adding a daily
  choice or discretionary follow-up. Writer/validator prompts agree on this rule.
  Validation now has the date and disclosed fact-change history, avoiding false
  missing-date/backstory flags. Two underspecified event dates were clarified.
- Source edits are minimal. Previous extraction/failure records remain in each
  revalidated session's `repair_validation` field; original backups and API logs
  stay in the project-local temporary directory. Every final extraction came from
  an actual Gemini validation call, not from manufactured passing metadata.
- QA, probes and fidelity commands synchronize generated outputs into the experiment
  copy. Fidelity handles unavailable LFS logs by preserving recorded generation
  usage with explicit provenance. Repair-validation usage is recorded separately.

## Reproduction and evidence

Regenerate QA/probes and rebuild conversation/fidelity outputs from the repaired
session files using the existing project environment:

```bash
TMPDIR="$PWD/tmp/v2-dataset-fixes" tmp/memory-v3-update-validation/bin/python -m generator_v2.run qa --user all --samples 0
TMPDIR="$PWD/tmp/v2-dataset-fixes" tmp/memory-v3-update-validation/bin/python -m generator_v2.run fidelity --user all
TMPDIR="$PWD/tmp/v2-dataset-fixes" tmp/memory-v3-update-validation/bin/python .research/v2-dataset-fixes/verify_dataset.py
```

- [Verification](verification.json) records all five users and the cumulative schedule.
- [Question coverage](question_coverage.json) covers all 1,000 regenerated records;
  [gold/](gold/) retains the independent world-truth checks with documented adaptations.
- [Finding resolutions](finding_resolutions.json) maps all 29 earlier affected question
  records. IDs and seeded samples changed on regeneration; it distinguishes new
  records from cases tested at generator level but not selected in a new sample.
- [Boundary resolutions](boundary_resolutions.json) checks all 20 full-text-reviewed
  date finding instances. [Source repairs](source_repairs.json) records exact edited
  turns, before/after hashes and validator outcomes; [source_validation/](source_validation/)
  contains the real final extractions and prompt hashes.
- [Tests](tests.json) records commands, pass counts and runtimes. [Repair usage](repair_usage.json)
  separates the Gemini validation calls from original dataset-generation usage.

The corrected inputs require a new evaluation run name; the monthly runner rejects
mixing old and new input hashes. Its original-date repetition and judging behavior
remain unchanged. These checks establish synthetic truth and resolve the audited
observation limits; they do not prove unique natural-language rule/causal inference
for every possible interpretation, or revalidate all previously passing chats.
