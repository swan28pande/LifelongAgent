# Audit of v2 monthly probing questions

## Objective

Audit all 1,000 questions in `datasets/v2/u1..u5/probing_questions.json`, beyond the 51 previously identified explicit future-date questions. Determine which errors can be established from the source data and generator, and distinguish remaining semantic uncertainties. Do not edit datasets, agents, or evaluation code. No paid API calls.

## Definitions and assumptions

- A probe's date and viewpoint day define the last conversation available to an evaluated agent.
- `world_state.json` describes synthetic truth; recorded conversations describe what an agent could observe. These are separate checks.
- Source-question IDs identify the reused entries in `qa_pool.json`; equality to the pool does not prove correctness at a changed viewpoint.
- Inferred questions can legitimately cite representative earlier observations; absence of a conversation on the target day is not automatically an error.
- Retrospective copies retain their original question and answer, so relative words such as "current" require renewed validation.
- Generator validators' recorded flags are source evidence, not new semantic judgments.
- Counts of error categories may overlap. Report both category counts and distinct affected questions.

## Workstreams

1. Structure and scheduling: all files and records, IDs, dates, counts, provenance, duplicates, retrospective links, calendar target dates, probe coverage, and reproducibility.
2. Gold answers and implicit timing: non-recall question families, rule periods, effective/stated dates, fact updates, event status, durations, causes, distractors, abstention, and moving-viewpoint semantics.
3. Recall truth and evidence: all recall targets against per-day world values and exceptions; completeness and suitability of cited evidence and observed conversation support.
4. Source and execution consistency: aggregate conversations versus individual sessions and world metadata, fidelity flags, experiment dataset copies, and loader/runner use of monthly cutoffs.

## Evidence and sources

Prefer the local canonical datasets, `generator_v2/qa.py`, `probing.py`, `simulator.py`, `rules.py`, conversation/validator code, recorded session validations, and the experiment loader/runner. Every confirmed issue needs record IDs, source locations, the observed value, and expected behavior/value with justification.

## Counter-evidence checks

- Is an apparent contradiction just a paraphrase, inference, temporary exception, retrospective statement, or multi-value fact?
- Does a future event date refer to a known plan, or ask for the status at a future date?
- Are references valid in synthetic truth but unsupported in the actual text?
- Does unchanged pool wording stay valid when moved to a new probe?
- Are missing sessions deliberate sampling rather than missing data?

## Stopping conditions

Finish when every probing record is accounted for by structural and family-specific checks; each confirmed issue has reproducible evidence; saved transcript flags and integration behavior have been examined; and residual semantic limitations are explicit. Do not claim an exhaustive proof of every possible language error.
