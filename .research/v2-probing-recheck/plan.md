# Verification plan

## Objective

Determine whether the updated canonical v2 datasets resolve the issues recorded in `.research/v2-probing-audit/report.md`, and identify remaining or new issues relevant to monthly evaluation.

## Definitions and assumptions

- A scheduling error asks about a historical target or completed interval later than its probe cutoff.
- A synthetic reference mismatch is different from an answer that agrees with hidden world state but cannot be established from recorded conversation.
- A citation gap is not automatically an unanswerable question when supporting text exists elsewhere in the pre-probe prefix.
- Removed or resampled questions do not prove that their generator or source defect was repaired.
- Original probe IDs can change when sampling changes; use `original_id` and actual question payloads to match findings.
- Repeated questions in `setup_2/` retain their original probe dates. Earlier legacy-runner integration findings are evaluated separately from dataset defects.

## Subquestions and evidence

1. What actually changed? Compare SHA-256 hashes to the original audit, and inventory current question types, probes, and sources.
2. Are scheduling, metadata, and payload invariants correct? Check every current instance and independently reproduce the probe builder in memory.
3. Are non-recall references, accepted dates, empty facts, durations, and citations corrected? Reuse offline checks where valid, then inspect current raw questions and source prefixes.
4. Are recall answerability and boundary findings corrected or absent through resampling? Check all current recall instances, independently recompute rule values, and review new paraphrase/boundary cases as needed.
5. Are source/copy problems and coverage omissions corrected? Compare source hashes and canonical/experiment copies, inspect coverage and current exposure to retained source problems.

Independent recall and non-recall/source workstreams use fresh contexts. Evidence is saved separately from interpretations. A fresh synthesis review will assess the resulting evidence and counts.

## Falsification questions

- Did identifiers change while the underlying bad question remained?
- Were defects merely excluded from this 200-question sample?
- Does a temporal fix leave an unsupported exact answer or omit necessary citations?
- Do accepted alternatives now include a correct counterexample, or merely retain the hidden date?
- Does agreement with the generator reproduce its mistakes?

## Stopping conditions

Every current question is covered by structural and family checks; every prior finding is matched to a current record, shown absent, or explicitly unresolved. Remaining confirmed issues have precise current locations and evidence. Record all limits. No model calls, agent evaluation, dataset edits, or writes to root `/tmp/`.
