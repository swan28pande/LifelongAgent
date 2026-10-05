# Updated non-recall questions: verified fixes and remaining findings

All **504 current non-recall instances** were checked offline by semantic family. **502 references match synthetic truth; two January hobby references are wrong.** All answer-type/grading combinations match their family. No dataset, generator, agent, or loader was changed; no model was called.

## Fixes that are established

- No current non-recall question has an explicit target after its probe; no completed duration answer ends after its probe. The new allocator considers dates in the question, answer, and acceptance list, and complete named-month endpoints. Twenty-six of the 47 prior non-recall future-target instances have matching current question text at valid dates; the remaining 21 are absent from this resampled evaluation. Matching uses question/type/domain and full reference/evidence payloads as well as `original_id`, because source and probe identifiers changed.
- Daniel's premature audiobooks duration is retained as `u4_p202707_0104` at the July probe, after its July 1 endpoint. Priya's earlier premature reading-duration question remains in the source pool but is absent from the current sample. The timing mechanism is fixed; absence from the sample alone is not the evidence for that conclusion.
- All four previously flagged empty-pet instances are absent from both the new pool and new probes. The generator now skips unsupported empty facts instead of citing day 1 as evidence for absence. This removes the unsupported questions, without adding negative pet statements to the unchanged conversations.
- Filtering historical-fact evidence removes the old later-value-update diagnostic: zero current instances cite a later update to one of their requested values. It also creates the specific fallback citation issue below.

## Two incorrect references

| Current ID | Question location | Probe | Reference | Correct observed and synthetic set |
| --- | --- | --- | --- | --- |
| `u5_p202701_0069` | `datasets/v2/u5/probing_questions.json:2034` | 2027-01-31 | bouldering, trail running | bouldering, **piano**, trail running |
| `u5_p202701_0073` | `datasets/v2/u5/probing_questions.json:2138` | 2027-01-31 | bouldering, trail running | bouldering, **piano**, trail running |

The first is an answerable abstention control; the second is the current-hobbies question found previously. Both copy final-world hobbies into a January probe using evidence for retained values only. January transcripts report piano; its explicit removal occurs April 24. Both acceptance lists repeat the incorrect set. `manual_transcript_checks.json` preserves indexed January and April source turns. These are two question instances of the same underlying stale-viewpoint defect.

## Remaining baseline-style observation and citation findings

Seven date-change instances cite a first new-regime choice that the old rule also predicts. Six also have a rejected, pre-probe alternative boundary consistent with the declared choices and allowed causal lag. The Marcus instance has only the citation limitation: his fixed cause-lag range leaves no rejected alternative in this bounded check.

| Current ID | Question line | Accepted range | Example rejected compatible date |
| --- | ---: | --- | --- |
| `u2_p202606_0024` | u2:584 | 2026-06-29 | None established; cited gym choice is shared, June 30 tennis is uncited |
| `u4_p202701_0060` | u4:1705 | 2027-01-28 | 2027-01-25 |
| `u4_p202710_0126` | u4:3989 | 2027-10-26 | 2027-10-24 |
| `u5_p202606_0017` | u5:449 | 2026-06-18–22 | 2026-06-25, with weekly value order rotated |
| `u5_p202701_0068` | u5:2005 | 2027-01-29–31 | 2027-01-28 |
| `u5_p202702_0078` | u5:2351 | 2027-02-08 | 2027-02-09 |
| `u5_r202708_0189` | u5:3832 | 2027-02-08 | 2027-02-09 |

Locations are within the corresponding user's canonical `probing_questions.json`. These are observation/grading limitations, not incorrect hidden-world answers. Indexed first-mention source text lacks a precise dated switch declaration for these cases. The counterfactuals retain later regimes, match exception base values, and respect persona lag ranges. They are not exhaustive proofs about every natural-language sentence or simulator randomness constraint.

**New historical-fact citation regression:** `u5_p202706_0105`, `datasets/v2/u5/probing_questions.json:3239`, asks for Priya's diet on June 15, 2027 and correctly answers vegetarian at the June 30 probe. Its evidence `[1]` instead points to March 1's pescatarian statement. The vegetarian change became effective June 13 but was stated June 19; evidence filtering removes that retrospective declaration because it is later than the target. The full June 30 prefix explicitly supplies both the vegetarian value and June 13 date, so the question is answerable and its reference is correct. `supplemental_issues.json` records this citation defect; `fact_evidence_support_checks.json` checks all 150 requested fact-value memberships across 128 historical-fact questions. This is the only unsupported day-1 value fallback.

Three closure uncertainties remain: `u3_p202605_0011` (u3:280, lunch ending May 31), `u4_p202607_0020` (u4:501, dinner ending July 30), and `u5_p202609_0041` (u5:1078, physio ending September 30). Their successors are first observed in the following month. The totals match hidden truth; two also equal the elapsed duration on the probe. They are kept separate from confirmed temporal errors.

The old reading confounder citation gap and two old reading date-boundary copies are absent through resampling, not repaired. The former remains `u5_q0350`, `qa_pool.json:7169`, with only distractor day 448 cited; the latter remains `u5_q0174`, `qa_pool.json:3622`, with the same narrow accepted boundary.

## Expanded boundary review, kept separate

The baseline audit tested coincident first choices. An additional bounded scan covers all **35 non-explicit change instances** among 54 current change questions. It finds 23 candidate instances across 18 regimes. Near-boundary user text was read for every regime, with exact source lines preserved. Twenty retain credible pre-probe witnesses consistent with the recorded choices and reviewed boundary text; six overlap the baseline-style cases above. These 14 additional bounded observability findings are recorded in `broader_boundary_reviews.json`, not merged into the baseline machine issue count.

Three candidates are excluded from the credible set: Daniel's primary/retrospective illness-related music reversion has a July 2 explicit recovery statement that can constrain the condition-linked return; Priya's November bedtime has November 17 decision wording that can pin adoption to an accepted date. These remain contested diagnostics. The broader assessment does not assert that 20 synthetic references are wrong or that a paid judge would reject each alternative.

## Other unchanged limits and diagnostics

All 30 genuinely unknown abstention pool questions are still excluded by empty evidence. The 16 selected abstention questions are answerable controls; no actual never-mentioned question is selected. No current-pattern or prediction family is selected. There are still 23 full-month probes, ending January 31, 2028.

Broad citation diagnostics remain separate: 31 preference-duration instances omit the successor transition; 35 of 52 patterns omit one or more rule values in the listed citations; 12 complete-list fact rows omit relevant removals. Additional source text may establish those facts or rules. Conversation-source and copy consistency are audited by the independent source workstream.

## Reproduction and limits

Run `audit_recheck.py`, `evidence_recheck.py`, `broader_boundaries.py`, and `supplemental_checks.py` in this directory with project-local `TMPDIR` and `PYTHONDONTWRITEBYTECODE=1`. The baseline script was first imported and called without editing it. The new wrapper adapts its old fixed 511-row assertions and makes an absent rejected boundary permissible for newly sampled rows; its old hardcoded prose is disabled. `coverage.json`, `issues.json`, `observation_limits.json`, `uncertainties.json`, `summary.json`, and `input_sha256.json` retain semantic-family checks and exact locations. `baseline_finding_matches.json` and `prior_findings_current_pool.json` distinguish scheduling fixes from resampling.

The independent review also required the new wrapper to exclude hypothetical starts after the original probe before constructing historical witnesses. Priya's January workout case now records only the valid January 28 rejected witness. February 1–3 hypotheses are excluded from the regenerated findings and coverage. This evidence correction does not change the question-instance counts and makes no dataset or implementation change.

This audit checks synthetic references, scheduling, indexed citations, and specific observed counterexamples. It does not exhaustively rejudge all conversational entailment, prove unique inferred rules/causes, or report agent/LLM-judge performance.
