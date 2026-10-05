# Evidence ledger

## Structural and explicit-date audit

**Established:** all 1,000 records reproduce exactly through the read-only `build_probing_questions()` function. Required fields, identities, source mappings, per-probe counts and dates, evidence-day ranges, and retrospective links pass the recorded structural checks.

Sources: `structure.py`, `structure_summary.json`, and `structure_checks.json`. Reproduction confirms the existing generator output; it does not establish semantic correctness at a changed viewpoint.

**Established:** 51 question instances ask about an explicit date/month later than the probe cutoff: 42 `pattern_at_time`, five `event_status`, four `recall`. Every instance and its exact source question line is recorded in `calendar_issues.json`.

**Established coverage limitation:** 30 never-mentioned abstention source questions (six per user) have empty evidence and are excluded by `generator_v2/probing.py:83`. The 13 selected `abstention` instances are answerable controls; there are zero never-mentioned abstention instances. Prediction and current-pattern types are also absent. Sources: `selection.json`, canonical QA pools, and `generator_v2/qa.py` abstention construction.

**Established coverage limitation:** each history ends 2028-02-28, while the last full-month probe is 2028-01-31. The final 28 days have no probe. Source: `selection.json` and the month-boundary logic in `generator_v2/probing.py:29`.

## Gold and implicit-timing audit

**Established:** all 511 non-recall instances were checked against synthetic truth by family and semantic target. There is one wrong current reference (`u5_p202701_0076`), which also has a wrong acceptance list: piano is omitted before its actual removal. The other 510 references match hidden truth. Evidence: `gold/coverage.json`, `gold/issues.json`, daily facts, and January user reports.

**Established:** two duration golds contain completed endpoints after the probe (`u4_p202706_0100`, `u5_p202608_0032`). This adds two timing defects beyond the explicit-date scan. The reference number is eventually correct. Approximately 10% judge tolerance prevents inferring that every reasonable elapsed answer would fail.

**Established observation limit:** five non-explicit change instances admit rejected, declared-choice-consistent boundary alternatives. Persona cause lags and exception bases were checked; the reading case also rotates the two weekly values. These overlap the five change-citation gaps, not five new hidden-world mismatches. Sources: `gold/observation_limits.json`, `gold/manual_evidence.md`, and date grading rules.

**Established observation limit:** four empty-pet references are synthetically correct but lack a negative/first-pet statement in available text. Day-1 evidence is a fallback. Silence does not establish an empty fact set without an added assumption. Evidence: focused full-prefix vocabulary and raw-session reviews in `gold/manual_evidence.md`.

**Established citation limitation:** two distractor explanations cite no observation of the nearby change they describe. That observation exists elsewhere in the prefix; this does not establish a wrong answer. Forty duration closure citations, 36 pattern value citations, eight omitted fact removals, and 22 historical fact citations of later value updates (including three later piano-removal citations) are broader diagnostics, not additional narrow error counts.

**Uncertain:** three completed-duration questions have no observed closing transition by the probe. Their synthetic arithmetic is correct. Sources: `gold/uncertainties.json` and subsequent first-mention dates.

## Recall and observation audit

**Established:** all 489 recall reference/accepted values match target-day hidden truth. All 707 citation references exist, agree across aggregate/individual sessions, and match the saved extraction. Seventy-three distinct nonliteral direct sources representing 94 instances received manual transcript review by the audit agent. Saved extraction remains source evidence, not fresh semantic validation.

**Established observation defects:** `u5_p202605_0011` expects a workout phase with zero observations; incidental bicycle language describes commuting or somebody else's exercise. `u3_p202711_0137` and `u5_p202610_0047` admit alternative hidden change dates preserving all 165 and 42 observed domain choices while changing the requested target answer. These are additional answerability limitations, not wrong synthetic gold. Full prefix evidence and counterfactuals: `recall/phase_transcript_review.json` and `boundary_transcript_reviews.json`.

**Overlap:** four future recall targets are already included in the 51 root calendar findings.

## Conversation-source and execution audit

**Established:** all 2,738 aggregate sessions match individual files and sampled world metadata; all 35 canonical/experiment JSON pairs match. All five worlds reproduce in memory, native checks pass, and all 15,330 preference cells plus 516 exceptions agree with resolved rules/options. Saved-validator replay is exact for all sessions; it is not a new extraction. u1–u4 usage logs are LFS pointers; u5's actual usage log reconciles with fidelity.

**Established source-contract findings:** among 28 retained failures, raw review confirms 12 avoidable absent-topic mentions, three required-versus-absent topic conflicts, and four unmodeled persistent details. Nineteen sessions are involved, cited by 33 probes. Seven saved failures are refuted by source support/correct relative dates; two more contain event content with underspecified dates. All 54 probes exposed to any saved flag are mapped. These figures establish zero additional wrong question references. Evidence: `source/report.md`, `flag_reviews.json`, `flagged-session-index.md`, and `verification.json`.

**Deferred integration requirements:** the existing loader reads 250 legacy questions rather than monthly probes; the current runner ingests all sessions before answering. A stub reproduction demonstrates potential monthly-cutoff leakage if probes are manually supplied. It is not an actual agent/LLM evaluation or evidence of a completed contaminated monthly run. The user explicitly deferred loader/reader setup; no change was made.

## Consolidation

`summary.json`, `findings.json`, and `question_coverage.jsonl` reconcile all 1,000 records: 54 distinct scheduling/reference errors; 12 observation limitations; seven specific citation limitations overlapping five of those observation cases, for 14 additional distinct limited questions; and three uncertainty cases. The narrow union is 71 instances. The full explanations and exact ID/location list are `report.md` and `question_findings.md`. Source exposure, broad citation diagnostics, selection gaps, and integration are outside that union.
