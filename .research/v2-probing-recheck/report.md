# Verification of the updated v2 datasets

The updated files are present. **The scheduling fixes work, but the previously recorded issues are not all gone.** All 1,000 current probe instances across five users pass 13,484 structural checks. No explicit historical target or reference endpoint is later than its probe. Two January hobby references are still incorrect, and observation, grading, citation, coverage, and source limits remain.

This is an offline dataset audit. No dataset, generator, agent, or runner was edited, and no model or live agent evaluation was run. The original audit is preserved in `.research/v2-probing-audit/`.

## Results by issue

| Issue | Earlier audit | Updated files | Disposition |
| --- | ---: | ---: | --- |
| Explicit historical targets after the probe | 51 instances | 0 | Fixed in the current probes; the allocator checks dates and named months across question/reference/acceptance text. |
| Completed duration references ending after the probe | 2 | 0 | Fixed timing. Daniel's retained question moved to July; Priya's question remains in the pool but is not sampled. |
| Unsupported empty-pet questions | 4 | 0 | Removed from both current pools and probes by skipping unsupported empty facts. Source conversations were not supplemented with negative statements. |
| Incorrect current hobby references | 1 | 2 | Same stale-viewpoint defect remains; the new sample also selects the corresponding answerable control. |
| Specific recall answerability limits | 3 | 2 | Priya's missing workout phase remains; the two old ambiguous targets were excluded. A different December boundary limit was detected by the expanded review. |
| Accepted change dates excluding compatible alternatives | 5 narrow cases | 6 narrow cases; 20 including expanded checks | Remain. The broader review adds 14 cases; it does not establish 20 wrong synthetic dates. |
| Specific citation gaps | 7 | 8 | Seven change-citation gaps and one new historical-diet fallback citation. Six overlap the narrow date limits. |
| Completed-versus-elapsed duration uncertainties | 3 | 3 | Remain as uncertainty, with one retained and two newly sampled cases. |
| Actual unknown-answer abstention coverage | 0 selected out of 30 pool questions | 0 out of 30 | Still absent. All 16 selected abstention instances are answerable controls. |
| Confirmed conversation-source contract sessions | 19 | 19 | Same text and hashes. Thirty-eight current questions cite these sessions; that is exposure, not 38 incorrect answers. |
| Canonical/experiment JSON copy mismatches | 0 of 35 pairs | 5 of 35 | New stale experiment QA pools; canonical inputs remain consistent. |

The sampling and source identifiers changed. Before/after counts therefore do not describe a fixed set of identical questions. Expanded date-boundary checks also cover more cases than the original first-choice criterion. [Prior finding matches](prior_finding_matches.json) account for every original narrow finding using full question payloads rather than assuming identifiers are stable.

## Concrete remaining and new cases

**Incorrect January hobby references:** [`u5_p202701_0069`](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:2034) and [`u5_p202701_0073`](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:2138) both give `bouldering, trail running` on January 31, 2027. The correct observed and synthetic set includes **piano**. January transcripts mention playing/practicing it, and the removal occurs on April 24. The first is an answerable abstention control; the second is the previously identified current-hobbies problem. Both acceptance lists omit piano too.

**Unsupported workout phase:** [`u5_p202605_0008`](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:215) still asks for Priya's May 9 bike workout. Its only cited day, May 7, shows bouldering. None of the relevant initial-regime bike Saturdays is observed. Removing later citations did not create the missing information.

**Ambiguous recall boundary:** [`u4_p202712_0152`](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:4820) expects jazz records for Daniel's December 19 work music. Starting the next uncaused rotation on December 19 instead of the hidden December 20 start preserves all 22 recorded work-music choices through the original probe but changes that day's answer to ambient electronic. This is newly detected, not proved newly introduced by the update. The six other challenged recall candidates were rejected or excluded after checking actual date/cause language. [Recall evidence and limits](recall/report.md).

**Overly narrow accepted change dates:** for example, [`u5_p202702_0078`](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:2351) accepts February 8 only, while February 9 also preserves the reviewed choices and allowed lag. Six baseline-style instances retain concrete pre-probe counterexamples. Expanding the check to all 35 non-explicit change instances finds 20 credible bounded limits, including those six; three contested candidates are excluded. The broader findings remain separately identified because they are not incorrect hidden-world gold answers or actual judge verdicts. [Boundary dispositions](gold/broader_boundary_reviews.json), [non-recall report](gold/report.md).

**New diet citation regression:** [`u5_p202706_0105`](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:3239) correctly answers vegetarian for June 15, but cites initial day 1, when Priya was pescatarian. The June 19 user report explicitly dates the switch to June 13 and is available at the June 30 probe. Filtering evidence to the historical target date removed this valid retrospective source. The full prefix still supports the answer. [Exact evidence](gold/supplemental_issues.json).

## Counts and their limits

All 496 recall and 504 other references were checked against their semantic targets. **998 match hidden synthetic truth; two hobby reference lists do not.** Truth agreement alone does not establish observational answerability.

Using the baseline-style checks plus the new specific citation and recall findings gives **15 distinct current question instances**: two reference errors, ten additional observation/citation limits, and three uncertainties. The expanded boundary review adds 14 distinct instances, producing **29** in the expanded union: two reference errors, 24 observation/citation limits, and three uncertainties. Overlapping categories are deduplicated. These totals are not counts of wrong answers from the agent. [Summary](summary.json), [per-question findings](question_findings.md), [all 1,000 coverage records](question_coverage.jsonl).

Broad citation diagnostics are separate: 31 preference-duration instances lack a closing-transition citation, 35 of 52 patterns omit a rule value in their listed citations, and 12 complete-list fact rows omit relevant removals. Additional observations elsewhere may support the content. The old reading-confounder citation gap remains in the pool but is absent through resampling; its omission is not a repair. Old later-value-update citations are gone, although the diet fallback above is a new consequence of the historical-date filter.

The review accepts normal inference from stable patterns and missing target sessions. It checks every record mechanically/by semantic family, then reviews concrete counterexamples and relevant text. It does not prove exhaustive natural-language entailment or uniqueness of all inferred causes/rules, and no LLM judge verdict was produced.

## Sources, copies, and monthly setup

All 2,738 individual sessions retain their baseline hashes and match aggregates/world sampling. All five worlds regenerate exactly; all 15,330 preference cells and 516 exception cells pass native checks; saved extraction verdicts reproduce for all sessions. These checks do not erase the retained 12 avoidable topic mentions, three conflicting writer obligations, or four unmodeled details in source text. [Source report](source/report.md).

The five `experiments/data/synthetic_v2/<user>/qa_pool.json` files are stale. Their old source IDs/payloads disagree with 786 updated probe instances; all 1,000 instances match the updated canonical pools. This is duplicate-copy provenance drift, not 786 wrong questions. `setup_2/` reads canonical conversations and probing files directly, so those duplicate pools are unused there.

There are still zero prediction/current-pattern probes and no new probes in the last 28-day partial month. The existing `setup_2/` loader successfully ingests that final month and repeats earlier questions as agreed: all 2,738 sessions over 24 checkpoints per user, with **11,934 scheduled answer instances**. This number is a schedule, not a completed model evaluation. [Loader schedule](loader_schedule.json).

## Reproduction

Use project-local `TMPDIR` and `PYTHONDONTWRITEBYTECODE=1`. `check_updates.py` reruns structural/calendar/provenance checks without overwriting the old audit. The workstream scripts in `gold/`, `recall/`, and `source/` preserve primary evidence and manual dispositions separately. `aggregate_updates.py` checks full coverage, consolidates findings, and verifies the 40 root input hashes were unchanged during inspection. [Evidence](evidence.md), [sources](sources.md), [input stability](input_stability.json).

A fresh [independent review](review/review.md) verifies counts, monthly cutoffs, retained source hashes, exact current source pointers, and the counterexamples against date/cause constraints. Its mechanical checks are recorded separately in [review verification](review/verification.json). The interpretation continues to distinguish bounded conversational limits from incorrect references and from actual judge outcomes.
