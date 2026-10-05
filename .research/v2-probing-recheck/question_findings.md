# Current question findings

All locations refer to the updated canonical probing files. Repeated evaluation attempts are not counted separately here; dataset retrospective records retain their own IDs.

## Incorrect references

| Current question | Original probe | Finding |
| --- | --- | --- |
| [u5_p202701_0069](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:2034) | 2027-01-31 | accepted_answer_mismatch_at_probe, gold_answer_mismatch_at_probe |
| [u5_p202701_0073](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:2138) | 2027-01-31 | accepted_answer_mismatch_at_probe, gold_answer_mismatch_at_probe |

Both references omit the still-active piano hobby at January 31, 2027. Their answer and acceptance errors are counted once per question.

## Specific observation and citation limits

| Current question | Original probe | Finding |
| --- | --- | --- |
| [u2_p202606_0024](/home/cloaked/projects/LifelongAgent/datasets/v2/u2/probing_questions.json:584) | 2026-06-30 | change_citations_do_not_observe_change |
| [u4_p202701_0060](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:1705) | 2027-01-31 | change_citations_do_not_observe_change, non_unique_change_date_rejected_by_grader |
| [u4_p202710_0126](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:3989) | 2027-10-31 | change_citations_do_not_observe_change, non_unique_change_date_rejected_by_grader |
| [u4_p202712_0152](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:4820) | 2027-12-31 | unobserved_shift_boundary |
| [u5_p202605_0008](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:215) | 2026-05-31 | unobserved_answer_phase |
| [u5_p202606_0017](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:449) | 2026-06-30 | change_citations_do_not_observe_change, non_unique_change_date_rejected_by_grader |
| [u5_p202701_0068](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:2005) | 2027-01-31 | change_citations_do_not_observe_change, non_unique_change_date_rejected_by_grader |
| [u5_p202702_0078](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:2351) | 2027-02-28 | change_citations_do_not_observe_change, non_unique_change_date_rejected_by_grader |
| [u5_p202706_0105](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:3239) | 2027-06-30 | fact_fallback_citation_does_not_support_value |
| [u5_r202708_0189](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:3832) | 2027-08-31 | change_citations_do_not_observe_change, non_unique_change_date_rejected_by_grader |

Six change-date rows also have citation limits; that overlap is counted once. Priya's diet reference is correct and supported elsewhere in the original prefix; its listed citation is wrong. Recall limits agree with hidden truth but isolate missing/ambiguous observations.

## Duration uncertainties

| Current question | Original probe | Finding |
| --- | --- | --- |
| [u3_p202605_0011](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/probing_questions.json:280) | 2026-05-31 | duration_completion_not_observed |
| [u4_p202607_0020](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:501) | 2026-07-31 | duration_completion_not_observed |
| [u5_p202609_0041](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:1078) | 2026-09-30 | duration_completion_not_observed |

The completed framing is not established by the available closure observations. These are uncertainties, not confirmed wrong totals.

## Additional expanded date-boundary findings

| Current question | Original probe | Finding |
| --- | --- | --- |
| [u3_p202606_0022](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/probing_questions.json:555) | 2026-06-30 | non_unique_change_date_rejected_by_grader |
| [u3_p202712_0144](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/probing_questions.json:4661) | 2027-12-31 | non_unique_change_date_rejected_by_grader |
| [u3_r202610_0166](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/probing_questions.json:1531) | 2026-10-31 | non_unique_change_date_rejected_by_grader |
| [u4_p202706_0096](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:2769) | 2027-06-30 | non_unique_change_date_rejected_by_grader |
| [u4_r202708_0197](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:3461) | 2027-08-31 | non_unique_change_date_rejected_by_grader |
| [u5_p202606_0016](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:423) | 2026-06-30 | non_unique_change_date_rejected_by_grader |
| [u5_p202608_0028](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:745) | 2026-08-31 | non_unique_change_date_rejected_by_grader |
| [u5_p202609_0043](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:1127) | 2026-09-30 | non_unique_change_date_rejected_by_grader |
| [u5_p202704_0087](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:2648) | 2027-04-30 | non_unique_change_date_rejected_by_grader |
| [u5_p202705_0093](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:2866) | 2027-05-31 | non_unique_change_date_rejected_by_grader |
| [u5_p202708_0114](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:3618) | 2027-08-31 | non_unique_change_date_rejected_by_grader |
| [u5_p202708_0117](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:3698) | 2027-08-31 | non_unique_change_date_rejected_by_grader |
| [u5_p202709_0122](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:3921) | 2027-09-30 | non_unique_change_date_rejected_by_grader |
| [u5_r202611_0159](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:1524) | 2026-11-30 | non_unique_change_date_rejected_by_grader |

These extend the original first-choice check. The 20 credible bounded date findings include six rows already listed above and the 14 additional rows here. Three contested candidates are excluded. Detailed alternative dates, source excerpts, constraints, and interpretation limits are in gold/broader_boundary_reviews.json.
