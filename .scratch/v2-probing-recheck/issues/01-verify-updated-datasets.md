# Verify the updated v2 datasets

Type: research
Status: resolved

Compare input hashes and question payloads to the prior audit; rerun relevant offline checks without overwriting prior evidence. Account for every current question and report concrete remaining examples and limits.

## Comments

The first comparison confirms that all five probing files and QA pools changed, along with `generator_v2/qa.py` and `generator_v2/probing.py`. Canonical worlds, aggregate conversations, fidelity reports, and legacy QA files match the prior recorded hashes.

## Answer

Verified all 1,000 updated probes and all 2,738 individual conversation files. The scheduling fixes work: zero explicit future targets/endpoints remain. Unsupported empty-pet questions were removed. Two January hobby references remain incorrect; observation, citation, date-grading, duration uncertainty, unknown-answer coverage, and retained source-contract limits remain. Five experiment QA pools are stale, while canonical inputs and `setup_2/` are consistent.

The baseline-style/supplemental current union is 15 distinct instances. A separately recorded broader date-boundary review adds 14 instances: 29 total, comprising two reference errors, 24 observation/citation limits, and three uncertainties. A fresh review independently checked the counts, bounded alternatives, and raw source pointers; its evidence-cutoff correction was applied only to new audit artifacts. No dataset, generator, agent, runner, or original audit was edited, and no model/API call was made.

Context: [full report](../../../.research/v2-probing-recheck/report.md), [current locations](../../../.research/v2-probing-recheck/question_findings.md), [summary](../../../.research/v2-probing-recheck/summary.json), and [fresh review](../../../.research/v2-probing-recheck/review/review.md).
