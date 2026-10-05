# Evidence

Initial SHA-256 comparison: the five canonical probing files and QA pools and the QA/probe generator modules changed. The five canonical world states, conversation aggregates, fidelity reports, and legacy QA sets are byte-identical to their baseline hashes. This establishes that source text problems were not repaired in those aggregates; their exposure in the new probe sample still requires checking.

Detailed workstream evidence and machine-readable results will be linked here after verification.

## Verified changes and scheduling

- `input_comparison.json` records the SHA-256 comparison for all 40 baseline inputs. Exactly 12 changed: five probing files, five canonical QA pools, and the two QA/probe generator modules. The additional baseline persona/rule fingerprints also show only these changes.
- All five original probing blobs at Git commit `98a2674` match the original audit hashes exactly. `prior_finding_matches.json` therefore matches the 71 old narrow finding instances using actual question text, type, domain, and user; shifted identifiers are not used as semantic matches. Of those instances, 35 have the same question selected now, 30 are present only in the updated pool, and six are absent from that pool. These are selection statuses, not repair verdicts.
- `structure_summary.json`: all 1,000 current questions across five users pass 13,484 structural checks. Probe generation reproduces exactly in memory from the updated canonical pools.
- `all_fields_future_dates.json` is empty: zero explicit ISO dates or named historical months after the probe occur in question text, answers, or accepted answers. `calendar_issues.json` is also empty.
- `loader_schedule.json`: the existing `setup_2/` loader reads the updated files successfully, schedules all 2,738 sessions, and produces 11,934 repeated answer instances over 24 checkpoints per user. This is an offline loader inspection, not a live evaluation.

## Concrete reference and citation evidence

- `datasets/v2/u5/probing_questions.json:2034` and `:2138`: both January 31, 2027 hobby references contain `bouldering, trail running`. World state for that date includes `bouldering, piano, trail running`; piano is removed on April 24, 2027. January 13, 22, and 26 user turns also mention playing or practicing piano. The extra selected abstention control shares the same reference omission.
- `datasets/v2/u5/probing_questions.json:3239`: the June 15, 2027 diet reference is vegetarian, but evidence is `[1]`, the initial pescatarian day. `datasets/v2/u5/sessions/2027-06-19.json:21` explicitly states that the vegetarian switch happened six days earlier, on June 13. The full June prefix supports the correct reference; the citation is incorrect. Filtering evidence to the target date discarded a valid retrospective report.

## Source and duplicate-copy evidence

- `source/summary.json` and `source/copy_comparison.json`: all 2,738 individual session files retain their old hashes; prior reviewed source-contract findings remain in 19 sessions. Thirty-eight current probes cite those sessions; this is exposure, not 38 wrong answers.
- All five duplicate `experiments/data/synthetic_v2/<user>/qa_pool.json` files are stale. The other 30 compared JSON files match their canonical counterparts. Updated `original_id` payloads mismatch those stale pools for 786 question instances; all match the canonical updated pools. `setup_2/` reads canonical conversations and probing files directly, so it does not use these stale pools.
- Coverage retains zero genuine unknown-answer abstention probes, despite 30 such pool questions. Sixteen selected abstention questions are answerable controls. Prediction and current-pattern probes remain absent; the final 28 days have no new probes. `setup_2/` still ingests that last partial month and repeats earlier questions as specified.

## Completed family and boundary reviews

- [Non-recall report](gold/report.md): all 504 instances checked; 502 references match their semantic target, and the two January hobby references do not. Every requested historical fact value has a relevant statement available by the original probe. Seven specific change citations omit a distinguishing observation; six also have pre-probe rejected-date witnesses. The separate diet citation brings specific citation limits to eight.
- [Recall report](recall/report.md): all 496 references match hidden truth; all 601 citations pass date/source metadata checks except the retained bike phase/value mismatch. All 76 distinct nonliteral direct sources were reviewed, using 37 exact-payload/hash-matched baseline reviews and 39 fresh transcript reviews. Priya's missing bike phase remains; Daniel's December 19 boundary admits a different answer while preserving all 22 observed values through the original probe. Six other challenged candidates were rejected or excluded after reviewing date/cause wording.
- [Expanded date dispositions](gold/broader_boundary_reviews.json): 35 non-explicit change instances checked; 23 candidates reviewed; 20 credible bounded limits, including the six baseline-style cases. Fourteen additional cases extend the old criterion. Three recovery/decision-linked candidates are excluded. Each counted witness is at or before the original probe and is kept separate from incorrect synthetic gold.
- [Summary](summary.json) and [full coverage](question_coverage.jsonl): 998 of 1,000 references match hidden truth. The baseline-style/supplemental union is 15 distinct instances. Including the additional boundary cases gives 29: two reference errors, 24 observation/citation limits, and three closure uncertainties. Overlapping categories are deduplicated.
- [Consolidation verification](verification.json): all current IDs covered, exact current payloads and source lines matched, counts and class overlaps reconciled, all 40 root input hashes stable, and root report links valid.
- The [independent review](review/review.md) recomputed the broader alternatives and checked raw source pointers. Its requested correction was made to the new audit adapter: starts after the original probe are excluded before recording historical rejected-date witnesses. Priya's January case now retains only January 28 in both family and consolidated findings; counts are unchanged.
