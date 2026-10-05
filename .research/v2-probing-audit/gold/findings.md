# Non-recall gold and implicit-time audit

This is an offline audit of all 511 non-recall records in the 1,000 canonical monthly probing questions. All records were mapped to a semantic family and checked against the daily synthetic world at the appropriate target and probe viewpoint. Pool equality is a provenance check, not the truth oracle. No datasets, generator, agents, or evaluation implementation were edited.

## Results

- 510/511 gold references match the synthetic truth at their semantic target; 1 current reference is wrong at its actual probe.
- One acceptance list repeats that wrong current reference. All 511 answer-type/grading combinations match their semantic family/cardinality.
- 47 non-recall questions have explicit future targets: 42 patterns and 5 event-status queries. These overlap the root scheduling audit and must not be added as disjoint findings.
- Two additional questions have an unreached completed endpoint only in the gold answer. Their eventual synthetic durations are correct, but they are not completed historical totals at the primary probe.
- Three duration rows have an unobserved closing transition: two end on the probe itself and one ends a day earlier but is first observed in the next month. They are recorded as closure/wording uncertainties.
- Four empty-pet answers require an unstated closed-world convention; they are observation/answerability limits. Specific citation gaps affect 5 change-boundary rows and 2 confounder explanations. None is counted as wrong synthetic gold.
- Those same 5 change-boundary rows also admit declared-choice-consistent alternative start dates rejected by date_exact. This is an overlapping observability/grading limit, not five additional hidden-world errors.
- 40 preference-duration records cite first/last old-routine observations but omit the closing transition. This is a diagnostic of citation incompleteness, not a claim that the full conversation prefix is missing the transition.
- All 30 genuinely unknown abstention pool questions are excluded because they have empty evidence; the probes retain 13 answerable controls and zero unknown abstention cases. This is a selection gap, not erroneous gold on those 13 controls.

| Family | Probing rows | Family-specific truth check |
| --- | ---: | --- |
| abstention | 13 | Answerable controls rechecked at actual probe; exclusion of unknown cases |
| fact_current | 16 | Complete daily fact set at the actual probe, including retained values |
| fact_at_time | 132 | Daily active fact list on the explicit target; cardinality and empty sets |
| event_status | 31 | Effective state at target/viewpoint, separate planned/happened/cancelled status |
| attribution | 57 | Cause reference, uncaused semantics, first observation and statement date |
| pattern_at_time | 59 | Entire named month/point inside the target regime; full rule and correct anchor |
| change_detection | 52 | Effective start, first mention, accepted-date bounds and distinguishability |
| fact_history | 11 | Maximal value intervals recomputed using only the prefix; 'now' at actual probe |
| duration | 41 | Inclusive calendar arithmetic, world-derived intervals and observed closure |
| exception_vs_shift | 53 | Exception value/reason versus previous rule; genuine regime shifts |
| distractor_probe | 23 | Distractor versus real cause; near confounder change and availability |
| reversion | 12 | Inclusive temporary period, actual return and resumed base anchor; current 'still' |
| other_person | 11 | Person/fact identity and all personal yes/no controls at actual probe |

No `pattern_current` or `prediction` items occur in these probes. Historical pattern copies remain anchored to their explicit target; none of the selected current fact/history/other-person/abstention/reversion copies produced a new retrospective-only stale reference.

## Confirmed reference mismatch

### u5_p202701_0076

Source: `datasets/v2/u5/probing_questions.json:2198`; gold: `datasets/v2/u5/probing_questions.json:2199`; accept: `datasets/v2/u5/probing_questions.json:2201`.
Question: What are Priya's current hobbies?
Probe: 2027-01-31. Observed gold: `bouldering, trail running`. Expected: `bouldering, piano, trail running`.
Piano is still an active and known hobby on the January probe. It is removed only on day 420 (2027-04-24). The final-world current-hobby generator keeps bouldering/trail running, and the probing allocator moves that unchanged final answer to January using only the remaining hobbies' evidence [1, 331]. The full January transcript also retains piano, including the January 13 user report of playing and January 22 intention to practice.
Authoritative daily truth: `datasets/v2/u5/world_state.json:42526`, `generator_v2/qa.py:370`, `generator_v2/probing.py:145`

## Future completed endpoints in the gold

These are additional to the 51 explicitly future-dated questions handled by the root audit.

| ID | Question source | Probe | Gold completed endpoint/total | Prefix truth |
| --- | --- | --- | --- | --- |
| u4_p202706_0100 | `datasets/v2/u4/probing_questions.json:3076` | 2027-06-30 | 18 days (2027-06-14 to 2027-07-01) | Still ongoing; 17 days elapsed; final total not yet established as completed |
| u5_p202608_0032 | `datasets/v2/u5/probing_questions.json:844` | 2026-08-31 | 65 days (2026-07-03 to 2026-09-05) | Still ongoing; 60 days elapsed; final total not yet established as completed |

## Specific evidence gaps

Gold truth and conversation observations are deliberately separated here. These findings show what cited metadata does not establish; they do not invalidate a reference that agrees with the synthetic world.

| ID | Category | Question source | Observation |
| --- | --- | --- | --- |
| u4_p202710_0129 | change_citations_do_not_observe_change | `datasets/v2/u4/probing_questions.json:4043` | The first new-regime mention has the same value the old rule predicts, and the writer is instructed not to announce the change. The cited first mention alone cannot justify 'first observed' as a change boundary. Later prefix observations can support a rule inference; the gold effective start still matches synthetic truth. |
| u5_p202609_0043 | change_citations_do_not_observe_change | `datasets/v2/u5/probing_questions.json:1147` | The first new-regime mention has the same value the old rule predicts, and the writer is instructed not to announce the change. The cited first mention alone cannot justify 'first observed' as a change boundary. Later prefix observations can support a rule inference; the gold effective start still matches synthetic truth. |
| u5_r202611_0189 | change_citations_do_not_observe_change | `datasets/v2/u5/probing_questions.json:1685` | The first new-regime mention has the same value the old rule predicts, and the writer is instructed not to announce the change. The cited first mention alone cannot justify 'first observed' as a change boundary. Later prefix observations can support a rule inference; the gold effective start still matches synthetic truth. |
| u5_p202702_0080 | change_citations_do_not_observe_change | `datasets/v2/u5/probing_questions.json:2437` | The first new-regime mention has the same value the old rule predicts, and the writer is instructed not to announce the change. The cited first mention alone cannot justify 'first observed' as a change boundary. Later prefix observations can support a rule inference; the gold effective start still matches synthetic truth. |
| u5_p202705_0094 | confounder_change_not_cited | `datasets/v2/u5/probing_questions.json:2973` | Gold's explanation correctly describes a nearby change, but the evidence list contains only the distractor statement. The actual change is available elsewhere in the prefix. |
| u5_r202708_0197 | change_citations_do_not_observe_change | `datasets/v2/u5/probing_questions.json:3847` | The first new-regime mention has the same value the old rule predicts, and the writer is instructed not to announce the change. The cited first mention alone cannot justify 'first observed' as a change boundary. Later prefix observations can support a rule inference; the gold effective start still matches synthetic truth. |
| u5_r202710_0193 | confounder_change_not_cited | `datasets/v2/u5/probing_questions.json:4492` | Gold's explanation correctly describes a nearby change, but the evidence list contains only the distractor statement. The actual change is available elsewhere in the prefix. |
| u2_p202603_0001 | empty_fact_has_no_negative_statement | `datasets/v2/u2/probing_questions.json:21` | The empty set is correct synthetic truth, but evidence [1] is a fallback. No background/fact-change statement asserts the empty set, so absence of mention is not observational proof of no pets. |
| u2_p202608_0038 | empty_fact_has_no_negative_statement | `datasets/v2/u2/probing_questions.json:1002` | The empty set is correct synthetic truth, but evidence [1] is a fallback. No background/fact-change statement asserts the empty set, so absence of mention is not observational proof of no pets. |
| u2_r202702_0188 | empty_fact_has_no_negative_statement | `datasets/v2/u2/probing_questions.json:2501` | The empty set is correct synthetic truth, but evidence [1] is a fallback. No background/fact-change statement asserts the empty set, so absence of mention is not observational proof of no pets. |
| u3_p202604_0007 | empty_fact_has_no_negative_statement | `datasets/v2/u3/probing_questions.json:178` | The empty set is correct synthetic truth, but evidence [1] is a fallback. No background/fact-change statement asserts the empty set, so absence of mention is not observational proof of no pets. |

The three primary empty-pet queries have no user pet vocabulary anywhere in their available prefixes (u2 through August 2026, u3 through April 2026). The u2 retrospective empty-pet query has subsequently learned the January 2027 adoption, but neither its day-1 citation nor the recorded fact statements establish that this was the person's first pet. Empty-list gold therefore depends on treating unmentioned synthetic facts as absent.

For the five non-explicit change rows above, the first mention is compatible with the old rule. The first differing values occur later: u4 music on day 606, u5 reading on day 197, and u5 music on day 351. Explicitly announced shifts with unchanged daily values are not flagged: their declaration can supply the missing boundary evidence.

## Date grading observability limits

These overlap the five change-citation gaps above. Every alternative respects the declared pre-probe choice values, exception bases, and the persona's cause-lag range. Subsequent regimes remain fixed. Actual first-mention transcripts contain daily choices without a dated switch announcement.

| ID | Question source | Accepted dates | Rejected compatible boundaries |
| --- | --- | --- | --- |
| u4_p202710_0129 | `datasets/v2/u4/probing_questions.json:4043` | 2027-10-26 | 2027-10-24 (same_rule); 2027-10-25 (same_rule) |
| u5_p202609_0043 | `datasets/v2/u5/probing_questions.json:1147` | 2026-09-06, 2026-09-07, 2026-09-08 | 2026-09-13 (rotated_weekly_value_order) |
| u5_r202611_0189 | `datasets/v2/u5/probing_questions.json:1685` | 2026-09-06, 2026-09-07, 2026-09-08 | 2026-09-13 (rotated_weekly_value_order) |
| u5_p202702_0080 | `datasets/v2/u5/probing_questions.json:2437` | 2027-02-08 | 2027-02-09 (same_rule); 2027-02-10 (same_rule); 2027-02-11 (same_rule); 2027-02-12 (same_rule); 2027-02-13 (same_rule) |
| u5_r202708_0197 | `datasets/v2/u5/probing_questions.json:3847` | 2027-02-08 | 2027-02-09 (same_rule); 2027-02-10 (same_rule); 2027-02-11 (same_rule); 2027-02-12 (same_rule); 2027-02-13 (same_rule) |

For u4 music, moving the same day-of-week rule to days 603/604 keeps cause lags 3/4 within the U4 1–5 range. Day 606 would also fit the observations but exceeds that lag range, so it is excluded from the recorded admissible alternatives. For u5 music, days 346–350 remain within U5's 1–10 range; day 351 would fit the choices but has an excessive lag of 11. For u5 reading, fixed ordered weekly values constrain the original anchor; reversing the two values and beginning on the first poetry day (197) yields the same continuing cycle without exposing any earlier behavioral difference. The question supplies neither cycle order nor anchor. These are bounded counterexamples, not a census of every compatible alternative world.

## Timing-specific citation diagnostics

Eight current/at-time fact rows omit an earlier removal relevant to establishing a complete list. Twenty-two at-time fact rows cite a later update to the same value; three of these hobby rows cite the much later piano removal (day 420) for targets in October/December 2026. Those updates are known by the actual probes, so this is not future leakage; it is temporally unsuitable direct support for the earlier membership and can delay question selection. The exact IDs, statements, and source lines are in evidence_diagnostics.json.

## Closure uncertainty

| ID | Question source | Probe | Synthetic interval days | First next-regime mention |
| --- | --- | --- | ---: | --- |
| u3_p202711_0136 | `datasets/v2/u3/probing_questions.json:4372` | 2027-11-30 | 240 | 2027-12-01 |
| u4_p202606_0016 | `datasets/v2/u4/probing_questions.json:407` | 2026-06-30 | 122 | 2026-07-01 |
| u4_p202607_0019 | `datasets/v2/u4/probing_questions.json:493` | 2026-07-31 | 152 | 2026-08-01 |

## Counter-evidence and limits

- All 11 fact-history answers were recomputed from the prefix and remain valid; no future completed fact change or stale 'now' interval was found.
- All 31 event references match their intended state and effective date. Plans for later months are legitimate recorded knowledge. Five historical status queries nevertheless target dates beyond their probe; knowing a current plan does not establish status at a later date.
- All selected exception, reversion, other-person, and answerable abstention controls match their world truth at the probe. Intended causal inference is not flagged merely because a cause link was implicit.
- Free-text `accept` hints can be incomplete labels (especially history); they are not treated as strict alternative answers. The evaluator's judge uses the complete reference, while `date_exact` uses its accepted dates.
- 36 of 59 pattern citations omit one or more values in their full rule. This is an evidence diagnostic, not an error count: additional prefix observations and intended inference can supply them.
- Rule descriptions, per-day truth, statement dates, and metadata were checked reproducibly for every non-recall row. Exhaustive entailment of every natural-language paraphrase and formal uniqueness of every inferred routine/cause remain outside this script's proof. Saved validator flags were not assumed correct.

## Reproduction and artifacts

Run `PYTHONDONTWRITEBYTECODE=1 python .research/v2-probing-audit/gold/audit_gold.py`.

- `coverage.json`: 511 per-question records with target/viewpoint truth, exact source lines, checks, caveats, and verdicts.
- `issues.json`: per-ID reference, temporal eligibility, and specific evidence findings; categories can overlap.
- `uncertainties.json`: completed-versus-elapsed wording cases.
- `observation_limits.json`: four empty-set answerability limits and five bounded counterfactual date-grading limits, the latter overlapping the corresponding citation findings.
- `evidence_diagnostics.json`: missing duration closing citations, fact removal/timing diagnostics, and pattern value coverage.
- `selection_gaps.json`: omitted unknown abstention pool IDs and exact locations.
- `summary.json`: counts and ID unions by finding kind; future-date scan across question, gold, and acceptance.
- `input_sha256.json`: hashes of audited canonical inputs and authoritative generator/evaluator sources.
- `sources.md` and `manual_evidence.md`: source-role inventory and the narrow transcript checks kept separate from interpretation.
