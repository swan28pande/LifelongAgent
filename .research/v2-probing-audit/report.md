# Audit of all 1,000 v2 probing questions

The earlier 51 explicit future-date findings were not the full set of issues. The audit confirms **54 distinct scheduling/reference errors**, **14 additional questions with observation or citation limitations**, and **3 duration-closure uncertainties**. These groups contain 71 distinct question instances. Observation limitations include five cases where accepted change dates exclude alternatives consistent with recorded choices. They are separated from incorrect synthetic references.

All five users and every question instance were checked. No dataset, generator, agent, loader, or evaluation implementation was changed. No model or paid API was used. Audit records are confined to `.research/v2-probing-audit/` and the local research issue. Reader/loader implementation is deferred as requested.

## Question findings

| Finding | Instances | What the evidence establishes |
| --- | ---: | --- |
| Explicit targets later than their probe | 51 | 42 historical patterns, 5 event-status queries, 4 recalls are placed before the date/month they ask about. |
| Future completed endpoints hidden in duration answers | 2 | The reference describes a completed interval whose end is after the probe. |
| Incorrect current-fact reference and acceptance | 1 | Priya's January 2027 hobby list prematurely omits piano. The reference and acceptance defect are one question. |
| Accepted change dates exclude compatible alternatives | 5 | Other boundaries preserve all declared pre-probe choices but fail the accepted-date check. These also have incomplete change citations. |
| Recall values unsupported or ambiguous in the conversation | 3 | One answer phase was never observed; two exact target answers differ under compatible unobserved boundaries. |
| Empty-pet answers rely on an unstated absence | 4 | The synthetic empty set is correct, but no negative or first-pet statement establishes it in the available text. |
| Explanations cite only the distractor, omitting the nearby change | 2 | Full-prefix observations can support the explanation; its listed evidence does not include the described transition. |
| Duration completion not observed by the probe | 3 | Hidden interval arithmetic is correct, but whether it had ended is not established by the recorded prefix. Kept as uncertainty. |

The table counts question instances, including retrospective copies. The five change-date cases are listed once in this table despite having both grading/observation and citation findings. The 51 explicit future targets include the four recall findings from the recall sub-audit and the 47 non-recall findings from the gold sub-audit; those are not extra cases. The complete ID/location list is [question_findings.md](question_findings.md); observed values, expected behavior, and source evidence are in [findings.json](findings.json).

## Additional cases beyond the original 51

**Incorrect current hobbies — `u5_p202701_0076`.** The [January 31, 2027 question](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:2198) expects `bouldering, trail running`. The correct list then is `bouldering, piano, trail running`. Piano is removed on April 24, 2027, not in January. Both the daily active/known facts and January user reports retain piano. The generator builds the original answer from the final world, then schedules it using evidence for the retained hobbies without recomputing the answer at the new viewpoint. [Gold evidence](gold/findings.md).

**Future completed durations — two cases.** [Daniel's June 30, 2027 probe](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:3076), `u4_p202706_0100`, expects 18 days ending July 1; only 17 days have elapsed. [Priya's August 31, 2026 probe](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:844), `u5_p202608_0032`, expects 65 days ending September 5; only 60 have elapsed. Both reference totals are correct eventually, but have not become completed historical totals at those probes. The judge allows approximately 10% error for duration numbers, so these defects do **not** establish that every reasonable elapsed-duration answer would be marked wrong. [Gold evidence](gold/findings.md), [judge rules](/home/cloaked/projects/LifelongAgent/experiments/core/grading.py:22).

**Missing workout phase — `u5_p202605_0011`.** The [May 9, 2026 recall](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:284) expects `bike ride`. The initial exercise regime contains zero observations of the relevant Saturday phase. Its three cited days instead show bouldering, a long run, and swimming. Incidental bicycle text describes Priya's commute or somebody else's workout. This is an observed-answerability defect despite correct hidden-world gold. [Full phase review](recall/phase_transcript_review.json).

**Two unresolved recall boundaries.** For [Sofia's October 24, 2027 podcast](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/probing_questions.json:4397), `u3_p202711_0137`, moving the change one day later preserves all 165 recorded domain choices through the probe but changes the target answer from comedy to Italian lessons. For [Priya's September 17, 2026 work music](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:1317), `u5_p202610_0047`, compatible later changes preserve all 42 recorded domain choices but change the target answer from synthwave to film scores. The conversations do not date the switches precisely. These are bounded counterexamples, not assertions that every inference question is invalid. [Boundary review](recall/boundary_transcript_reviews.json).

**Five change-date grading limits.** These are `u4_p202710_0129`, `u5_p202609_0043`, `u5_r202611_0189`, `u5_p202702_0080`, and `u5_r202708_0197`. For example, Priya's February music change accepts February 8 only, although February 9–13 preserve the observed choices and stay within the persona's allowed causal lag. Daniel's music also permits earlier compatible dates; Priya's reading permits a later start with the weekly values rotated. The question supplies neither hidden cycle order nor anchor. Exact alternatives, declared-choice checks, first-mention transcript excerpts, and their limitations are recorded in [observation_limits.json](gold/observation_limits.json) and [the non-recall report](gold/findings.md). Both strict date checking and the date judge specification use the accepted dates. No actual judge verdict was produced in this audit.

**Four unstated empty-pet answers.** These are `u2_p202603_0001`, `u2_p202608_0038`, `u2_r202702_0188`, and `u3_p202604_0007`. Day 1 is used as fallback evidence despite containing no statement about having no pets. The available primary prefixes omit pets; the retrospective adoption report does not establish there were no earlier pets. Answering `none` therefore requires an unstated assumption that unmentioned facts are absent. [Transcript checks](gold/manual_evidence.md).

**Two explanation citation gaps.** `u5_p202705_0094` and its retrospective `u5_r202710_0193` describe a nearby reading change in the gold explanation, while their evidence contains only the bookstore-closing distractor. The actual reading observations are available elsewhere in the prefix, so this is citation incompleteness. [Exact records](gold/issues.json).

**Three closure uncertainties.** `u3_p202711_0136`, `u4_p202606_0016`, and `u4_p202607_0019` ask for completed durations whose transitions are first observed in the following month. Two intervals end on the probe day; one ends the preceding day. The numbers match hidden truth. Completed-versus-elapsed wording and observed closure are uncertain. [Exact records](gold/uncertainties.json).

## Coverage and evidence metadata

All 30 never-mentioned abstention questions in the source pools are dropped because their evidence lists are empty. The 13 selected `abstention` instances are answerable controls, so the new probes test **zero actual unknown-answer cases**. There are also no prediction or current-pattern instances. These are coverage omissions, not 30 incorrect selected answers. [Selection checks](selection.json), [excluded IDs](gold/selection_gaps.json).

The final full-month probe is January 31, 2028, while each world ends February 28. The last 28 days therefore have no probe. This follows the full-calendar-month boundary rule rather than missing data. Each user has 158 primary questions and 42 retrospective copies, 200 total over 23 probe points.

Broader evidence metadata diagnostics are separate from the narrow question-error counts: 40 preference-duration records omit closing-transition citations, 36 of 59 pattern citations omit at least one rule value, eight current/at-time fact rows omit relevant prior removals, and 22 historical fact rows cite value updates later than their target. The last group includes three historical hobby rows citing the much later piano removal. Those updates are available by the actual probe; they are not after-cutoff evidence. They can delay question selection without directly supporting the earlier membership. Additional conversation observations can establish content that is absent from the listed citations. [Diagnostics](gold/evidence_diagnostics.json).

## Conversation source issues

There are 26 retained flagged sessions with 28 saved failures. Raw-text review confirms 12 avoidable absent-topic mentions, three conflicts between required content and absent-topic instructions, and four unmodeled persistent details. These findings concern 19 sessions cited by 33 probes. They establish source-contract problems, not 33 incorrect answers. All 54 probes citing **any** saved flag are mapped separately.

Seven saved flags are refuted by source evidence: two alleged inventions already have support, and five alleged missing events explicitly appear with correct relative dates. Two other alleged missing events have content present but dates underspecified; they remain unconfirmed. There is no wholly absent event among those seven `statement_missing` examples. An additional raw cycling observation disagrees with a rest-day world value, but whether it counts as modeled workout is uncertain and no monthly probe directly cites it. [Source report](source/report.md), [per-session probe mapping](source/flagged-session-index.md), [raw review annotations](source/flag_reviews.json).

Three confirmed writer-obligation conflicts require discussion of cycling, a climbing injury, or a podcast-host announcement on days those preference topics are marked absent. Native world validation does not check those prompt conflicts. The four unmodeled details concern a future trip, Phoenix venue backstory, a dog's leash behavior, and a future promotion celebration. These are concrete source-generation findings; no new gold mismatch is inferred solely from them.

## Validation and execution scope

- All **13,472 structural checks** pass across 1,000 unique IDs. Probing outputs reproduce exactly from the existing builder. This proves provenance, not correctness of its scheduling logic.
- The 489 recall and 511 other references were checked by family against their semantic target and actual probe. **999 match synthetic truth**; the stale current-hobby reference is the exception. A correct future-world answer can still be scheduled too early or lack observable support.
- All **2,738 sessions** match aggregate conversation records and intended sampling. The 912 unsampled world days are intentional.
- All **35 canonical/experiment JSON comparisons** match byte-for-byte.
- All five worlds reproduce in memory; native schema/spec/world/ladder checks pass. All **15,330 daily preference cells** and **516 exception pairs** agree with resolved rules and options.
- Replaying the validator over all saved extractions reproduces their flags exactly. This is not fresh semantic validation. Direct recall paraphrases absent from literal matching were reviewed by the audit agent: 73 unique sources representing 94 instances.

The existing evaluation loader reads 250 legacy questions and ignores the 1,000 monthly probes; the runner ingests the complete history before answering. An offline in-memory reproduction demonstrates that manually supplied March probes would see later sessions. These are recorded implementation requirements for the future reader/loader, not evidence of an already completed contaminated monthly run. No loader/reader changes were made. [Reproduction](source/integration.json).

## Artifacts and limits

[summary.json](summary.json) records reconciled counts; [question_coverage.jsonl](question_coverage.jsonl) accounts for every one of the 1,000 question instances. [question_findings.md](question_findings.md) lists every narrow question finding and location, while [findings.json](findings.json) preserves its rationale. Source, gold, recall, and review subdirectories keep primary evidence separate from interpretation. Input hashes are recorded in the root and workstream source inventories.

A fresh independent [synthesis review](review/review.md) verified coverage, the narrow union, all five change-date counterexamples, source hashes, and the distinction between hidden truth and observable support. [Consolidation verification](verification.json) checks source hashes, exact question payloads/locations, count overlaps, report links, and the absence of tracked implementation edits.

Reproduction scripts are local and offline: `structure.py`, `gold/audit_gold.py`, `recall/audit_recall.py`, the four scripts listed in `source/report.md`, then `aggregate.py`. Use `TMPDIR=/home/cloaked/projects/LifelongAgent/tmp PYTHONDONTWRITEBYTECODE=1` when running them. They write audit artifacts only.

The audit covers every record mechanically and by semantic family, with focused raw-text reviews and counterfactual checks. It does not prove exhaustive natural-language entailment or uniqueness for every inferred rule and cause in every conversation. The 71-instance union counts only the narrow findings described here; broad citation diagnostics, source exposure, selection omissions, and deferred integration are separate. No agent performance result or paid LLM judgment is implied by the offline harness stub outputs.
