# Recall audit conclusions

All **489 monthly recall instances** have the correct reference answer and sole accepted value for their target day in `world_state.json`. Independent rule evaluation also agrees with each target's base value, including resumed anchors after temporary regimes. The 10 exception instances use valid persona exceptions and the exceptional value, not the ordinary routine value.

There are **7 distinct flagged instances**: 4 future recall targets already included in the broader future-date problem, 1 inferred answer whose workout phase was never observed, and 2 inferred answers across unobserved routine boundaries. The latter 3 are additional **conversational answerability defects**, not incorrect synthetic gold answers. There are no overlapping instances among these recall categories.

| User | Recall instances | Retrospective copies | Inferred | No target-day session | Exceptions |
| --- | ---: | ---: | ---: | ---: | ---: |
| u1 | 152 | 32 | 0 | 0 | 0 |
| u2 | 127 | 26 | 14 | 4 | 1 |
| u3 | 98 | 19 | 34 | 20 | 4 |
| u4 | 72 | 15 | 41 | 22 | 3 |
| u5 | 40 | 5 | 28 | 14 | 2 |
| Total | 489 | 97 | 117 | 60 | 10 |

All 489 original IDs map to the QA pool and preserve the original question, answer, accepted values, grading, evidence, and tags. All `inferred`, `no_session`, `exception`, and `boundary` tags agree with target-day metadata. All selected evidence lists reproduce the generator's selection. All 707 cited day/domain references exist in the world, individual sessions, and aggregate conversations; the two conversation representations agree, and every saved validator preference extraction matches that evidence day's truth. All evidence dates are within the probe cutoff.

The audit accepts ordinary inference from earlier or later pre-probe observations. **110 inferred instances** have same-phase observations and none of the specific timing or evidence defects found here. Missing target-day sessions alone are not counted as errors.

## An exercise phase never observed

**u5_p202605_0011 / u5_q0006** asks what Priya's workout was on **2026-05-09 (day 70)**, at the May 31 probe. Its gold is **bike ride**. The [question and answer](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:284) agree with [hidden target-day truth](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/world_state.json:10734).

The two-week exercise rule assigns bike rides to Saturdays in its second block, but **none of the six days in that phase in the entire initial regime has an exercise observation**. Day 14 has a conversation that omits exercise; days 28, 42, 56, 70, and 84 have no conversation. The generator [falls back to unrelated phases when no same-phase mention exists](/home/cloaked/projects/LifelongAgent/generator_v2/qa.py:193). It cites days 68, 77, and 79, whose workouts are:

- **Bouldering:** climbing short walls with crash pads and no harness. [Day 68 text](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-05-07.json:45).
- **Long run:** jogging for nearly two hours. [Day 77 text](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-05-16.json:29).
- **Swim:** laps in the indoor pool. [Day 79 text](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-05-18.json:29).

The full pre-probe bicycle/pedaling/rotation language was reviewed for attribution. Priya's bicycle reports describe commuting to the office; workout cycling is attributed to coworker Dana, including [May 24](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-05-24.json:29) and [May 30](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-05-30.json:13). There is no stated exercise rule that fills the missing phase. The question has one monthly instance and no later retrospective copy. The specific bicycle-workout answer is therefore unsupported, even granting the intended pattern-inference assumption. Full observations, matched incidental text, attribution review, and the six phase days are preserved in [phase_transcript_review.json](phase_transcript_review.json).

## Unobserved shift boundaries leave two exact answers ambiguous

These findings were challenged using the entire pre-probe sequence of recorded choices, rather than the first observation alone. Both new rules are insensitive to their start-day anchor: a calendar weekday rule and a constant rule. Moving the start later can preserve **every observed choice** while changing the requested target's answer. Thus later cycle observations cannot resolve these cases. The generator's [change-detection policy itself accepts every date from effective start through first observation](/home/cloaked/projects/LifelongAgent/generator_v2/qa.py:263).

| Instance / original | Target and gold | Compatible later start | Different target answer | Observed domain days checked |
| --- | --- | --- | --- | ---: |
| u3_p202711_0137 / u3_q0130 | Oct 24, 2027, d603: comedy podcast | d604 instead of d603 | Italian lessons podcast | 165 |
| u5_p202610_0047 / u5_q0048 | Sept 17, 2026, d201: synthwave playlists | d202 or d203 instead of d200 | film scores | 42 |

**u3:** The [recall](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/probing_questions.json:4397) is asked on Nov 30. The target Sunday has no session. The last old observation on Oct 22 reports an [Italian language lesson podcast](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/sessions/2027-10-22.json:29). The first new observation on Oct 25 says the podcast routine has **“changed recently”** and reports dog training after Dash's adoption, but [does not date the change](/home/cloaked/projects/LifelongAgent/datasets/v2/u3/sessions/2027-10-25.json:29). Starting the weekday rule on Monday d604 preserves all 165 observed domain days through the probe, including all subsequent Sunday comedy observations, while leaving d603 in the old Italian routine.

**u5:** The [recall](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:1317) is asked on Oct 31. Its target-day conversation omits music. Sept 12 reports [wordless instrumental hip-hop](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-09-12.json:37); Sept 19 first reports [retro-futuristic electronic synthesizer beats](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/sessions/2026-09-19.json:29). Neither dates the switch. Starting the constant synthwave routine on d202 or d203 preserves all 42 observed domain days through the probe. The old weekly rotation would yield **film scores** on d201; the earlier observed lo-fi value is not that target day's old-cycle value.

In both cases the gold remains the correct hidden value, and a plausible guess can match it. The confirmed limitation is that the actual text does not distinguish the hidden start from another compatible start. Retain this distinction when deciding whether to exclude these questions or permit uncertainty. [issues.json](issues.json) records all counterfactual start checks and zero mismatch results; [boundary_transcript_reviews.json](boundary_transcript_reviews.json) preserves the boundary and later pre-probe observations.

## Four future recall targets

| Instance / original | Probe date | Target date | Days after probe |
| --- | --- | --- | ---: |
| u4_p202605_0012 / u4_q0015 | 2026-05-31 | 2026-08-23 | 84 |
| u4_p202707_0104 / u4_q0108 | 2027-07-31 | 2027-08-20 | 20 |
| u4_p202710_0132 / u4_q0072 | 2027-10-31 | 2027-12-14 | 44 |
| u4_p202801_0153 / u4_q0029 | 2028-01-31 | 2028-02-04 | 4 |

Each later-day gold is synthetically correct. The wording asks a past-tense recall at a viewpoint that has not reached the target date. Earlier observations could support a prediction, but they do not make this a past recall. Exact record and state lines are in [issues.json](issues.json).

## Scope and residual uncertainty

All 372 direct target observations were checked for conversational support. Of these, 278 contain the gold value literally in a user turn and have matching saved extraction; the remaining 94 instances correspond to **73 distinct source questions**, all given manual transcript review by the audit agent and judged supporting paraphrases. Their exact excerpts are in [manual_paraphrase_reviews.json](manual_paraphrase_reviews.json). For example, “slow-steeped chilled coffee” supports cold brew; “a cozy knit pullover” and “denim trousers” support sweater and jeans. The idiom “pounded the pavement on foot” for one u4 running question is less explicit but was not flagged as a contradiction.

Saved closed-option validator outputs are supporting source evidence, not an independent proof of every sentence's semantics. This audit does not claim to have semantically revalidated every incidental statement in every conversation. Ordinary inferred questions remain pattern-based inferences in a dataset that permits exceptions. The three additional findings isolate concrete missing information that defeats their particular gold answer, rather than treating all inference or all absent sessions as invalid.

Reproduce with `python .research/v2-probing-audit/recall/audit_recall.py`. The script writes only audit artifacts in this directory and makes no network or model calls. [coverage.jsonl](coverage.jsonl) contains every one of the 489 instances, [evidence.jsonl](evidence.jsonl) contains all 707 citation checks, [summary.json](summary.json) contains counts, and [sources.json](sources.json) contains source-file hashes.
