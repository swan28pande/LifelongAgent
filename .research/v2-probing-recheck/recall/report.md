# Current recall verification

All **496 current monthly recall instances** were checked, including **390 distinct source questions**, **122 inferred instances**, **374 direct instances**, and **601 day/domain citations**. All reference answers and sole accepted values agree with independent raw-rule evaluation and hidden target-day truth. All targets and citations are at or before the probe date; every inferred citation is now also at or before the target date. No wrong synthetic reference or future recall remains in this sample.

**Two current instances have confirmed conversational answerability limitations:** one retained missing-phase problem and one newly detected boundary problem. These are not incorrect answers in hidden world state. The boundary check was broadened to consider targets on either side of a hidden change, so the second finding must not be described as introduced by this update merely because the old audit did not flag it.

## Remaining instances

| Current instance / source | Question and sole gold | Evidence and limitation |
| --- | --- | --- |
| `u5_p202605_0008` / `u5_q0005` | Priya's workout on May 9, 2026: **bike ride** | [Question](/home/cloaked/projects/LifelongAgent/datasets/v2/u5/probing_questions.json:215). Now cites only day 68, May 7, whose exercise is **bouldering**. None of the six Saturdays in this bike-ride phase in the whole initial regime has an exercise observation. This is the previous `u5_p202605_0011` problem under shifted IDs. |
| `u4_p202712_0152` / `u4_q0071` | Daniel's work music on Dec 19, 2027: **jazz records** | [Question](/home/cloaked/projects/LifelongAgent/datasets/v2/u4/probing_questions.json:4820). Target session omits music. The next, uncaused ambient/classical routine has a hidden Dec 20 start; moving it to Dec 19 preserves **all 22 recorded work-music choices through the Dec 31 probe**, but changes the requested answer to **ambient electronic**. No raw conversation fixes this start date or weekday anchor. |

For Priya, truncating the earlier three references `[68,77,79]` to `[68]` removes post-target citations but does not establish the unseen bike-ride workout. The entire pre-probe cycling/rotation wording and its attribution were checked: her bicycle reports are commutes, and other workout cycling belongs to Dana. The six unobserved phase days are 14, 28, 42, 56, 70, and 84. `phase_reviews.json` preserves the observations and matched text. This agrees with the baseline attribution review because the relevant source conversations and world state are unchanged.

For Daniel, December 17 reports indie guitar music; December 19 contains no work-music observation; December 21/22/24/25 report ambient synthesizer music; December 27 reports orchestral/chamber music. Starts on December 19 or 20 yield the same observed sequence before the probe. An earlier December 18 start and a later December 21 start are rejected by actual later values, so this is a tested counterexample rather than a claim that every inferred boundary is unknowable. `new_boundary_transcript_review.json` preserves all 22 observed domain days, the target session, and all matched date/change language across the prefix.

## Prior recall findings

| Prior category | Current disposition |
| --- | --- |
| Four future targets | **0 current future recall instances.** The August 23 breakfast and August 20 dinner questions are now introduced on August 31, after their targets. The December 14 music question remains in the pool but is not selected. The February 4, 2028 workout question remains in the pool but is not selected in the 23-probe sample. |
| Priya's unseen Saturday bike phase | **Still present**, with the same question/answer and a shorter reference list. |
| Sofia's Oct 24, 2027 podcast boundary | Exact question absent from both current QA pool and monthly probes. |
| Priya's Sept 17, 2026 work-music boundary | Exact question absent from both current QA pool and monthly probes. |

The updated `generator_v2/qa.py` retains only inferred references at or before the target and skips a sampled recall if no such reference exists. That excludes the two old targets before the first new-regime observation. It does **not** repair the undated source transitions or generally prove all remaining boundary answers. The missing-phase fallback remains in `_nearest_mentions`, and a target immediately before an unobserved change can still be ambiguous. `baseline_matches.json` matches exact question text and domain, not old shifted identifiers.

## Direct text and rejected candidates

- **278 direct instances** contain the intended value literally in a user turn with matching saved extraction.
- **96 direct nonliteral instances / 76 unique source questions** received transcript support review. Thirty-seven unique reviews were reused only after matching user, domain, target date, complete question/answer payload, and unchanged source-file SHA-256. The remaining **39 unique transcripts** were read afresh; their selected supporting turns are in `new_manual_turns.json`. No direct answer contradiction was found.
- Seven inferred boundary candidates were challenged by moving their starts across observational gaps. Two candidates have no alternative value compatible with all observations. Four Daniel dinner instances have value-sequence alternatives only if the new deadline-driven delivery-sushi period begins before the explicitly dated August 22 deadline; these are **not counted** as confirmed findings. One uncaused music case remains. `boundary_candidate_dispositions.json` preserves the exclusions.

## Evidence and limits

`coverage.jsonl` covers every current recall instance; `evidence.jsonl` covers every citation; `summary.json` contains counts; `issues.json` contains the two final findings. `audit_recall.py` reproduces raw coverage and counterfactual candidates offline; final interpretation and rejected-candidate dispositions are separate review artifacts. It uses the old audit's JSON line indexer and independent rule interpreter, but never imports the generator evaluator or calls a model.

Saved validator extraction is supporting evidence rather than a fresh independent model validation. Ordinary stable-pattern inference is accepted, including missing target sessions. The specific two findings isolate absent information and a concrete compatible alternative; this report does not treat all inferred answers as errors. No dataset, generator, agent, setup, or prior audit file was changed.
