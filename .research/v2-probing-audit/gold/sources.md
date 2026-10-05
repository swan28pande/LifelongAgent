# Source roles for the gold audit

The audit uses local canonical sources and no network or model calls. A missing root `CONTEXT.md` and missing `docs/adr/` were checked; `generator_v2/DECISIONS.md` contains the applicable existing semantics. Source evidence and audit interpretation are separated in the records below.

| Source | Authority in this audit | Limitation |
| --- | --- | --- |
| `datasets/v2/u1..u5/probing_questions.json` | Actual question, gold, accepted answer, evidence and changed monthly viewpoint; exact JSON-pointer lines in coverage | Pool equality does not prove gold validity at the new viewpoint |
| `datasets/v2/u1..u5/world_state.json` | Per-day active/known facts, base preference values, exceptions, resolved regimes and effective/stated updates | Synthetic truth is not proof of conversational observation |
| `datasets/v2/u1..u5/qa_pool.json` | Original source question and provenance | Original pool questions are written for day 731 |
| `generator_v2/personas/u1..u5.yaml` | Fact cardinality/labels, event chains and named other people/distractors | Does not establish what an evaluated memory has observed |
| `generator_v2/personas/difficulty_ladder.yaml` | Permitted cause-lag intervals used in bounded counterfactual checks | Private specification constraints can be stronger than conversational knowledge |
| `generator_v2/rules.py:82` | Pure rule evaluation and anchors; describe at line 176 | Rule/cause inference uniqueness is not automatically guaranteed |
| `generator_v2/simulator.py:74` | Active facts by effective day; known facts at line 145; temporary return at line 195 | No explicit negative statement is generated for a never-added multi-valued fact |
| `generator_v2/qa.py:262` | Change bounds: effective start through first mention | First mention may have the same value as the previous rule |
| `generator_v2/qa.py:347` | Maximal fact intervals; untimed per-value evidence at line 360; final current facts at line 370 | Evidence for current remaining members can omit a later removal of another member |
| `generator_v2/qa.py:426` | Event final/midpoint status references | A midpoint target can lie after the assigned monthly probe |
| `generator_v2/qa.py:441` | Inclusive durations; old regime's first/last observation cited at line 454 | No closing successor observation is cited for preference durations |
| `generator_v2/qa.py:471` | Causal distractor controls and deliberate confounder explanations | Confounder explanation cites only the distractor |
| `generator_v2/qa.py:522` | Genuine unknown abstention versus answerable controls | Unknown questions carry empty evidence |
| `generator_v2/probing.py:78` | Assignment by latest evidence, only fact_at_time target clamping; unchanged copying at line 145 and retrospective copying at line 197 | Other historical target dates, gold-only endpoints and relative/current meaning are not revalidated |
| `generator_v2/conversation.py:84` | Explicit change declarations; implicit daily reporting at line 88 and no-reason reporting at line 96 | These instructions do not guarantee every recorded paraphrase is correct |
| `generator_v2/DECISIONS.md` | Effective dates, bounded date acceptance, continued base anchors, no-retraction semantics and intended implicit inference | These conventions do not resolve every observational ambiguity |
| `experiments/core/grading.py:19` | Judge's date/list/history contract; date_exact at line 50; complete reference in judge prompt at line 65 | A short free-text accept hint is not a complete strict-match answer |
| Selected canonical `conversations.json` user turns | Narrow corroboration of retained piano and absence of an announced date in selected first-change observations | No exhaustive entailment judgment for all 511 questions is claimed |

Exact per-claim source locations are in `coverage.json`, `issues.json`, `observation_limits.json`, `uncertainties.json`, and `evidence_diagnostics.json`. `input_sha256.json` fixes the source versions for the automatic checks; `manual_transcript_checks.json` fixes the aggregate conversation versions for the narrow transcript scans.
