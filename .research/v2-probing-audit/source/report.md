# Canonical v2 conversation sources and execution audit

The canonical inputs are complete and internally consistent at the file and synthetic-world levels. All 2,738 aggregate sessions match their individual session files and the world's sampled-session metadata. All 35 compared canonical/experiment JSON files are byte-identical. The ordinary experiment entry point currently evaluates 250 curated end-of-history questions, not the 1,000 monthly probing questions. Its runner does not enforce per-question monthly cutoffs.

There are 26 previously recorded flagged sessions containing 28 saved failure strings. They are cited by 54 monthly questions. Independent transcript review confirms several source-contract departures, but also refutes seven of those flags and leaves two alleged missing-event failures unconfirmed. Citation exposure is not a count of erroneous probing questions. This workstream establishes no additional wrong gold answer solely from these flags.

## Exhaustive mechanical checks

| User | Recorded sessions | World days without a session | Saved flagged sessions | Monthly questions citing flags |
|---|---:|---:|---:|---:|
| u1 | 730 | 0 | 0 | 0 |
| u2 | 657 | 73 | 10 | 7 |
| u3 | 561 | 169 | 7 | 18 |
| u4 | 437 | 293 | 9 | 29 |
| u5 | 353 | 377 | 0 | 0 |
| Total | 2,738 | 912 | 26 | 54 |

`audit_source.py` checks every aggregate date against the corresponding individual session and `world_state.days[].has_session`. Each aggregate's `day`, `date`, `weekday`, and complete `turns` array equals the individual-session projection. User/name metadata, date arithmetic, weekdays, chronological aggregate order, turn shape, and alternating speakers also agree. There are no missing/orphan/extra sampled sessions or invalid/empty turns. Every session has 12–20 turns. No monthly question cites an unsampled world day. JSON input loading rejects duplicate object keys.

All seven compared files per user match byte-for-byte: `conversations.json`, `probing_questions.json`, `world_state.json`, `qa_pairs.json`, `qa_pool.json`, `fidelity.json`, and `stats.json`. SHA-256 values for both copies are in `copy_comparison.json`; every individual canonical session's SHA-256 is in `session_coverage.jsonl`. The comparison covers all inputs used by the current v2 loader and all canonical monthly probing files.

Native Pydantic persona/world parsing, `checks.check_spec`, `checks.check_world`, and measured ladder-knob checks all pass for all five users. Calling the deterministic simulator reproduces every saved world JSON exactly. Evaluating each stored resolved regime with its stored anchor, date, and city context validates all 15,330 day/preference cells. There are zero base-value, regular-value, regime-membership, or exception-option mismatches. All 516 exception value/reason pairs are allowed persona options. These checks establish consistency of synthetic truth, separately from conversational observability.

Replaying the real `validator.build_check` and `validator.diff` over all 2,738 **saved extractions** reproduces all saved failure strings exactly, including the 136 scheduled statement occurrences. This replay is a deterministic consistency check, not a new semantic extraction. Non-usage fields in every `fidelity.json` also reproduce exactly from the final session files. The `passed_after_retries` field counts all eventual passes, including first-attempt passes, as implemented in `conversation.py:219`.

The canonical u1–u4 `llm_log.jsonl` files are Git LFS pointers, so their model usage cannot be independently recomputed from logs in this checkout. u5 has a real 1,735-record JSONL usage log, and its recomputed usage exactly matches the saved fidelity usage. These logs contain model-call metadata, not discarded transcript content. The LFS pointers do not truncate the independently available conversation JSON or session JSON inputs. The audit never calls the full fidelity function that would try to parse pointer text as JSONL. Final u5 repair history is documented in `datasets/v2/u5/GENERATION_NOTES.md:3`; zero final u5 flags does not imply an independent semantic audit of every turn.

## Recorded flags versus independently checked claims

The 28 saved failures comprise 15 `pref_leak`, six `invention`, and seven `statement_missing` strings. Every flag lives in an individual session's `passed_validation`, `failures`, and `extracted` fields. `fidelity.json` mirrors those retained records. First-attempt failures that disappear after retries are historical diagnostics and are not defects in the final conversation text.

| Raw review classification | Failure occurrences | Interpretation |
|---|---:|---|
| Avoidable discussion of an absent topic | 12 | Topic really appears even though the planned mention flag is false. |
| Required content overlaps an absent topic | 3 | Topic appears, but literal writer obligations conflict. |
| Unmodeled persistent detail | 4 | The text adds a plan/backstory/detail forbidden by the writer contract. |
| Alleged invention contradicted by source support | 2 | Existing world/prior transcript entails the flagged fact. |
| Alleged missing statement contradicted by correctly dated text | 5 | Event and correct relative date are explicitly present. |
| Event content present, date underspecified | 2 | A precise wrong/missing date is not established from the wording. |

The 15 topic mentions are independently visible in raw text, including references to earlier days, future plans, and questions. The validator explicitly labels historical/planned mentions as `other`, and the writer forbids absent topics even in passing (`prompts/validator_system.txt:11`; `prompts/conversation_system.txt:23`). u2 day 651 supplies the correct **current** tennis value as an extra observation (`sessions/2027-12-11.json:37`).

u4 day 111 reveals a masked value discrepancy: the user reports today's bike ride (`sessions/2026-06-19.json:21`), the saved extractor chooses `cycling`, and the world workout value is `rest day`. Because `mentioned` is false, `validator.py:89` reports only a preference leak against `not mentioned`, not the disagreement with the actual world value. This is a confirmed extra cycling observation and a recorded-extraction/world disagreement. Whether the ride represents a workout rather than leisure cycling is a semantic limitation; no monthly probe directly cites day 111. The discrepancy is preserved in `flag_reviews.json.masked_absent_topic_value_disagreements`.

Three retained flags expose contradictory prompt obligations:

- u4 day 2 must introduce `hobbies.1 = cycling` while `workout.mentioned` is false. Cycling is stated in `datasets/v2/u4/sessions/2026-03-02.json:37`.
- u4 day 210 must report a wrist fracture while climbing while workout is absent. The required climbing incident appears in `datasets/v2/u4/sessions/2026-09-26.json:13`; an additional discretionary bicycle question appears at line 9.
- u3 day 562 must mention `distractor2`, her favorite podcast host's book tour, while podcast is absent. The podcast-host news appears in `datasets/v2/u3/sessions/2027-09-13.json:33`.

`conversation.py:54` prohibits absent topics, including historical and other-person mentions; `conversation.py:65` requires fact/event updates and `conversation.py:101` requires distractors. `checks.check_world` validates session scheduling and preferences (`checks.py:216`) but does not test these lexical topic/content conflicts. Thus native world checks passing does not prove every writer instruction is simultaneously satisfiable.

The four independently visible unmodeled additions are u3 day 461's future trip (`sessions/2027-06-04.json:45`), u3 day 483's previous focus on Phoenix wedding venues (`sessions/2027-06-26.json:53`), u3 day 641's Dash leash-pulling/excitement backstory (`sessions/2027-12-01.json:45`), and u4 day 261's promotion celebration next weekend (`sessions/2026-11-16.json:29`). The writer forbids unprovided lasting trips/plans/backstory/pet details (`prompts/conversation_system.txt:47`), whereas everyday texture is allowed at line 52. These are source-contract departures, not established contradictions of the day's required choices. Phoenix venue browsing also appears in an earlier transcript; the venue flag therefore concerns embellishment beyond the world, not inconsistent conversation history.

Two invention flags are refuted:

- u2 day 121's tennis-club membership was already `hobby.tennis`, effective and stated day 120. The day-120 user says “I joined a tennis club today!” (`datasets/v2/u2/sessions/2026-06-28.json:29`). Repeating that it happened “this weekend” on Monday is consistent. The checker retains hobbies as values but not all historical fact-change wording: `validator.py:42` only includes earlier event updates, other-person news, and distractors in historical text.
- u3 day 181's “has a roommate” is entailed by the mandatory `distractor0` text, “her roommate bought a new espresso machine”. It is not an unsupported new roommate (`datasets/v2/u3/sessions/2026-08-28.json:21`).

Five alleged missing statements are explicitly present with correct dates:

| User/day | Statement | Verbatim timing/content | Raw session line | Effective day |
|---|---|---|---|---:|
| u3/141 | `flu.0` | “came down with a really nasty case of the flu yesterday” | `u3/sessions/2026-07-19.json:13` | 140 |
| u3/641 | `ankle.0` | “sprained my ankle yesterday while I was out on a hike” | `u3/sessions/2027-12-01.json:13` | 640 |
| u4/151 | `cooking_class.0` | “last night” and an instructional session preparing Thailand's dishes | `u4/sessions/2026-07-29.json:21` | 150 |
| u4/301 | `vinyl.0` | “Yesterday” and grandfather's vinyl compilation/stack of albums | `u4/sessions/2026-12-26.json:29` | 300 |
| u4/542 | `deadline.0` | “Two days ago” and a massive design competition deadline | `u4/sessions/2027-08-24.json:21` | 540 |

The other two flagged event contents also appear. In u3 day 451, a new sandwich shop “just opened” (`sessions/2027-05-25.json:29`) but May 24 is not precisely identified. In u3 day 562, the assistant saw news “this morning” that the host “just announced” a book tour (`sessions/2027-09-13.json:33`); seeing news that morning does not prove the announcement itself occurred that morning instead of September 11. Distractors are a separate writer block from required fact/event updates, so the assistant introducing this small talk is not independently proven to violate a user-only update rule. Their saved extractor verdicts are retained as reported evidence; neither missing content nor an unambiguous wrong date is confirmed.

There is no wholly absent event content among the seven retained `statement_missing` examples. No malformed/truncated turn was identified by the exhaustive shape/terminal-punctuation checks. This is not a semantic guarantee that every unflagged turn contains every intended nuance: raw semantic review focused on all 26 retained flagged sessions and representative leakage language, not all 34,659 turns.

The confirmed absent-topic/unmodeled-detail findings concern 19 distinct sessions cited by 33 probes. Excluding the three contradictory required/absent cases leaves 16 sessions cited by 19 probes. Four unmodeled-detail sessions are cited by 16 probes. Those overlaps are recorded explicitly in `flag_reviews.json`. The 54 broad flag-citation exposures are neither 54 wrong questions nor 54 unavailable gold answers: many questions cite unaffected facts/choices in the same session, and several depend on the correctly stated event refuting the original flag. The complete per-session ID mapping is in `flagged-session-index.md` and machine-readable `flag_reviews.json`.

## Execution integration

`experiments/run.py:29` routes the v2 benchmark to `experiments/benchmarks/v2.load`. The loader reads `qa_pairs.json` at `v2.py:16`, constructs 50 questions per user, and never reads `probing_questions.json`. Offline execution of this actual loader produces five instances, 250 question IDs, and **zero** of the 1,000 monthly IDs. All loaded questions are asked on February 29, 2028, day 731. Metadata is correctly preserved for these curated questions: answer, accepted alternatives, answer type, grading, category, date, capability, tags, and evidence days match the canonical curated source exactly. Session dates are sorted.

The actual runner ingests all sessions at `runner.py:73` before any answers at `runner.py:99`. It never filters or advances memory according to `Question.asked_on`. `prompt_for` exposes a question date (`types.py:51`) but a date in prompt text does not enforce memory isolation. Manually supplying the canonical `u1_p202603_0001` and `u1_p202603_0002` to the unchanged runner confirms that both March 31, 2026 questions are answered after all 730 sessions have been ingested, including **699** later sessions through February 28, 2028. This is an offline reproduction with in-memory substitutes; the ordinary loader currently never supplies monthly questions. It proves a monthly integration blocker and a potential leakage path, not that a completed monthly evaluation run used leaked data.

Grading receives the expected question/answer/accepted-answer metadata. The runner always calls the common LLM judge (`runner.py:150`), separately stores strict exact/date checks (`grading.py:48`), and reports judge accuracy (`runner.py:238`). The grading field does not switch the headline metric to exact-match accuracy. An offline stub deliberately returning WRONG while an exact response matches its accept list verifies that dispatch and summary choice. No actual model was instantiated. `judge_prompt` does not include `asked_on` (`grading.py:65`); relative wording is judged against the supplied reference. This is a harness limitation when auditing changed viewpoints, not evidence that any particular accepted gold is incorrect.

The two integration records are separate from source/question-error counts in `issues.json`: monthly files ignored, and monthly cutoffs absent. No code or datasets were changed.

## Reproduction and limitations

Run from the project root with existing Python dependencies, in this order:

```bash
python .research/v2-probing-audit/source/audit_source.py
python .research/v2-probing-audit/source/check_native_world.py
python .research/v2-probing-audit/source/reproduce_integration.py
python .research/v2-probing-audit/source/review_flags.py
python .research/v2-probing-audit/source/verify_artifacts.py
```

`audit_source.py` inventories/hashes inputs and saves exhaustive session/probe coverage. `check_native_world.py` exercises native schema/spec/world/rule checks and saved-extraction replay with LLM construction disabled. `reproduce_integration.py` uses the actual loader/runner with explicit offline memory/judge substitutes and confines its outputs to `source/offline_runner/`. `review_flags.py` materializes transparent manually reviewed annotations; it does not manufacture a new extractor verdict. Source-code and persona hashes are in `native_world_checks.json`.

The report distinguishes synthetic truth, observed transcript text, recorded model judgments, and inferred implications. It does not claim exhaustive semantic correctness of all conversation turns, pattern inference, or gold questions. First-attempt diagnostic flags, plausible everyday details, paraphrases, event content already present, and intentional sampling gaps are not automatically counted as input defects. No network, paid API, external writes, dataset edits, generator edits, or evaluation-code edits were used. All artifacts and temporary runtime files remain under this source-audit directory.
