# Updated v2 sources, copies, and coverage

The updated questions use unchanged conversation sources. All 2,738 individual session files have exactly their original audit SHA-256 hashes, and every aggregate/session/world projection still agrees. The 19 previously confirmed conversation-source contract findings therefore remain. They are cited by 38 current question instances; this is citation exposure, **not 38 wrong reference answers**. The question sampling changed, so exposure was recomputed from current `evidence_days` and actual payloads rather than old probe identifiers.

The update also leaves the true-unknown abstention coverage gap and the lack of predictions/current patterns. A new duplicate-copy inconsistency exists: the five experiment-directory `qa_pool.json` files are old, although all five experiment probing files match their updated canonical copies. The new `setup_2` runner reads canonical inputs and is unaffected by those stale pools.

## Source integrity and retained findings

| User | Sessions checked | Retained flagged sessions | Current questions citing any retained flag | Current questions citing confirmed source-contract findings | Final 28-day tail sessions |
|---|---:|---:|---:|---:|---:|
| u1 | 730 | 0 | 0 | 0 | 28 |
| u2 | 657 | 10 | 5 | 2 | 23 |
| u3 | 561 | 7 | 19 | 13 | 23 |
| u4 | 437 | 9 | 33 | 23 | 12 |
| u5 | 353 | 0 | 0 | 0 | 14 |
| Total | 2,738 | 26 | 57 | 38 | 100 |

[Session coverage](session_coverage.jsonl) checks every individual file hash, aggregate projection, calendar date, world `has_session`, turn shape, and validation flag consistency. There are no missing/orphan sampled sessions, metadata mismatches, shape failures, fidelity non-usage mismatches, or selected citations to unsampled days. The 912 world days without a session remain intentional sampling, not missing files.

[Current native checks](native_world_checks.json) independently repeat schema/spec/world/ladder checks, world regeneration, preference evaluation, and saved-extraction replay. All five worlds regenerate exactly; all 15,330 preference cells and 516 exception cells pass; all 2,738 saved validator verdicts reproduce; all 136 required-statement occurrences are accounted for in that replay. The 22 native-check source/persona hashes match their prior values in [the hash comparison](native_source_hash_comparison.json). Replay uses the saved extraction, not a new model extraction or exhaustive transcript entailment audit.

Every previously quoted transcript turn and every saved failure string was checked against the current individual session. None changed. [Flag reviews](flag_reviews.json) retain the earlier interpretations while replacing citation sets with current question payloads and locations:

- **12 avoidable absent-topic mentions** remain. Example: the unplanned current tennis report in `datasets/v2/u2/sessions/2027-12-11.json:37`.
- **Three conflicting writer obligations** remain: required cycling while workout is absent (`u4/sessions/2026-03-02.json:37`), a required climbing injury while workout is absent (`u4/sessions/2026-09-26.json:13`), and required podcast-host news while podcast is absent (`u3/sessions/2027-09-13.json:33`).
- **Four unmodeled persistent details** remain: a future trip (`u3/sessions/2027-06-04.json:45`), Phoenix venue-planning backstory (`u3/sessions/2027-06-26.json:53`), Dash's leash behavior (`u3/sessions/2027-12-01.json:45`), and a future promotion celebration (`u4/sessions/2026-11-16.json:29`).

These are unchanged contract/observation-generation findings. They do not individually establish incorrect selected gold answers. Current exposures excluding the three forced conflicts total 16 question instances; exposures to unmodeled details total 14. Their overlap is preserved in the machine records. The saved u4/day-111 extraction/world discrepancy (`cycling` versus `rest day`) is unchanged, still has zero directly citing current probes, and retains the leisure-versus-workout semantic caveat.

Seven old saved allegations were refuted by existing world/transcript support in the prior review and remain refuted. Two event dates remain underspecified in the transcript; neither a wholly absent event nor a definitely wrong date is established. There are still 26 flagged sessions with 28 retained diagnostic strings. Those diagnostics should not be equated with 28 proven errors.

The u1–u4 model logs remain LFS pointer text; u5 has its real usage log. This limits regeneration of historical usage from logs, not completeness of the conversation JSON. The available u5 usage still matches fidelity. No discarded transcript or model-call content was fetched.

## Canonical and experiment copies

[Copy comparison](copy_comparison.json) checks the same 35 JSON file pairs as before. **30 pairs match byte-for-byte and as parsed JSON. Five differ**, exactly the per-user `qa_pool.json` files. Canonical qa pools and probing files changed; experiment probing files changed to match, but experiment qa pools remain at their original audit hashes. Conversations, worlds, legacy curated QA, fidelity, and stats remain unchanged and equal across both locations.

All 1,000 current probing payloads agree with their canonical pool record at `original_id`. Only 214 agree with the stale experiment pool record; 786 do not:

| User | Current probe instances mismatching experiment pool payload |
|---|---:|
| u1 | 0 |
| u2 | 196 |
| u3 | 191 |
| u4 | 199 |
| u5 | 200 |

[Pool provenance](pool_provenance.json) lists every differing field and precise probe location. For example, `datasets/v2/u2/probing_questions.json:16` references `u2_q0172`, asking about March 28, 2026 TV. Canonical `datasets/v2/u2/qa_pool.json:3060` agrees (nature documentary). Experiment `experiments/data/synthetic_v2/u2/qa_pool.json:3598` instead asks about October 19, 2027 TV (sitcom reruns). This is a stale-copy/provenance issue for consumers joining experiment probes to the experiment pool; the self-contained current probes retain the canonical question and answer. The usual generator points at canonical `datasets/v2` (`generator_v2/config.py:9`), and `setup_2/loader.py:17` uses that canonical dataset.

## Question coverage

| Coverage | Before update | Current |
|---|---:|---:|
| Never-mentioned unknown questions in pool | 30 | 30 |
| Never-mentioned unknown instances in monthly probes | 0 | 0 |
| Answerable abstention control instances | 13 | 16 |
| Prediction instances in monthly probes | 0 | 0 |
| Current-pattern instances in monthly probes | 0 | 0 |
| Calendar probe points per user | 23 | 23 |

[Coverage evidence](coverage.json) lists all 30 omitted unknown payloads and locations, all selected controls, every pool prediction target, and every omitted current-pattern evidence set. Current answerable controls are u1=5, u2=3, u3=1, u4=4, u5=3. These are actual answers under the `abstention` category, not demonstrations that the agent can refrain from answering unknown facts.

`generator_v2/qa.py:545` creates true-unknown questions with an empty evidence list. `generator_v2/probing.py:84` still skips every empty-evidence item. The resulting absence of true-unknown cases is a coverage omission, not 30 invalid selected reference answers.

The current pools contain 369 predictions, all targeting dates after the dataset's February 28, 2028 end. The scheduler now excludes targets beyond `num_days` (`probing.py:119`). Future-target questions are appropriate for prediction as a question type, but this monthly generator does not create historical-checkpoint predictions; their absence remains a capability coverage limitation. The 21 end-of-history current-pattern questions all cite days 724–730, after the final monthly probe at day 702, and are dropped because February has no generated calendar probe (`probing.py:122`). No claim is made that all capabilities must appear in every user/month.

The final canonical probe is January 31, 2028 (day 702); the world and recorded data extend through February 28 (day 730). The calendar month ends on February 29, day 731, so `_month_boundaries` stops before it (`probing.py:39`). Thus the final 28 days contain 100 recorded sessions but no newly introduced questions. This is a probe coverage limit, not missing conversation data.

## Monthly setup integration

The former legacy-runner finding is **not an unresolved problem in `setup_2`**. [Actual loader checks](setup_2_load_check.json) load current canonical inputs: each user has 24 checkpoints, including February 28, 2028; all sessions enter exactly once in their own month and before the checkpoint cutoff; all repeated question wording, reference answers, accepted answers, and original probe dates agree with current source payloads. Each final checkpoint has 200 repeated questions and zero newly introduced questions. Ingestion therefore includes the final partial month, while the dataset still lacks probes about new February information.

`setup_2/runner.py:395` ingests only the current checkpoint's sessions, then finalizes summaries and distillation at line 402, answers at line 409, and grades at line 414 before advancing. The old `experiments` entry point still reads the unchanged curated `qa_pairs.json` and ingests everything before answers; that remains a separate supported legacy evaluation protocol, not a new monthly-dataset defect. No live evaluation, model construction, or new performance measurement was carried out here.

## Reproduction and limits

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python .research/v2-probing-recheck/source/recheck_source.py
```

The script imports the original mechanical source and native-check scripts with outputs explicitly redirected into this source recheck folder, then performs current hash comparisons, transcript-quotation checks, current citation mapping, coverage/provenance checks, and an offline `setup_2` load. It writes only beside itself. [Summary](summary.json) gives machine counts; [recorded stdout](checks.stdout.txt) preserves check output. Dataset, generator, agent, setup, and original audit files were not edited. No root `/tmp` writes, network access, or paid API calls were made.

This workstream does not re-prove all natural-language entailments in all 34,659 turns. Exact source identity allows carrying forward the prior focused raw-transcript reviews; new selected-question answerability is evaluated by the question-family workstreams. Source citation exposure and coverage counts stay separate from wrong-question counts.
