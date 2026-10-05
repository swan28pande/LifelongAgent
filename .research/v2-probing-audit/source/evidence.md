# Saved evidence and coverage

The following files are the reproducible source-evidence layer. Interpretations are in `report.md`, `flag_reviews.json`, and `issues.json`.

| Artifact | Coverage/content |
|---|---|
| `inventory.json` | All five users; aggregate/individual/world metadata equality, saved fidelity reconciliation, counts, LFS-pointer status; u5's real usage-log reconciliation. |
| `copy_comparison.json` | 35 canonical/experiment file pairs, both SHA-256 hashes, byte and JSON equality. |
| `session_coverage.jsonl` | Every one of 2,738 dated canonical sessions, file SHA-256, source line, shape/validation fields, first-attempt diagnostics, and citing probe IDs. |
| `probe_coverage.jsonl` | Every one of 1,000 monthly questions, exact file line, evidence days, flagged references, and hypothetical whole-history post-cutoff session count. |
| `flagged_sessions.json` | All 26 retained flagged sessions, 28 original failure strings, saved extractor values, world-day evidence, required statements, and all raw turns with exact line numbers. |
| `native_world_checks.json` | All native schema/spec/world/ladder checks, simulator replay equality, resolved-rule/exception check summaries, and source hashes. |
| `world_preference_coverage.jsonl` | All 15,330 day/preference cells; rule, anchor, evaluated/stored base, selected value, and exception metadata. |
| `validator_replay.jsonl` | Every saved extraction passed through the actual validator's deterministic diff; required statement IDs/count, original/replayed failures, and failed sessions' known/cause inputs. |
| `integration.json` | Actual loader output and metadata comparisons; actual runner chronology/cutoff observations with explicit offline substitutes; grade dispatch evidence. |
| `transcript_candidates.jsonl` | Lexical screening candidates only; routine words are not automatically labeled as errors. |
| `verification.json` | Final input/code hash rechecks, coverage assertions, corrected u5 usage-log status, and confirmation the unsupported speaker allegation was retracted. |

There are no aggregate/session/world discrepancies, copy discrepancies, native schema/spec/world/ladder errors, resolved-rule/exception errors, or recorded-validation replay discrepancies. These are mechanical/source findings, not proof of language-level correctness.

Saved remaining failure strings by user are 10 for u2, nine for u3, nine for u4, and zero for u1/u5. Their category totals are 15 preference leaks, six invention strings, and seven missing-statement strings. Every original string and exact validation/raw-turn/world-day location is retained in `flagged_sessions.json`.

The offline execution evidence is confined to `offline_runner/`; its stub verdicts intentionally do not represent model accuracy or real benchmark results. No LLM instance was created and no model call occurred.
