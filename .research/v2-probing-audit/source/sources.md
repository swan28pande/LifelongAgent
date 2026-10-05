# Local source index

All sources are repository files. No external research or paid services were used.

- `datasets/v2/u1..u5/conversations.json`, `sessions/*.json`, and `world_state.json`: canonical observed sessions, recorded validation, and synthetic world respectively. Every session input has a SHA-256 entry in `session_coverage.jsonl`; top-level canonical/experiment hashes are in `copy_comparison.json`.
- `datasets/v2/u1..u5/probing_questions.json`: all 1,000 monthly IDs, dates, evidence-day references; `probe_coverage.jsonl` contains every ID's exact source line and flag exposure.
- `datasets/v2/u1..u5/fidelity.json`: saved counts and flagged dates; reconstructed non-usage values are in `inventory.json`.
- `datasets/v2/u5/GENERATION_NOTES.md:3`: repairs and remaining independent-audit limitation.
- `generator_v2/DECISIONS.md:25`: deliberate session sampling and moved statements; line 59 describes writer observability, line 61 distinguishes lasting inventions from everyday details, line 62 retains best failed attempts, line 66 distinguishes curated questions from pool questions.
- `generator_v2/config.py:9`: canonical generator output directory; lines 78–80 define retries and turn bounds.
- `generator_v2/schema.py:148`: fact label behavior; line 269 loads native personas/ladder; line 339 defines days; line 356 defines worlds.
- `generator_v2/simulator.py:235`: sampled sessions; line 253 computes rule values; line 269 samples exceptions; line 353 constructs stored day states.
- `generator_v2/rules.py:55`: context; line 82 evaluates resolved rules with anchor; line 86 uses 7-day anchored blocks.
- `generator_v2/checks.py:168`: native world checks; line 216 checks exception observability/session instructions; line 271 compares ladder knobs.
- `generator_v2/validator.py:30`: required statement formatting; line 40 restricts historical narrative inputs; line 47 builds exact checks; line 83 compares saved extractions; line 89 distinguishes absent/leak/wrong preferences; line 93 flags missing statements; line 106 emits recorded invention strings.
- `generator_v2/conversation.py:32`: writer day input; line 54 bans absent topics; line 65 separates required fact/event updates; line 101 adds distractors; line 114 validates shape; line 126 retries and retains best attempt; line 200 reconstructs fidelity; line 229 merges individual sessions.
- `generator_v2/prompts/conversation_system.txt:18`: today's choices and no schedules; line 23 forbids absent topics; line 27 specifies required updates; line 43 describes distractor small talk; line 47 prohibits new lasting facts/backstory; line 52 permits daily texture.
- `generator_v2/prompts/validator_system.txt:7`: closed-list topic extraction; line 11 allows `other` for past/future/person/question references; line 15 specifies statement/date checks; line 26 defines lasting inventions and permitted everyday details.
- `experiments/config.py:6`: experiment data directory.
- `experiments/run.py:29`: v2 loader dispatch.
- `experiments/benchmarks/v2.py:16`: reads curated questions; line 17 sorts sessions; lines 20–24 preserve grading/category/date/metadata.
- `experiments/core/runner.py:73`: all sessions ingested before answers; line 99 answers supplied questions; line 150 invokes common judge; line 155 also records strict score; line 238 uses judge headline accuracy.
- `experiments/core/types.py:51`: date prefix for method prompts.
- `experiments/core/grading.py:48`: exact/date checks; line 65 builds judge prompt with question, answer type, reference, accepted alternatives, and prediction.

`native_world_checks.json.source_sha256` fingerprints the generator, source prompts, all five persona YAMLs, ladder, and relevant loader/runner/grading code. `flagged_sessions.json` preserves flagged raw evidence without replacing it with review interpretations. `flag_reviews.json` records the separate review judgments and limitations.
