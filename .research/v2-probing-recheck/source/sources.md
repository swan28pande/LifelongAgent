# Sources and provenance

Primary current evidence is the five canonical user directories under `datasets/v2/`, including every `sessions/*.json`, aggregate conversations, worlds, probing questions, QA pools, fidelity, stats, and model-log status. Corresponding JSON inputs under `experiments/data/synthetic_v2/` provide independent copy checks. Hashes and exact file locations are retained in [session coverage](session_coverage.jsonl), [copy comparison](copy_comparison.json), [pool provenance](pool_provenance.json), and [coverage](coverage.json).

The comparison baseline is `.research/v2-probing-audit/source/session_coverage.jsonl`, `copy_comparison.json`, `flag_reviews.json`, `native_world_checks.json`, and `report.md`, plus the original root audit's coverage findings. The original raw-transcript interpretations are reused only after exact session hashes, each saved failure, and every quoted turn are checked. Earlier citation IDs are retained only under explicitly named `baseline_*` fields.

The reproduced check code is `.research/v2-probing-audit/source/audit_source.py` and `check_native_world.py`, imported by [recheck_source.py](recheck_source.py) with output paths reassigned into this directory. Their original files and outputs remain unchanged. Current native-code/persona hashes are in [native_world_checks.json](native_world_checks.json); differences from baseline are checked in [native_source_hash_comparison.json](native_source_hash_comparison.json).

Coverage provenance is read directly from current `generator_v2/qa.py`, especially prediction creation at line 215, current patterns at line 234, and empty-evidence unknowns at line 545; and current `generator_v2/probing.py`, especially calendar-boundary stopping at line 39, empty-evidence rejection at line 84, beyond-history target rejection at line 119, and absent-month rejection at line 122. The default output root is `generator_v2/config.py:9`.

The monthly integration check invokes current `setup_2/loader.py` with canonical inputs and reads the month execution barrier at `setup_2/runner.py:395`, `:402`, `:409`, and `:414`. This does not invoke the memory agent, embeddings, or judge. The unchanged legacy evaluation loader/runner remain a separate protocol.

No web material, model output newly generated for this recheck, paid API, external write, or root `/tmp` file was used. All new artifacts in this workstream are under `.research/v2-probing-recheck/source/`.
