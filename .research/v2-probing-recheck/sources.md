# Sources

Primary evidence: current `datasets/v2/u1` through `u5`, `generator_v2/qa.py`, `generator_v2/probing.py`, unchanged world/rule/source definitions, and `setup_2/` loader decisions. The baseline is the recorded artifacts and input hashes in `.research/v2-probing-audit/`. All checks are local and offline.

- [Root manifest](manifest.json) and [before/after comparison](input_comparison.json) preserve current and original SHA-256 hashes. The original probing payloads were read from Git commit `98a2674` only after verifying their bytes against the baseline hashes.
- [Non-recall input hashes](gold/input_sha256.json), [report](gold/report.md), and indexed primary-source excerpts in `gold/manual_transcript_checks.json`, `gold/broader_boundary_candidates.json`, and `gold/supplemental_issues.json` support the reference/citation/date findings.
- [Recall source inventory](recall/sources.json) and [report](recall/report.md) preserve independent raw-rule evaluation, exact session source locations, manual paraphrase support, missing-phase excerpts, and accepted/rejected boundary alternatives.
- [Source evidence](source/evidence.md) and [source list](source/sources.md) identify all individual sessions, aggregate/world records, old source review, 35 duplicate-file comparisons, canonical/duplicate QA payload comparisons, native checks, and the actual offline `setup_2/` loader inspection.
- [Current question findings](question_findings.md), [complete coverage](question_coverage.jsonl), and [primary evidence](evidence.md) link conclusions to updated locations, keeping observation limitations separate from wrong references and from source-contract exposures.
