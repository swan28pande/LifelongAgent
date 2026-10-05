# Regenerate, synchronize, and verify

Type: task
Status: resolved

Synchronize canonical dataset JSON with experiment copies, validate all generated
questions, and record a before/after resolution for the updated audit findings.

## Answer

QA/fidelity commands regenerate and synchronize the canonical/experiment outputs.
40 file pairs match, five world states remain byte-identical, and all 1,000 questions
are validated. 90 generator and 21 monthly setup tests pass. Agent and shared judge
implementations were not edited. See `.research/v2-dataset-fixes/report.md`.
