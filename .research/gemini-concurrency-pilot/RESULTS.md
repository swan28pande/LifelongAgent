# Gemini pilot observations

The user requested an ASAP answer while preparation was still running. The longer comparisons were stopped and replaced with a 12-request API probe.

The shortened probe took 38.75 seconds, including local initialization. HTTP statuses: {'200': 12}.

| Stage | Prompt stream | Alone (s) | Concurrent (s) |
| --- | --- | ---: | ---: |
| answer | timem | 3.35 | 2.89 |
| answer | naive_rag | 3.05 | 2.37 |
| answer | full_context | 4.15 | 5.26 |
| judge | timem | 3.30 | 2.87 |
| judge | naive_rag | 2.78 | 2.66 |
| judge | full_context | 3.13 | 2.28 |

## Interpretation

Gemini accepted three simultaneous answer calls and three simultaneous judge calls without a 429 response. This does not establish capacity or equal speed at the proposed 120-answer/240-judge load. There is one observation per stream per condition, and solitary requests precede concurrent requests.

TiMem preparation hit two 180-second timeouts with the existing settings. Bounded preparation clients received two HTTP 504 responses before a retry succeeded; a separate diagnostic also received one 504 response. These failures occurred before concurrent measurements and do not establish concurrency as their cause.

The quick TiMem prompt uses completed L1/L2/L3 nodes from its interrupted first-month store. Its five-level hierarchy was not finalized. The probe explicitly captures that partial retrieval context; it does not claim that the complete TiMem monthly benchmark works. The other two prompt streams use their complete first-month stores.

The quick API requests use a 20-second timeout and one SDK attempt to bound waiting. Original benchmark code and settings were not edited. The larger same-checkpoint, late-history and 120/240 request profiles were not run.

Protected source files unchanged: True. Canonical user input hashes unchanged: True.

Raw evidence: quick_api/events.jsonl and summary.json; u5_march_20261004_{a,b,c}/; stream_diagnostic/. Methodology and prepared reproduction scripts are kept alongside these artifacts.
