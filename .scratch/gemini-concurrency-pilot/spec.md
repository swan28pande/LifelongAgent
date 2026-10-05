# Gemini concurrency pilot

Status: concluded with a shortened probe after the user requested an ASAP answer

Determine whether TiMem, NaiveRAG and full-context evaluation can overlap without
material API slowdown, using the configured Vertex AI project, Flash backbone,
and Pro LLM judge. The user authorized live Gemini calls and requested the pilot.

## Scope and controls

- Reuse existing baseline implementations, canonical dataset, original probe dates,
  answer prompt and judge. Do not change agents, datasets or benchmark code.
- Prepare separate stores using user u5's complete first month (March 2026): 13
  observed conversations and six scheduled questions. TiMem uses real generation.
- Compare the same question sets with each baseline running alone versus all three
  together. Reverse the condition order on a second round to reduce ordering bias.
- Record actual HTTP attempt statuses, including retried 429/503 responses, logical
  request latency, successful request and token throughput, and final errors.
- Keep results, scripts and temporary files inside LifelongAgent. Persistent
  evidence belongs in `.research/gemini-concurrency-pilot/`.
- Run locally against the VM's Vertex AI project; distinguish API measurements from
  the VM's RAM/CPU capacity. This small workload cannot establish sustained capacity
  at 120 answer requests or 240 judge requests, or late-history full-context sizes.
- If useful, add a bounded late-history full-context comparison; label its workload
  separately rather than mixing it into the first-month measurements.
- Add one isolated API replay comparison at the proposed five-user settings:
  40 answers/80 judgments per method alone, then 120 answers/240 judgments together.
  Use captured real prompts and fixed real responses, at most 720 measured logical
  calls. These are workload replicas, not 15 independent user histories. Stop higher
  load if a stage exhausts retries. This tests bursts rather than sustained capacity.

## Completion criteria

Publish concrete solitary/concurrent latency and throughput measurements, observed
  throttling/retries and runtime. Explain the concurrency and prompt-size range
  actually tested. Do not claim unchanged speed for untested production loads.

## Outcome

The complete TiMem preparation did not finish: two original 180-second timeouts
were followed by two HTTP 504 responses under a bounded preparation override.
A separate same-prompt diagnostic added one 504 response. Preparation was stopped
at the user's request for an ASAP answer. A shortened direct API probe completed
12 calls using three concurrent prompt streams and the existing judge. Its TiMem
prompt explicitly uses partial completed nodes; its HTTP deadline is 20 seconds
with one attempt. Full-load capacity and end-to-end baseline execution remain
unverified. Evidence and interpretation: `.research/gemini-concurrency-pilot/RESULTS.md`.
