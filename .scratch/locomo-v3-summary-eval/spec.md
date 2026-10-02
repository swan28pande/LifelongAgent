# One-conversation LoCoMo evaluation of the integrated summary reader

Status: completed and audited; all 199 answers and grades saved.

Evaluate the current `memory_v3_update/` on only `conv-26` from the existing LoCoMo
dataset: 19 sessions, 419 turns, all 199 questions. Reuse the previous evaluation's
Gemini 3.5 Flash answer model, Gemini 3.1 Pro preview judge, temperatures, four
workers, grading prompt, and adversarial abstention protocol.

Build a fresh store, hierarchical summaries, and the optional four-theme knowledge
documents before answering. Check every trace for exact planned dispatch and a
maximum of five actual retrieval attempts. Keep source files unchanged and put all
run artifacts, caches, and temporary files inside project `tmp/`.

Report judge accuracy, category results, retrieval behavior, and elapsed time.
Compare with the prior run on the same conversation; this single run does not
isolate summarization from extraction, distillation, or model variability.
