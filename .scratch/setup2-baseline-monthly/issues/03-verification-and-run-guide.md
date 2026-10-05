# Verify and document the monthly baseline setup

Type: task
Status: resolved

Install minimal missing dependencies in the existing project-local validation
environment, run meaningful offline tests using real vector libraries, exercise
monthly dry runs on all five current users, and document install/run/resume
commands plus TiMem's retrieval profile. Do not launch a paid benchmark.

## Answer

Installed Qdrant 1.19.1 in the existing project-local validation environment;
all 110 installed packages pass dependency consistency checks. Reduced TiMem's
benchmark requirements to the dependencies used by its adapter, preserving the
native application's full requirements separately. Added install, preview, run,
parallel-user, output, timing and resume documentation.

51 offline tests passed in 11.47 seconds; all four full-dataset CLI previews match.
Five users have 24 checkpoints each, with 1,000 distinct questions and 11,721
cumulative answers per method. Full-history contexts fit the configured estimate.
All 4,055 protected original files remain unchanged. No paid model calls occurred.

Native thread/event-loop checks stalled inside the restricted sandbox and passed
outside it with process/thread IPC available. The stalled task-owned diagnostic
processes were stopped. Existing dependency deprecation warnings were not expanded
into unrelated migrations. The saved report is `validation.json` in this directory.
