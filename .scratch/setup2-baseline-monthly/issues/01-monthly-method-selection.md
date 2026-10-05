# Reuse the monthly evaluation runner for all methods

Type: task
Status: resolved

Add method selection, method-specific manifests and summaries, and optional
process parallelism across users. Keep the default memory_v4 behavior, schedule,
judge, and checkpoint recovery barriers intact.

## Answer

Added lazy method selection, method/profile-aware manifests and summaries, and
optional spawned user processes. The original per-user checkpoint logic is reused;
all original 21 setup tests still pass. Baseline output directories are independent
of the default memory_v4 directory. Manifest checks refuse cross-method reuse.
All four CLI previews produce identical schedules for the canonical dataset.
