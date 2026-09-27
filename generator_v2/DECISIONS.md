# Decisions

One line per decision where the spec was ambiguous. Phase in brackets.

## Semantics (defaults recommended before Phase 1; approved with "go")

- [1] No retractions. Everything the user tells the agent is true; facts change (set/add/remove) but are never stated wrongly or corrected. The `retractions` knob and the `retraction` QA type were removed.
- [1] Temporary regimes: the reversion resumes the base regime's anchor (the cycle continues as if uninterrupted). A temporary shift plus its reversion counts as one shift on the ladder.
- [1] The 60% guard is kept strict. `exception_vs_shift`, `distractor_probe`, `reversion`, `other_person` and `abstention` all include positive or answerable controls, so "no" / "not known" is never the constant answer.
- [1] Exceptions are placed by the simulator from a per-domain `exception_pool` (explicit `exceptions:` are still honoured). Rate = exceptions / (session days × domains).
- [1] `do_not_mention` means "the user changes behaviour and gives no reason; do not invent one". There is no hidden-cause visibility.
- [1] Outputs in `datasets/v2/` are not committed (about 2 MB per user); regenerate them with `simulate`.

## Spec and schema

- [1] The persona YAML may omit `difficulty`; the loader fills it from `difficulty_ladder.yaml`, and rejects a copy that differs.
- [1] Every fact change, event state, regime and distractor has an id (explicit or `entity.i` / `chain.i` / `domain.ri`). `cause.event_ref` points at an id; a bare `chain_id` means the chain's first state. `cause.day` is derived and optional.
- [1] Lag 0 is allowed (U1), so the invariant is cause day ≤ regime start, with start == cause day + lag.
- [1] Reversions are created by the simulator (`<regime>.revert`), not written in YAML. Temporary `end` is inclusive. A reversion must last ≥ 14 days, as must every regime.
- [1] Cycles use 7-day blocks counted from the regime start, not ISO calendar weeks. `nested` is a list of sub-rules cycled in `block_days` blocks from the same anchor.
- [1] `conditional` uses the key `condition` (YAML parses a bare `on` as a boolean). Cases may be strings or nested rules; `default` is allowed.
- [1] Weather comes from a small city × month table in `rules.py`, with a season fallback for unknown cities. The U5 persona uses only `is_workday` and `season`, so a move never changes a value without a regime.
- [1] Added `other_people` (with `confusable_with` a user fact), a regime `label`, domain `noun`, fact `noun` and event `title`. The ladder lists other people for U4 as well as U5 (the QA list includes U4).
- [1] Fact `text` accepts a `{value}` placeholder.
- [1] "Fact changes" counts only changes after day 1; initial values don't count.
- [1] `rule_types` in the ladder is `required ⊆ used ⊆ allowed`, over top-level regime rules.
- [1] The spec asked for `simulate() -> list[DayState]`; it returns a `WorldState` holding `days` plus the resolved regimes and all statements, which QA and checks need.

## Simulation

- [1] Each sampling step has its own RNG stream (`seed:sessions`, `seed:mentions`, …), so changing one step doesn't reshuffle the others.
- [1] Day 1 always has a session. Explicit exception days are forced to have one.
- [1] Required statements move to the next session day, at most 4 per session (3 for day-1 background facts). Causes are scheduled first. `retrospective` = stated after the effective day, and only happens when a statement is moved.
- [1] Mentions are sampled first (this is the measured `p_mention`), then forced for exceptions, for the first session of a new regime (with `p_regime_start_mention`, 1.0 for U1–U3, 0.9 for U4, 0.85 for U5), and topped up so every regime is observed ≥ 5 times.
- [1] Exceptions are never within 3 days of a regime boundary, never within 3 days of another exception in the same domain, and never equal to the regular value.
- [1] Cause instructions go on the first non-exception observation of each new regime: explicit → `link_explicitly`, implicit → `mention_without_link`, uncaused → `do_not_mention`. An implicit cause also gets `mention_without_link` on the day it is stated.
- [1] Tolerances: counts exact, visibility mix ±0.10, p_session / p_mention ±0.05, exception rate ±0.02.
- [1] Distractor proximity uses ± the ladder's maximum lag (10 when the ladder allows 0). U5 has one deliberate confounder (d445, next to the uncaused reading shift on d455).

## QA

- [1] Every question is asked on day 731. `recall`, `fact_at_time` and `pattern_at_time` refer to an earlier date. `recency_distance_days` = 731 − latest evidence day.
- [1] Recall and prediction fill the total up to 390 (2:1), between minimums of 60/30 and maximums of 220/110. Other types are driven by the structure (one per shift, per temporary, per fact, …), with caps in `config.QA_TARGETS`.
- [1] Sampling is greedy and weighted: ×3 within ±7 days of a boundary, down-weighting repeated strata (regime × phase) and weekdays, and capping each answer at half of the group.
- [1] Guard: only groups of ≥ 4 questions; groups where every item is tagged `small_answer_space` (predictions for a constant current regime, asked only twice) are exempt.
- [1] `change_detection` accepts every date from the effective start to the first observation (tag `bounded`), since the memory cannot know more.
- [1] Fact questions use the entity as `domain`; events use `events`, other people use `other_people`.
- [1] Multi-valued sets on a date are `fact_at_time` items tagged `set`, not a separate type.

## Storylines

- [2] U2–U4 were drafted by hand in this session (like U1 and U5) rather than by `storyline.py` calling an LLM API, so there was no API cost. They are saved as `uN.draft.yaml` for review; `storyline.py` has not been built.
- [2] Domains differ per user: U1 coffee/outfit/lunch, U2 breakfast/workout/evening TV, U3 morning drink/lunch/commute/podcast, U4 breakfast/workout/work outfit/music/dinner, U5 exercise/commute/dinner/music/reading/bedtime.
- [2] Run a draft with `simulate --user uN.draft`; `--user all` only reads non-draft files.
