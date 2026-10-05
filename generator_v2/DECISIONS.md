# Decisions

One line per decision where the spec was ambiguous. Phase in brackets.

## Semantics (defaults recommended before Phase 1; approved with "go")

- [1] No retractions. Everything the user tells the agent is true; facts change (set/add/remove) but are never stated wrongly or corrected. The `retractions` knob and the `retraction` QA type were removed.
- [1] Temporary regimes: the reversion resumes the base regime's anchor (the cycle continues as if uninterrupted). A temporary shift plus its reversion counts as one shift on the ladder.
- [1] The 60% guard is kept strict. `exception_vs_shift`, `distractor_probe`, `reversion`, `other_person` and `abstention` all include positive or answerable controls, so "no" / "not known" is never the constant answer.
- [1] Exceptions are placed by the simulator from a per-domain `exception_pool` (explicit `exceptions:` are still honoured). Rate = exceptions / (session days × domains).
- [1] `do_not_mention` means "the user changes behaviour and gives no reason; do not invent one". There is no hidden-cause visibility.
- [1, repair] Canonical outputs land in `datasets/v2/`. QA/probe/fidelity commands synchronize the generated files into `experiments/data/synthetic_v2/`; API logs are kept separately.

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

- [1] Full-pool questions use viewpoint day 731. `recall`, `fact_at_time` and `pattern_at_time` refer to an earlier date. Monthly probes use their original probe date and recompute current references there; evaluation repetitions retain the original record.
- [1] Recall and prediction fill the total up to 390 (2:1), between minimums of 60/30 and maximums of 220/110. Other types are driven by the structure (one per shift, per temporary, per fact, …), with caps in `config.QA_TARGETS`.
- [1] Sampling is greedy and weighted: ×3 within ±7 days of a boundary, down-weighting repeated strata (regime × phase) and weekdays, and capping each answer at half of the group.
- [1] Guard: only groups of ≥ 4 questions; groups where every item is tagged `small_answer_space` (predictions for a constant current regime, asked only twice) are exempt.
- [1, repair] `change_detection` cites a choice distinguishing the old and new routines and waits until it is disclosed. Acceptance includes dates compatible with pre-probe choices, permitted cause lags and equivalent weekly rotations. Explicit announcements and condition-ending reversions retain their start-to-disclosure bound. `bounded` denotes date uncertainty, not a uniquely observable hidden start.
- [1] Fact questions use the entity as `domain`; events use `events`, other people use `other_people`.
- [1] Multi-valued sets on a date are `fact_at_time` items tagged `set`, not a separate type.
- [repair] Fact evidence includes relevant set/add/remove disclosures. Historical disclosures may follow the target date, but must precede the probe; no day-1 fallback is allowed. Complete-current lists are recomputed at the probe, including removals.
- [repair] Unobserved recall phases and unmentioned days within seven days of a transition are excluded. Full pattern references require observations covering their applicable phases and all described values. Completed durations cite the closing transition; ongoing durations explicitly say "as of".
- [repair] Monthly probes contain 200 questions per user, including all six genuine unknown-answer templates plus answerable controls. Sampling keeps proportional month allocation and explicit family coverage, and includes the final partial month.
- [repair] On October 3, 2026, the user confirmed that monthly questions remain historical. Predictions stay in the separate end-of-timeline QA pool and are deliberately excluded from monthly probes.
- [repair] Newly scheduled retrospective records are grounded at their own probe date. They differ from the monthly runner's repetitions, which freeze the earlier question's date, wording and gold.

## Storylines

- [2] U2–U4 were drafted by hand in this session (like U1 and U5) rather than by `storyline.py` calling an LLM API, so there was no API cost. They are saved as `uN.draft.yaml` for review; `storyline.py` has not been built.
- [2] Domains differ per user: U1 coffee/outfit/lunch, U2 breakfast/workout/evening TV, U3 morning drink/lunch/commute/podcast, U4 breakfast/workout/work outfit/music/dinner, U5 exercise/commute/dinner/music/reading/bedtime.
- [2] Run a draft with `simulate --user uN.draft`; `--user all` only reads non-draft files.

## Conversations

- [3] Vertex AI only: `gemini-3.5-flash` writes (temperature 0.9), `gemini-3.1-pro-preview` validates (temperature 0). Both are Gemini, so the validator is a stronger tier rather than a different family.
- [3] The writer sees only that day's state and the facts known before it; never regime labels, rules or earlier text, so it cannot leak a pattern.
- [3] The validator picks each topic's value from a closed option list (all values the persona can have), and the comparison with ground truth is exact, in code.
- [3] "Inventions" are lasting facts only (new people, pets, jobs, history such as "vegetarian for three years"). Everyday details (working from home today, plans for tonight) are allowed; flagging them made retries push conversations towards bland, form-like dialogue.
- [3] Up to 5 attempts per session; the attempt with the fewest failures is kept and flagged if it still fails. `first_try_failures` is kept for the fidelity report.
- [repair] Required updates/news/distractors may name an otherwise absent daily topic only to state the required fact. They do not permit a daily-choice report, activity history or discretionary follow-up. This resolves conflicts such as a required climbing injury with an unmentioned workout.
- [repair] Validation includes the session date and previously disclosed fact-change history. Retrospective updates and distractors give exact dates or unambiguous relative days; tonight's reported dinner/show/bedtime counts as today's choice.
- [repair] Source corrections retain the previous extraction/failures in `repair_validation`, together with the real validator result's date/model/prompt hash. LFS-placeholder logs retain previously recorded generation usage with explicit provenance; repair-validation usage is logged separately.

## Curated questions

- [3] Each user ships 50 curated questions (`qa_pairs.json`); the full generated pool (~390) is kept as `qa_pool.json`. Curation gives every type the user has at least one question, weights harder types (attribution, change detection, patterns) up, and within a type spreads the picks over answer types, answers, topics and tags, so yes/no and abstention stay balanced.
- [3] Questions name a routine by the items it involves ("the fitted t-shirt / oversized hoodie routine"), never by its rule, so a change or duration question cannot give away a pattern question's answer.
