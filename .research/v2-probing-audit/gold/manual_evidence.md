# Narrow transcript evidence and interpretation

The complete semantic-family checks use synthetic truth. These additional transcript checks target specific risks; they do not substitute saved validator flags for human review. The reproducible transcript extracts and source lines are stored in `manual_transcript_checks.json`.

Run `PYTHONDONTWRITEBYTECODE=1 python .research/v2-probing-audit/gold/manual_checks.py` after the main gold audit.

## Current hobby list

Source evidence:

- The January 2027 daily active/known hobby set contains bouldering, piano and trail running.
- The January 13 user reports sitting down at the piano yesterday.
- The January 22 user says they might practice piano tomorrow.
- The January 25 user explicitly dates taking up trail running to January 24, 2027.
- The April 24 user says they have officially decided to stop playing the piano, explaining that they no longer have time to find lessons after the move. The corresponding effective/stated world update is day 420.

Interpretation: January's omission of piano is a confirmed current-reference error, corroborated by the visible transcript. The acceptance hint repeats the same wrong list. This is not a case where synthetic truth is correct but the conversation simply never mentioned the retained item.

## Empty pets

Source evidence:

- `u2_p202603_0001`, `u2_p202608_0038`, and `u3_p202604_0007` cite day 1. The worlds have empty pet sets at their targets, but no negative/background pet statement.
- A user-turn scan finds no pet/cat/dog/animal vocabulary anywhere in each of those three primary prefixes, including u2 through August 31, 2026 and u3 through April 30, 2026.
- `u2_r202702_0188` subsequently knows the January 24, 2027 cat adoption. The recorded adoption does not say that this was Marcus's first pet, and the day-1 citation still has no pet statement. The negative/first-pet lexical scan through this retrospective probe returns no candidates.

Interpretation: `none` is correct synthetic truth, but it is not established by these citations or the recorded factual updates. Answerability requires an unstated closed-world convention. The lexical scan is not an exhaustive theorem about every possible paraphrase; the direct absence of a negative fact statement and unsuitable day-1 citation are established independently.

## Non-explicit change boundaries

Source evidence:

- Daniel's October 26, 2027 music turn describes spinning horn-heavy improvisational vinyl while sketching. It does not announce a new routine or date a switch. October 27 describes instrumental electronic soundscapes. The old cyclic rule also predicted jazz records on the first date.
- Priya's September 8, 2026 reading turn describes library suspense novels; September 12 still describes those novels. September 13 describes anthologies of verse. The first new-regime mention therefore matches the previous constant routine.
- Priya's February 8, 2027 music turn describes retro-futuristic electronic beats. February 14 describes running playlists. The old constant synthwave rule also predicted the first mention's music.

Interpretation: the five probing copies of these three changes have two separate limitations that overlap by ID: cited first observations do not demonstrate change; and bounded counterfactual starts preserve every declared choice through the probe while date_exact rejects the corresponding dates. The checker preserves exception base values, respects cause-lag limits, and tests the new cycle anchor. Rotating the weekly reading order is necessary for the day-197 alternative; postponing its fixed ordered rule alone does not work. This does not contradict the hidden world's specified effective start.

The counterfactuals hold later regimes and facts fixed. They are counterexamples to unique dating from declared observations, not a full regeneration of every scheduling/randomness constraint and not exhaustive re-judging of every later conversational paraphrase.

## Completion and citation checks

The two premature duration references were checked against their still-active regimes at the probe; their gold-only endpoints lie later. Three other durations close on or just before the probe but have their first successor observation in the next month. They are reported separately because their hidden-world totals are correct.

Forty preference-duration citations omit the successor transition; eight complete-list fact citations omit relevant removals; twenty-two at-time fact citations contain later updates to the same value. These are metadata diagnostics. Actual prefixes can contain the missing statements, and a later update can help a legitimate retrospective inference even when it is not direct evidence of membership at the earlier target.
