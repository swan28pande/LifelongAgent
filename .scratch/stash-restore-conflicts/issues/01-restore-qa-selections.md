# Restore saved QA selections

Type: task
Status: resolved

The rebase and push completed: `HEAD` equals `origin/main` at `515c218`.
`git stash pop` then conflicted in eight generated QA-selection files, leaving
`stash@{0}` (`eded87b`, "before rebase") available as a backup.

## Evidence

Each saved selection contains 50 distinct question IDs, and all 50 records exactly
match its working-tree `qa_pool.json`. Each upstream selection has zero exact
matches in the restored pool. Saved canonical and experiment selections are
byte-identical. The upstream change regenerates these selections; the saved work
also regenerates the corresponding pools and generator.

## Resolution

Restore the stage-3 saved selections and mark only those eight paths as resolved.
Verify JSON validity, pool membership, mirrored bytes, and Git conflict state.

## Answer

The eight saved selections were restored and staged. All ten canonical/experiment
selections contain 50 unique records exactly matching their pools; all five mirror
pairs are byte-identical. No unmerged paths remain. Recovery preserved the stash
and made no commit or push. See `../verification.json`.
