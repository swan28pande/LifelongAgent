# Resolve stash restoration conflicts

Restore the saved, uncommitted v2 dataset work after the completed rebase and push.
Only the eight conflicted `qa_pairs.json` files need resolution. Preserve the
remaining staged and untracked work, the current branch, and the saved stash.

## Acceptance

- No unmerged paths remain.
- Restored selections exactly match their already-restored QA pools.
- Canonical and experiment selections remain byte-identical and valid JSON.
- No new commit, push, or stash deletion is performed during recovery.

## Authorized follow-up

The user subsequently requested committing and pushing their saved local dataset,
and clarified the scope as `datasets/v2/` only. Commit the existing local bytes
from this directory and push normally to `origin/main`. Keep other staged work
outside this commit and preserve the backup stash.
