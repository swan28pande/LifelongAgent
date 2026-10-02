# Domain docs

This repo uses a single-context layout: `CONTEXT.md` at the root and ADRs under `docs/adr/`.

## Before exploring

Read `CONTEXT.md` if it exists, then read ADRs that touch the area you are about to work in.

If these docs do not exist, proceed silently. The `/domain-modeling` skill creates them when terms or decisions are resolved.

## Use the glossary's vocabulary

When naming a domain concept in an issue, proposal, hypothesis, or test, use the term defined in `CONTEXT.md`. If the term is missing, reconsider whether it fits the project or note the gap for `/domain-modeling`.

## Flag ADR conflicts

If your output contradicts an existing ADR, name the ADR and explain the reason for reopening the decision.
