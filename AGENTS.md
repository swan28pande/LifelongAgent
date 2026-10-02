## Agent skills

### Issue tracker

Track issues and specs as local markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Use a single-context layout: root `CONTEXT.md` and `docs/adr/`. See `docs/agents/domain.md`.

Before designing, planning, or making non-trivial changes:
- Read `CONTEXT.md` for project terminology, concepts, and domain assumptions.
- Read relevant ADRs in `docs/adr/` before changing or revisiting architectural decisions.
- Preserve and update this project knowledge when a decision or domain concept materially changes.

### Research artifacts

For substantial literature, technical, or scientific investigations:

- Use the `research-engineering` skills when appropriate.
- Store persistent investigation artifacts under `.research/<topic>/`.
- Use `CONTEXT.md` for project terminology and research concepts.
- Read relevant ADRs before making claims that depend on settled project decisions.
- Keep source evidence separate from interpretation.