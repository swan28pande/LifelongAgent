# Research plan: what remains if full context solves the current task?

Date: 2026-09-29. Root `CONTEXT.md` is absent and `docs/adr/` has no ADRs; current terminology comes from `memory_v3/`, the dataset generator, and earlier research artifacts. No model runs are authorized in this investigation.

## Decision to support

Decide whether the project should continue as a memory architecture study, narrow to dated preference transitions, or change the target to longitudinal personalization and agent behavior. Specify the minimum evaluation that could distinguish the resulting claim from a strong full-context model and existing systems.

## Working assumption and definitions

- Assume a strong model given the complete 60-day history answers the existing synthetic questions well.
- **Research contribution:** an experimentally demonstrated capability or tradeoff not already explained by the reader model, prompt length, or a standard retrieval baseline.
- **Current dataset:** one synthetic user, three generated preference schedules, 60 conversations, 67 questions; plus the repo's one-dialogue LoCoMo check.
- **Online evaluation:** after each session, answer or act using only information available by that time, including later corrections and changes.

## Subquestions

1. Which capabilities do the current generator and scorer directly test? What shortcuts, missing cases, and sample-size limits remain?
2. Which adjacent lines of work already study evolving preferences, personalization in action, continual feedback, realistic long-horizon memory, and temporal correction?
3. Under full-context saturation, which precise claims remain testable: scaling/cost, online updates, evidence fidelity, calibrated predictions, user-controlled memory, or downstream task quality?
4. What benchmark and matched baselines would falsify each candidate claim?

## Evidence sought

- Local code, data, and archived results for dataset claims; no assumption that generated ground truth matches every conversation statement.
- Primary benchmark/method papers and official dataset repositories for external work. Record task, model, data scale, available artifact, and limits.
- Same-model, same-prompt comparisons where possible. Never rank scores from different scorers or datasets.

## Counter-evidence and gap check

- Strong full-context and simple chronological baselines may solve the proposed task.
- Existing preference-evolution or interactive benchmarks may already cover the claimed novelty.
- More complex memory may increase cost and error without improving user decisions.
- Synthetic trajectories may reward schedule inference but not natural preference learning.
- Search alternate terms: user modeling, recommender systems, continual preference learning, personalization, event sourcing, temporal databases, online learning, feedback, proactive agents, and procedural memory.

## Stopping conditions

Stop when the current dataset's supported and unsupported claims are explicit, at least several distinct related-work families are checked against primary sources, two or three falsifiable research options are defined, and unresolved novelty/performance questions are marked as such.
