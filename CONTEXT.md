# Lifelong memory evaluation

Synthetic life histories test whether a personal memory system can retain facts,
recognize changing routines, and answer questions as information arrives over time.

## Language

**World state**:
The hidden life history containing the actual facts and daily choices. A value in
world state is not necessarily observable in conversation.

**Observation**:
A fact or daily choice disclosed in a conversation on its stated date.

**Effective date**:
The date a fact, event, or routine became true, which may precede its disclosure.

**Target date**:
The historical date or period a question asks about.

**Probe date**:
The original date a question is introduced, limiting the information available for
its answer. "Current" refers to this date even when the question is repeated later.

**Checkpoint**:
The end of a calendar month, or the last available day of a partial month, at which
new conversations have been ingested and accumulated questions are evaluated.

**Retrospective question**:
A distinct question introduced at a later probe to test retention. A repetition of
an existing question keeps the existing question's probe date and reference answer.

**Evidence days**:
Conversation dates whose observations support an answer. They may follow a target
date when the user discloses a historical fact retrospectively.

**Accepted answers**:
Reference alternatives that reflect both the hidden truth and uncertainty left by
the available observations.
