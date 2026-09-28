"""One judge and one set of string checks for every method, published with the results."""

from __future__ import annotations

import json
import re

from dateutil import parser as dateparser

from .types import Question

JUDGE_SYSTEM = """\
You grade answers from a personal assistant with long-term memory against a reference.
Judge only whether the system's answer is correct; ignore style and verbosity.

Rules by answer type:
- value: RIGHT if the system's final answer is the reference value (paraphrases count).
  WRONG if it names a different value, or hedges between several values.
- date: RIGHT if the system gives one of the ACCEPTED dates, in any format. WRONG otherwise.
- yes_no: the yes/no must match the reference. WRONG if the explanation contradicts it.
- free_text: RIGHT if it captures the key content of the reference.
  * causes: the same cause, in any words.
  * routines/patterns: the same structure (which values, how many days each, or which
    weekdays). An anchor date is not required.
  * durations: within about 10% of the reference number of days.
  * histories and lists: every item of the reference, no wrong items.
- abstain: RIGHT only if the system says it does not know, it was never mentioned, or
  (for "why" questions) that no reason was given. WRONG if it states or invents an answer.
If the reference has an answer and the system says it does not know, it is WRONG.

Return only JSON: {"verdict": "RIGHT" | "WRONG", "why": "<one short sentence>"}"""

MONTHS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
DATE_RE = re.compile(rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"
                     rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\s+\d{{4}}\b", re.I)


def dates_in(text: str) -> set[str]:
    out = set()
    for m in DATE_RE.finditer(text):
        try:
            out.add(dateparser.parse(m.group(0), fuzzy=True).date().isoformat())
        except (ValueError, OverflowError):
            pass
    return out


def strict_check(q: Question, pred: str) -> bool | None:
    """String check for exact/date items; None where only the judge applies."""
    if q.grading == "date_exact":
        return bool(dates_in(pred) & set(q.accept))
    if q.grading == "exact":
        p = pred.lower()
        return any(re.search(r"\b" + re.escape(a.lower()) + r"\b", p) for a in q.accept)
    return None


def locomo_f1(q: Question, pred: str) -> float | None:
    if q.grading != "locomo_f1":
        return None
    from benchmarks.locomo.common import score_answer
    return float(score_answer(pred, q.answer, q.meta["category_id"]))


def judge_prompt(q: Question, pred: str) -> str:
    return (f"QUESTION: {q.question}\nANSWER TYPE: {q.answer_type}\n"
            f"REFERENCE: {q.answer}\nACCEPTED: {json.dumps(q.accept)}\n"
            f"SYSTEM ANSWER: {pred}")
