"""LoCoMo (ACL 2024): 10 two-person conversations. Category 5 (adversarial) is excluded, following
the Mem0/Zep protocol; category ids are mapped to names explicitly (see CATEGORY_NAMES)."""

import json

from benchmarks.locomo.common import (CATEGORY_NAMES, LOCOMO_PATH, get_sessions,
                                      session_to_turns)

from ..core.types import Instance, Question, Session, Turn

INCLUDED_CATEGORIES = (1, 2, 3, 4)


def load(n: int | None = None) -> list[Instance]:
    with open(LOCOMO_PATH) as f:
        convs = json.load(f)[:n] if n else json.load(f)
    out = []
    for conv in convs:
        cid = conv["sample_id"]
        sessions = [Session(id=f"s{i}", date=date, turns=[Turn(t["speaker"], t["text"]) for t in session_to_turns(turns)])
                    for i, (date, turns) in enumerate(get_sessions(conv), 1)]
        questions = [Question(
            id=f"{cid}_q{i}", question=q["question"], answer=str(q["answer"]), accept=[str(q["answer"])],
            answer_type="free_text", grading="locomo_f1", category=CATEGORY_NAMES[q["category"]],
            meta={"category_id": q["category"], "evidence": q.get("evidence", [])},
        ) for i, q in enumerate(conv["qa"]) if q.get("category") in INCLUDED_CATEGORIES and "answer" in q]
        out.append(Instance(id=cid, sessions=[s for s in sessions if s.turns], questions=questions))
    return out
