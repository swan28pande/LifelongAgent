"""The generator_v2 synthetic users (datasets/v2/<user>/): one instance per user, 50 curated questions."""

import datetime as dt
import json

from ..config import ROOT
from ..core.types import Instance, Question, Session, Turn


def load(users: list[str] | None = None) -> list[Instance]:
    base = ROOT / "datasets" / "v2"
    users = users or sorted(p.name for p in base.iterdir() if (p / "conversations.json").exists())
    out = []
    for user in users:
        conv = json.loads((base / user / "conversations.json").read_text())
        qa = json.loads((base / user / "qa_pairs.json").read_text())
        sessions = [Session(id=date, date=date, turns=[Turn(t["speaker"], t["text"]) for t in s["turns"]])
                    for date, s in sorted(conv["sessions"].items())]
        day1 = dt.date.fromisoformat(sessions[0].date)      # day 1 always has a session
        questions = [Question(
            id=q["id"], question=q["question"], answer=q["answer"], accept=q["accept"],
            answer_type=q["answer_type"], grading=q["grading"], category=q["type"],
            asked_on=(day1 + dt.timedelta(days=q["viewpoint_day"] - 1)).isoformat(),
            meta={"capability": q["capability"], "tags": q["tags"], "evidence_days": q["evidence_days"]},
        ) for q in qa]
        out.append(Instance(id=user, sessions=sessions, questions=questions, user=conv["name"]))
    return out
