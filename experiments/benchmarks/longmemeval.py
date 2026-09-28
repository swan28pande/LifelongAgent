"""LongMemEval-S (ICLR 2025), cleaned release. Every question has its own ~50-session history,
so each question is its own instance. `sample` draws a stratified, seeded subset by question
type (abstention questions stay in proportion) because full runs ingest ~25k sessions."""

import datetime as dt
import json
import random
import subprocess
from collections import defaultdict

from ..config import DATA_DIR
from ..core.types import Instance, Question, Session, Turn

URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
PATH = DATA_DIR / "longmemeval_s_cleaned.json"


def _date(raw: str) -> tuple[str, str]:
    """'2023/05/20 (Sat) 02:21' → ('2023-05-20', '02:21')."""
    d, _, rest = raw.partition(" (")
    return dt.datetime.strptime(d, "%Y/%m/%d").date().isoformat(), rest.split(") ")[-1]


def load(sample: int | None = None, seed: int = 0) -> list[Instance]:
    if not PATH.exists():
        PATH.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSL", "-o", str(PATH), URL], check=True)
    data = json.loads(PATH.read_text())
    if sample:
        by_type = defaultdict(list)
        for e in data:
            by_type[e["question_type"]].append(e)
        rng = random.Random(seed)
        exact = {t: sample * len(v) / len(data) for t, v in by_type.items()}
        quota = {t: int(x) for t, x in exact.items()}
        for t in sorted(exact, key=lambda t: exact[t] - quota[t], reverse=True)[: sample - sum(quota.values())]:
            quota[t] += 1
        data = [e for t in sorted(by_type) for e in rng.sample(by_type[t], quota[t])]
    out = []
    for e in data:
        sessions = []
        for sid, raw, turns in zip(e["haystack_session_ids"], e["haystack_dates"], e["haystack_sessions"]):
            date, time = _date(raw)
            sessions.append(Session(id=sid, date=date, time=time,
                                    turns=[Turn(t["role"], t["content"]) for t in turns]))
        sessions.sort(key=lambda s: (s.date, s.time or ""))
        abstain = e["question_id"].endswith("_abs")
        q = Question(
            id=e["question_id"], question=e["question"], answer=str(e["answer"]), accept=[str(e["answer"])],
            answer_type="abstain" if abstain else "free_text", grading="llm_judge",
            category=e["question_type"] + ("_abs" if abstain else ""),
            asked_on=_date(e["question_date"])[0],
        )
        out.append(Instance(id=e["question_id"], sessions=sessions, questions=[q]))
    return out
