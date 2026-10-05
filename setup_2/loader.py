"""Read canonical conversations and monthly probes without modifying either."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

from experiments.core.types import Question, Session, Turn


DATA_DIR = Path(__file__).resolve().parents[1] / 'datasets' / 'v2'


@dataclass(frozen=True)
class Checkpoint:
    month: str
    cutoff: str
    sessions: tuple[Session, ...]
    questions: tuple[Question, ...]  # Cumulative; asked_on is the original date.
    new_question_ids: tuple[str, ...]


@dataclass(frozen=True)
class Timeline:
    id: str
    name: str
    checkpoints: tuple[Checkpoint, ...]
    input_hashes: dict[str, str]


def _date(value: str) -> dt.date:
    date = dt.date.fromisoformat(value)
    if date.isoformat() != value:
        raise ValueError(f'Expected an ISO date, got {value!r}')
    return date


def _json(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def load(data_dir: Path = DATA_DIR, users: list[str] | None = None) -> list[Timeline]:
    """Load all checkpoints, including a final partial month and empty months.

    Conversations are the only ingestion input. Questions enter the cumulative
    set in their scheduled month and retain all original judging/date metadata.
    Structural inconsistencies fail explicitly; semantic dataset errors are not
    repaired or filtered here.
    """
    base = Path(data_dir)
    selected = users if users is not None else sorted(
        p.name for p in base.iterdir() if (p / 'probing_questions.json').is_file()
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Select at least one user, without duplicates')
    timelines = []
    for uid in selected:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', uid):
            raise ValueError(f'Invalid user ID: {uid!r}')
        conv, conv_hash = _json(base / uid / 'conversations.json')
        probes, probe_hash = _json(base / uid / 'probing_questions.json')
        if probes['user_id'] != uid:
            raise ValueError(f'{uid}: probe user_id does not match directory')
        start = _date(probes['start_date'])
        days = probes['num_days']
        if not isinstance(days, int) or isinstance(days, bool) or days < 1:
            raise ValueError(f'{uid}: num_days must be a positive integer')
        end = start + dt.timedelta(days=days - 1)
        sessions_by_month: dict[str, list[Session]] = {}
        for date, session in sorted(conv['sessions'].items()):
            when = _date(date)
            if not start <= when <= end:
                raise ValueError(f'{uid}: session {date} is outside the history')
            if session.get('date', date) != date:
                raise ValueError(f'{uid}: session date disagrees with its key {date}')
            turns = [Turn(t['speaker'], t['text']) for t in session['turns']]
            if not turns or any(not t.speaker or not isinstance(t.text, str) for t in turns):
                raise ValueError(f'{uid}: invalid turns on {date}')
            sessions_by_month.setdefault(date[:7], []).append(
                Session(date, date, turns, session.get('time_of_day'))
            )

        new_by_month: dict[str, list[Question]] = {}
        seen_questions, seen_months = set(), set()
        for probe in sorted(probes['probes'], key=lambda p: p['probe_date']):
            date = probe['probe_date']
            when = _date(date)
            month = probe['probe_month']
            month_end = when.replace(day=calendar.monthrange(when.year, when.month)[1])
            if month != date[:7] or month in seen_months or when != min(month_end, end):
                raise ValueError(f'{uid}: invalid or duplicate monthly probe {date}')
            if not start <= when <= end or probe['viewpoint_day'] != (when - start).days + 1:
                raise ValueError(f'{uid}: invalid probe viewpoint on {date}')
            seen_months.add(month)
            rows = probe['questions']
            if probe.get('num_questions', len(rows)) != len(rows):
                raise ValueError(f'{uid}: probe count mismatch on {date}')
            for q in rows:
                if (not isinstance(q['id'], str) or not q['id']
                        or not isinstance(q['question'], str) or not isinstance(q['answer'], str)
                        or not isinstance(q['accept'], list)
                        or any(not isinstance(a, str) for a in q['accept'])
                        or q['answer_type'] not in ('value', 'date', 'free_text', 'yes_no', 'abstain')
                        or q['grading'] not in ('exact', 'date_exact', 'llm_judge')):
                    raise ValueError(f'{uid}: invalid question metadata')
                if q['id'] in seen_questions:
                    raise ValueError(f'{uid}: duplicate question ID {q["id"]}')
                if (q.get('probe_date', date) != date or q.get('probe_month', month) != month
                        or q['viewpoint_day'] != probe['viewpoint_day']):
                    raise ValueError(f'{uid}: question/probe date mismatch for {q["id"]}')
                seen_questions.add(q['id'])
                meta = {k: v for k, v in q.items() if k not in {
                    'id', 'question', 'answer', 'accept', 'answer_type', 'grading', 'type'
                }}
                new_by_month.setdefault(month, []).append(Question(
                    id=q['id'], question=q['question'], answer=q['answer'],
                    accept=list(q['accept']), answer_type=q['answer_type'],
                    grading=q['grading'], category=q['type'], asked_on=date, meta=meta,
                ))
        if probes.get('total_questions', len(seen_questions)) != len(seen_questions):
            raise ValueError(f'{uid}: total question count mismatch')
        if probes.get('num_probes', len(seen_months)) != len(seen_months):
            raise ValueError(f'{uid}: probe count mismatch')

        checkpoints, cumulative = [], []
        cursor = start.replace(day=1)
        while cursor <= end:
            month = cursor.strftime('%Y-%m')
            month_end = cursor.replace(day=calendar.monthrange(cursor.year, cursor.month)[1])
            introduced = new_by_month.get(month, [])
            cumulative.extend(introduced)
            checkpoints.append(Checkpoint(
                month, min(month_end, end).isoformat(),
                tuple(sessions_by_month.get(month, [])), tuple(cumulative),
                tuple(q.id for q in introduced),
            ))
            cursor = month_end + dt.timedelta(days=1)
        timelines.append(Timeline(uid, conv['name'], tuple(checkpoints), {
            'conversations.json': conv_hash, 'probing_questions.json': probe_hash,
        }))
    return timelines


def describe(timelines: list[Timeline], through: str | None = None) -> dict:
    """An offline schedule preview; does not construct an agent or call a model."""
    users = {}
    for timeline in timelines:
        months = [c for c in timeline.checkpoints if through is None or c.month <= through]
        users[timeline.id] = [{
            'month': c.month, 'cutoff': c.cutoff, 'new_sessions': len(c.sessions),
            'new_questions': len(c.new_question_ids), 'cumulative_questions': len(c.questions),
        } for c in months]
    return {
        'protocol': 'monthly_cumulative_original_probe_date',
        'users': users,
        'sessions': sum(m['new_sessions'] for rows in users.values() for m in rows),
        'answer_instances': sum(m['cumulative_questions'] for rows in users.values() for m in rows),
    }
