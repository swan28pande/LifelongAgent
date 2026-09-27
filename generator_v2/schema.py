"""Pydantic models for persona specs, the difficulty ladder, world state, QA and sessions."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Difficulty ladder ───────────────────────────────────────────────

class IntRange(Strict):
    min: int
    max: Optional[int] = None  # None = unbounded ("4+")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> Any:
        if isinstance(v, int):
            return {"min": v, "max": v}
        if isinstance(v, (list, tuple)):
            return {"min": v[0], "max": v[1]}
        return v

    def contains(self, x: int) -> bool:
        return x >= self.min and (self.max is None or x <= self.max)

    def __str__(self) -> str:
        if self.max == self.min:
            return str(self.min)
        return f"{self.min}-{self.max}" if self.max is not None else f"{self.min}+"


class RuleTypes(Strict):
    required: list[str] = []
    allowed: list[str]


class Difficulty(Strict):
    level: int
    preference_domains: int
    rule_types: RuleTypes
    shifts_per_domain: IntRange
    visibility_mix: dict[Literal["explicit", "implicit", "uncaused"], float]
    lag_days: IntRange
    temporary_shifts: IntRange
    cross_domain_causes: int
    distractors: int
    exception_rate: float
    p_session: float
    p_mention: float
    p_regime_start_mention: float
    fact_changes: IntRange
    wording: str
    other_people: bool


class Ladder(Strict):
    users: dict[str, Difficulty]


# ── Rules ───────────────────────────────────────────────────────────

class ConstantRule(Strict):
    type: Literal["constant"]
    value: str


class WeeklyAlternateRule(Strict):
    type: Literal["weekly_alternate"]
    values: list[str] = Field(min_length=2)


class Segment(Strict):
    value: str
    days: int = Field(ge=1)


class NDayCycleRule(Strict):
    type: Literal["n_day_cycle"]
    segments: list[Segment] = Field(min_length=2)


class DayOfWeekRule(Strict):
    type: Literal["day_of_week"]
    map: dict[str, str]

    @field_validator("map")
    @classmethod
    def _all_weekdays(cls, v: dict[str, str]) -> dict[str, str]:
        v = {k.lower()[:3]: val for k, val in v.items()}
        missing = [d for d in WEEKDAYS if d not in v]
        if missing or len(v) != 7:
            raise ValueError(f"day_of_week map needs exactly mon..sun, missing {missing}")
        return v


class NestedRule(Strict):
    """Cycles through sub-rules in blocks of `block_days`, anchored to the regime start."""
    type: Literal["nested"]
    block_days: int = 7
    blocks: list["Rule"] = Field(min_length=2)


class ConditionalRule(Strict):
    """Picks a case by a world-context variable: is_workday, season or weather."""
    type: Literal["conditional"]
    condition: Literal["is_workday", "season", "weather"]
    cases: dict[str, Union[str, "Rule"]]

    @field_validator("cases", mode="before")
    @classmethod
    def _stringify_keys(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k).lower(): val for k, val in v.items()}
        return v


Rule = Annotated[
    Union[ConstantRule, WeeklyAlternateRule, NDayCycleRule, DayOfWeekRule, NestedRule, ConditionalRule],
    Field(discriminator="type"),
]
NestedRule.model_rebuild()
ConditionalRule.model_rebuild()


# ── Persona spec ────────────────────────────────────────────────────

class FactChange(Strict):
    id: Optional[str] = None
    day: int = Field(ge=1)
    op: Literal["set", "add", "remove"]
    value: str
    cause_event: Optional[str] = None
    reason: Optional[str] = None
    text: Optional[str] = None


class FactEntity(Strict):
    entity: str
    noun: Optional[str] = None
    cardinality: Literal["single", "multi"]
    changes: list[FactChange]

    @property
    def label(self) -> str:
        return self.noun or self.entity.replace("_", " ")


class EventState(Strict):
    id: Optional[str] = None
    day: int
    status: Literal["planned", "happened", "cancelled"]
    text: str


class EventChain(Strict):
    chain_id: str
    title: str
    states: list[EventState] = Field(min_length=1)


class Cause(Strict):
    event_ref: str
    lag_days: int = Field(ge=0)
    visibility: Literal["explicit", "implicit"]
    day: Optional[int] = None  # derived from event_ref when omitted


class Regime(Strict):
    id: Optional[str] = None
    start: int
    end: Optional[int] = None
    temporary: bool = False
    rule: Rule
    cause: Optional[Cause] = None  # None on a non-initial regime = uncaused
    label: Optional[str] = None
    revert_reason: Optional[str] = None


class ExceptionDay(Strict):
    day: int
    value: str
    reason: str


class ExceptionOption(Strict):
    value: str
    reason: str


class PreferenceDomain(Strict):
    domain: str
    noun: str
    regimes: list[Regime] = Field(min_length=1)
    exceptions: list[ExceptionDay] = []
    exception_pool: list[ExceptionOption] = []


class Distractor(Strict):
    id: Optional[str] = None
    day: int
    text: str
    domain: str
    confounder: bool = False


class OtherPersonFact(Strict):
    day: int
    noun: str
    value: str
    text: str
    confusable_with: Optional[str] = None  # a user fact entity it could be mistaken for
    question: Optional[str] = None


class OtherPerson(Strict):
    person: str
    relation: str
    facts: list[OtherPersonFact]


class PersonaSpec(Strict):
    user_id: str
    name: str
    seed: int
    start_date: dt.date
    num_days: int = 730
    difficulty: Optional[Difficulty] = None
    facts: list[FactEntity]
    events: list[EventChain] = []
    preferences: list[PreferenceDomain]
    distractors: list[Distractor] = []
    other_people: list[OtherPerson] = []
    abstention_exclude: list[str] = []

    @model_validator(mode="after")
    def _assign_ids(self) -> "PersonaSpec":
        for f in self.facts:
            for i, c in enumerate(f.changes):
                c.id = c.id or f"{f.entity}.{i}"
        for e in self.events:
            for i, s in enumerate(e.states):
                s.id = s.id or f"{e.chain_id}.{i}"
        for p in self.preferences:
            for i, r in enumerate(p.regimes):
                r.id = r.id or f"{p.domain}.r{i}"
        for i, d in enumerate(self.distractors):
            d.id = d.id or f"distractor{i}"
        return self

    def date_of(self, day: int) -> dt.date:
        return self.start_date + dt.timedelta(days=day - 1)

    @property
    def viewpoint_day(self) -> int:
        return self.num_days + 1


def load_ladder(path: Path) -> Ladder:
    return Ladder.model_validate(yaml.safe_load(path.read_text()))


def load_spec(path: Path, ladder: Ladder) -> PersonaSpec:
    raw = yaml.safe_load(path.read_text())
    spec = PersonaSpec.model_validate(raw)
    target = ladder.users[spec.user_id]
    if spec.difficulty is None:
        spec.difficulty = target
    elif spec.difficulty != target:
        raise ValueError(f"{spec.user_id}: difficulty block differs from difficulty_ladder.yaml")
    return spec


# ── World state ─────────────────────────────────────────────────────

class ResolvedRegime(Strict):
    id: str
    domain: str
    kind: Literal["initial", "shift", "temporary", "reversion"]
    start: int
    end: int  # inclusive
    anchor: int
    rule: Rule
    label: str
    visibility: Literal["initial", "explicit", "implicit", "uncaused"]
    cause_ref: Optional[str] = None
    cause_day: Optional[int] = None
    cause_text: Optional[str] = None
    lag_days: Optional[int] = None
    reverts_to: Optional[str] = None
    revert_reason: Optional[str] = None
    first_mention_day: Optional[int] = None
    mention_days: list[int] = []


class Statement(Strict):
    id: str
    kind: Literal["background_fact", "fact_change", "event_update",
                  "other_person", "distractor"]
    text: str
    entity: Optional[str] = None
    op: Optional[str] = None
    value: Optional[str] = None
    effective_day: int
    stated_day: Optional[int] = None
    retrospective: bool = False
    is_cause: bool = False


class PrefState(Strict):
    value: str
    base_value: str
    regime_id: str
    is_exception: bool = False
    exception_reason: Optional[str] = None
    mentioned: bool = False
    mention_reason: Optional[str] = None  # sampled | exception | regime_start | top_up


class CauseInstruction(Strict):
    domain: str
    regime_id: str
    instruction: Literal["link_explicitly", "mention_without_link", "do_not_mention"]
    cause_ref: Optional[str] = None
    cause_text: Optional[str] = None
    new_routine: str


class DayState(Strict):
    day: int
    date: dt.date
    weekday: str
    has_session: bool
    preferences: dict[str, PrefState]
    active_facts: dict[str, list[str]]
    known_facts: dict[str, list[str]]
    facts_to_state_today: list[Statement] = []
    event_updates_today: list[Statement] = []
    other_people_today: list[Statement] = []
    distractors_today: list[Statement] = []
    cause_instructions: list[CauseInstruction] = []
    followup_candidates: list[str] = []
    forbidden_inventions: list[str] = []


class WorldState(Strict):
    user_id: str
    name: str
    seed: int
    start_date: dt.date
    num_days: int
    regimes: list[ResolvedRegime]
    statements: list[Statement]
    days: list[DayState]

    def regimes_for(self, domain: str) -> list[ResolvedRegime]:
        return [r for r in self.regimes if r.domain == domain]


# ── QA ──────────────────────────────────────────────────────────────

class QAItem(Strict):
    id: str
    type: str
    capability: str
    domain: str
    difficulty_user: int
    question: str
    answer: str
    answer_type: Literal["value", "date", "free_text", "yes_no", "abstain"]
    accept: list[str]
    evidence_days: list[int]
    recency_distance_days: int
    grading: Literal["exact", "date_exact", "llm_judge"]
    viewpoint_day: int
    tags: list[str] = []


# ── Sessions (used from Phase 3) ────────────────────────────────────

class Turn(Strict):
    speaker: Literal["assistant", "user"]
    text: str


class Session(Strict):
    user_id: str
    day: int
    date: dt.date
    turns: list[Turn]
    model: str
    attempts: int = 1
    passed_validation: Optional[bool] = None
