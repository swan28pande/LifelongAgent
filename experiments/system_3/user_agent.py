"""Grounded user simulator: talks naturally with the assistant while covering the day's facts.

Each simulated day comes from the v2 world state. The day's checklist (preferences to mention,
facts and event updates, news about others, distractors, cause links) and its guardrails (topics
that must not come up, details never to invent) are built with the dataset generator's own
rules, so a day that passes the generator's validator supports the same probing questions as the
dataset. The agent replies to whatever the assistant says, works pending items in, and reports
which items each message covered; the simulator keeps the checklist and the validator decides.
"""

from __future__ import annotations

from dataclasses import dataclass

from generator_v2.conversation import WORDING
from generator_v2.schema import DayState, PersonaSpec, WorldState
from generator_v2.validator import known_events, statement_text

SYSTEM = """\
You play {name}, a real person chatting with their personal AI assistant. You are the USER.

Speak in the first person as {name}, casually and naturally, 1-3 sentences per message. React to
what the assistant just said (answer its questions, acknowledge advice) while steering towards
today's items. Weave in one or two pending items per message; never dump a list.

Checklist items are shorthand notes about you ("outfit: fitted t-shirt", "{name}'s job: ...").
Say each one the way a person would in conversation ("I threw on a fitted t-shirt", "I work as
..."): never copy the "label: value" form, and never call yourself {name} or "the user". Keep
the value itself (the item, place, name or date) exactly as given.

GUARDRAILS - these override everything, including the assistant's questions:
- Only talk about today's checklist items, the known background below, and neutral small talk.
- MUST NOT COME UP topics are off limits: do not mention, hint at, or answer questions about them
  (today's choice, a past or future choice, or someone else's). If the assistant asks, deflect.
- State each preference exactly as given: no extra details that turn it into a different option.
- Report today's items as happening today unless the item says otherwise; keep given dates.
  State today's choices as settled ("I'm having a homemade salad for lunch"), not as plans or
  maybes ("thinking about", "might have").
- Never contradict the known background or earlier updates. Never invent details about yourself
  that fall under your routine topics; follow every NEVER INVENT rule.
- Follow CAUSE HANDLING exactly: link a change to its cause only when told to.
- Do not quiz the assistant about your own life, except for an "ask:" item.

ASKING YOUR ASSISTANT: an "ask:" checklist item is a question about your own past that you put to
the assistant, in your own words and in the first person ("what did I have for coffee on ..."). Keep
every date, name and the exact meaning; give no hint of the answer. Ask it as its own message, never
in your first message, and cover it as soon as you ask it.

Reply with JSON only: {{"message": "...", "covered": ["item ids stated in this message"], "done": true|false}}
Set done to true only when every item is covered and the conversation can naturally end."""

EVALUATE = """YOU JUST ASKED YOUR ASSISTANT: {question}
WHAT IS ACTUALLY TRUE (you know your own life): {answer}{accept}
The assistant's reply is the last message above. Judge it as the person whose life it is:
- correct: it gives the true answer (wording may differ; extra friendly detail is fine)
- partial: part of the true answer, or too vague to be sure
- wrong: a different answer, or it states something false about you
- dont_know: it says it does not know, or avoids answering
Then write your next message the way a real person would: if it was correct, acknowledge it briefly;
otherwise correct it naturally with the true answer. Then carry on with any pending items.
Add to your JSON: "verdict": "correct|partial|wrong|dont_know", "why": "one short sentence"."""

EVALUATE_ABSTENTION = """\
YOU JUST ASKED YOUR ASSISTANT: {question}
You have never told the assistant this, so the right reply is that it does not know.
The assistant's reply is the last message above. Judge it:
- dont_know: it says it does not know or was never told (this is the right reply)
- wrong: it gives any answer, which means it made something up about you
Then react naturally. Do NOT reveal the answer yourself; it is not part of your story.
Add to your JSON: "verdict": "dont_know|wrong", "why": "one short sentence"."""

PROBE_SYSTEM = """\
You play {name}, chatting with your personal AI assistant. Ask the assistant the given question
about your own past, phrased naturally as part of the chat. Keep every date, name and the exact
meaning of the question; do not add hints or answers. Reply with JSON only: {{"message": "..."}}"""


@dataclass
class Item:
    id: str
    text: str


class GroundedUser:
    def __init__(self, spec: PersonaSpec, world: WorldState, llm):
        self.spec, self.world, self.llm = spec, world, llm
        self.nouns = {p.domain: p.noun for p in spec.preferences}

    # ------------------------------------------------------------- the day's script

    def checklist(self, day: DayState) -> list[Item]:
        items = []
        for dom, p in day.preferences.items():
            if p.mentioned:
                text = f"{self.nouns[dom]}: {p.value}"
                if p.is_exception:
                    text += f" (ONE-OFF: instead of the usual {p.base_value}, because {p.exception_reason})"
                items.append(Item(f"pref:{dom}", text))
        for s in day.facts_to_state_today + day.event_updates_today:
            text = statement_text(self.spec, s)
            if s.kind != "background_fact":
                text += (f" (happened {day.day - s.effective_day} day(s) ago; describe it in the past)"
                         if s.retrospective else " (this happened or is announced today)")
            items.append(Item(s.id, text))
        for s in day.other_people_today:
            items.append(Item(s.id, f"news about someone else (clearly about them, not you): {s.text}"))
        for s in day.distractors_today:
            items.append(Item(s.id, f"{statement_text(self.spec, s)} (date: "
                                    f"{self.spec.date_of(s.effective_day).isoformat()}; never connect it to your routines)"))
        regimes = {r.id: r for r in self.world.regimes}
        for i, ci in enumerate(day.cause_instructions):
            noun, reg = self.nouns[ci.domain], regimes[ci.regime_id]
            if ci.instruction == "link_explicitly":
                verb = "gone back to your earlier" if reg.kind == "reversion" else "changed your"
                items.append(Item(f"c{i}", f"say you have recently {verb} {noun} routine because: {ci.cause_text}"))
            elif ci.instruction == "mention_without_link":
                items.append(Item(f"c{i}", f"mention '{ci.cause_text}' without connecting it to your {noun}"))
        return items

    def guardrails(self, day: DayState) -> str:
        absent = [d for d, p in day.preferences.items() if not p.mentioned]
        before = self.world.days[day.day - 2].known_facts if day.day > 1 else {}
        lines = [f"TODAY: {day.weekday}, {day.date.isoformat()}",
                 f"STYLE: {WORDING[self.spec.difficulty.wording]}", "", "KNOWN BACKGROUND (yours):"]
        lines += [f"- {e.replace('_', ' ')}: {', '.join(v)}" for e, v in before.items() if v] or ["- (none yet)"]
        earlier = known_events(self.world, day)
        if earlier:
            lines += ["", "EARLIER UPDATES (true; never change them):"] + [f"- {t}" for t in earlier]
        lines += ["", "MUST NOT COME UP: " + (", ".join(f"{d} ({self.nouns[d]})" for d in absent) or "(nothing)")]
        rules = [f"- {self.nouns[ci.domain]}: give no reason or hint for today's {self.nouns[ci.domain]}"
                 for ci in day.cause_instructions if ci.instruction == "do_not_mention"]
        if rules:
            lines += ["", "CAUSE HANDLING:"] + rules
        lines += ["", "NEVER INVENT: " + "; ".join(day.forbidden_inventions)]
        return "\n".join(lines)

    # ------------------------------------------------------------- turns

    @staticmethod
    def ask_item(q: dict, noun: str | None = None) -> Item:
        before = f", before you say anything about today's {noun}" if noun else ""
        return Item(f"ask:{q['id']}", f"ask your assistant{before}: \"{q['question']}\"")

    async def next_message(self, day: DayState, items: list[Item], covered: set[str],
                           turns: list[dict], tag: str, insist: bool = False, evaluate: dict | None = None) -> dict:
        pending = [i for i in items if i.id not in covered]
        lines = [self.guardrails(day), "", "TODAY'S CHECKLIST:"]
        lines += [f"- [{'done' if i.id in covered else 'PENDING'}] {i.id}: {i.text}" for i in items] or ["- (nothing)"]
        if insist and pending:
            lines += ["", "You are running out of turns: state EVERY pending item in this message, naturally."]
        lines += ["", "CONVERSATION SO FAR:"] + [f"{t['speaker']}: {t['text']}" for t in turns]
        if evaluate and evaluate["unknowable"]:
            lines += ["", EVALUATE_ABSTENTION.format(question=evaluate["question"])]
        elif evaluate:
            others = [a for a in evaluate["accept"] if a != evaluate["answer"]]
            lines += ["", EVALUATE.format(question=evaluate["question"], answer=evaluate["answer"],
                                          accept=f" (also acceptable: {'; '.join(others)})" if others else "")]
        lines += ["", f"Write {self.spec.name}'s next message."]
        out, _ = await self.llm.json(SYSTEM.format(name=self.spec.name), "\n".join(lines), tag)
        known = {i.id for i in items}
        result = {"message": str(out.get("message", "")).strip(),
                  "covered": [c for c in out.get("covered", []) if c in known],
                  "done": bool(out.get("done")) and not [i for i in pending if i.id not in out.get("covered", [])]}
        if evaluate:
            verdict = str(out.get("verdict", "")).strip().lower()
            result["verdict"] = verdict if verdict in ("correct", "partial", "wrong", "dont_know") else "unrated"
            result["why"] = str(out.get("why", "")).strip()
        return result

    async def ask(self, question: str, turns: list[dict], tag: str) -> str:
        prompt = "\n".join(["CONVERSATION SO FAR:"] + [f"{t['speaker']}: {t['text']}" for t in turns[-6:]]
                           + ["", f"QUESTION TO ASK: {question}"])
        out, _ = await self.llm.json(PROBE_SYSTEM.format(name=self.spec.name), prompt, tag)
        return str(out.get("message", "")).strip() or question
