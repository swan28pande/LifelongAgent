"""Apply minimal source corrections and revalidate the previously flagged sessions.

Backups, staged sessions and API logs stay under tmp/v2-dataset-fixes. Publication
requires every checked session to pass; cached results are bound to the exact turns
and validator prompt. The unchanged world state is the validation reference.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from generator_v2 import config, conversation, validator
from generator_v2.llm import LLM
from generator_v2.schema import WorldState, load_ladder, load_spec

OUT = Path(__file__).resolve().parent
TMP = ROOT / "tmp/v2-dataset-fixes"

# Keys are zero-based turn indices. Only erroneous text and dependent replies change.
EDITS = {
    ("u2", "2026-09-01"): {
        4: "Definitely, safety first in the lab! Has the back-to-school rush left you any time to unwind?",
        5: "The schedule is pretty packed right now, so downtime has been limited. Grace and I are just trying to unwind in the evenings.",
    },
    ("u2", "2026-09-11"): {
        4: "That's wonderful to hear. Has the week left you any time for your woodworking shop?",
        5: "Not much this week, honestly. The weather in Portland has been nice, but I'm pretty exhausted.",
    },
    ("u2", "2026-11-01"): {
        2: "Oatmeal is the perfect way to warm up on a chilly Portland morning. Have you had any time for woodworking today?",
        3: "I actually spent a couple of hours out in the garage doing some woodworking. I've been sanding down some birch for a small spice rack I'm making for Grace.",
    },
    ("u2", "2027-07-25"): {
        4: "That sounds lovely. Are you keeping things low-key at home today?",
        5: "Yes, I'm keeping it pretty low-key today. It's been nice to have some quiet time at home.",
        6: "A quiet Sunday sounds lovely. How is Olive doing?",
    },
    ("u2", "2027-09-06"): {
        2: "I'm doing great, thanks for asking! A protein smoothie is a solid way to kick off a busy day. Has school kept you busy today?",
        3: "It has! There was plenty to organize today, but I managed to get through it all.",
        4: "I'm glad it went well. How are things shaping up with the science department staff this term?",
    },
    ("u2", "2027-11-15"): {
        2: "That sounds like a great, solid breakfast to kick off the school week. Did you have a quiet weekend, or was it mostly prep for classes?",
        3: "It was fairly quiet. The weather was a bit damp, so Grace and I mostly kept things low-key at home.",
    },
    ("u2", "2027-12-01"): {
        4: "I can imagine the pre-break energy is quite high! Have you had any quiet time at home to unwind today?",
        5: "A little this afternoon, thankfully. It was nice to take a breather after getting through the grading.",
        6: "I'm glad you got a breather! What are your plans for this evening? Just relaxing at home?",
    },
    ("u2", "2027-12-11"): {
        6: "It definitely is. Has it been a quiet afternoon at home?",
        7: "Yes, it's been lovely to have a quiet afternoon after such a busy week.",
        8: "I'm glad you've had some downtime. How are things going at the high school lately?",
    },
    ("u2", "2027-12-31"): {
        6: "I'm sure Olive is absolutely thrilled to have her favorite human home. Has the Portland winter weather made it a cozy day indoors?",
        7: "It has. The weather hasn't been great, but it's been lovely to have some quiet time at home.",
        8: "That sounds relaxing. How is your neighbor's vegetable garden looking these days? I imagine there's not much going on in late December.",
        11: "Probably just getting to spend that time with Grace. Anyway, I should probably go help her finish up dinner. I hope you have a great night!",
    },
    ("u3", "2026-08-28"): {
        3: "Not this one, he's been busy with work, but I'll save some for him. Oh, by the way, my roommate bought a brand new espresso machine yesterday, August 27, so the kitchen has been smelling incredible all morning.",
    },
    ("u3", "2027-05-25"): {
        5: "It's a mystery thriller set in New England. It's really gripping so far! Oh, by the way, a new sandwich shop opened up right down the street from the clinic yesterday, May 24.",
    },
    ("u3", "2027-06-04"): {
        9: "Yes, I'm still keeping up with it! I usually do a quick twenty-minute lesson in the evenings. I'm starting to learn how to order food, which makes the lessons fun.",
    },
    ("u3", "2027-06-26"): {
        11: "We haven't done any deep planning yet. Right now we're concentrating on unpacking. Our summer 2028 wedding is still planned, but I don't have any other planning updates today.",
        12: "That makes sense. Unpacking is plenty to handle today! I hope you get to relax and enjoy the rest of your weekend.",
    },
    ("u3", "2027-09-13"): {
        6: "You've got this. Any other news you'd like to share before your shift?",
        7: "Yes! The host of that true-crime podcast announced a book tour two days ago, on September 11. That was a nice bit of news to hear.",
        8: "That's interesting news! How have things been going with your online Italian course lately? Have you had much time to practice?",
    },
    ("u3", "2027-12-01"): {
        8: "That sounds like an interesting listen. What did the program explain?",
        9: "They discussed positive reinforcement techniques and how rewards help with training. It was an interesting way to pass the time during the ride.",
        10: "That sounds informative. How are you and Tom doing this evening?",
        11: "We're taking it easy at home. I'm keeping my ankle elevated and resting after the shift.",
    },
    ("u4", "2026-03-02"): {
        7: "I did get some downtime. Oh, and I should mention that cycling is one of my hobbies.",
        8: "Thanks for telling me. Is there any other personal news you'd like to share?",
        9: "Yes! I'm dating Leah now, and things are going really well.",
    },
    ("u4", "2026-04-13"): {
        4: "It's great that you were able to find that zone of deep focus when you needed it. Hopefully the schematics turned out well! Did you get any time for photography yesterday?",
        5: "I did, actually! I took some photos around the Chain of Lakes yesterday. The Minneapolis spring weather is finally starting to show up, which was great.",
        6: "That sounds like a lovely afternoon. Did Leah join you?",
        7: "No, she had some family things to attend to, so it was just me and my camera.",
    },
    ("u4", "2026-04-28"): {
        4: "That sounds like the perfect soundtrack for getting into a creative flow. Do you have anything planned for this evening after work?",
        5: "I'm planning to make a nice meal tonight to wind down. I've got some ground meat and fresh cilantro, and I'm planning to assemble some seasoned fillings inside warm, folded corn tortillas.",
    },
    ("u4", "2026-06-19"): {
        3: "They definitely did. I got everything submitted right before five. I even had some time afterward to take a couple of photos of the sunset reflection on the water near the Chain of Lakes.",
        4: "That sounds like a beautiful scene! Minneapolis has some of the best lake views for photography. Are you and Leah doing anything special to celebrate the weekend?",
    },
    ("u4", "2026-09-26"): {
        0: "Hi Daniel! Happy Saturday. How is your weekend in Minneapolis starting out?",
        6: "I'm glad you two got to enjoy that late-morning meal together before the accident. How did work fit into your day before the accident?",
        7: "I actually had to stop by the office for an hour to prep some project files. I didn't want to dress up, so my work attire was just a pair of blue denim pants and a cozy woolen pullover.",
    },
    ("u4", "2026-11-16"): {
        5: "We just did a quick toast at home. Today was straight to business anyway; I had to be out of the door incredibly early for an on-site inspection, so I actually ended up completely bypassing my morning meal. Usually, I'd have my blended fruit bowl, but there just wasn't time.",
    },
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


async def main() -> int:
    reviews = json.loads((ROOT / ".research/v2-probing-recheck/source/flag_reviews.json").read_text())
    targets = sorted({(r["user"], r["date"]) for r in reviews["failure_reviews"]})
    ladder = load_ladder(config.LADDER_PATH)
    specs = {uid: load_spec(config.PERSONA_DIR / f"{uid}.yaml", ladder) for uid, _ in targets}
    worlds = {uid: WorldState.model_validate_json((config.OUTPUT_DIR / uid / "world_state.json").read_text())
              for uid in specs}
    checker = LLM(config.VALIDATOR_MODEL, 0, TMP / "source_validation.jsonl")
    sem = asyncio.Semaphore(6)
    results = []

    async def one(uid, date):
        path = config.OUTPUT_DIR / uid / "sessions" / f"{date}.json"
        backup = TMP / "original_sessions" / uid / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        record = json.loads(backup.read_text())
        previous = {k: record[k] for k in ("passed_validation", "failures", "extracted")}
        changes = []
        for index, text in EDITS.get((uid, date), {}).items():
            changes.append({"turn_index": index, "before": record["turns"][index]["text"], "after": text})
            record["turns"][index]["text"] = text
        check = validator.build_check(specs[uid], worlds[uid], worlds[uid].days[record["day"] - 1])
        prompt = validator.validator_prompt(specs[uid], check, record["turns"])
        signature = digest([validator.SYSTEM, prompt])
        output = OUT / "source_validation" / uid / path.name
        output.parent.mkdir(parents=True, exist_ok=True)
        cached = json.loads(output.read_text()) if output.exists() else {}
        if cached.get("signature") != signature:
            async with sem:
                fails, extracted = await validator.validate(checker, specs[uid], check, record["turns"],
                                                            f"repair {uid} {date}")
            cached = {"signature": signature, "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "model": checker.model, "failures": fails, "extracted": extracted}
            output.write_text(json.dumps(cached, indent=2, ensure_ascii=False) + "\n")
        record.update(passed_validation=not cached["failures"], failures=cached["failures"], extracted=cached["extracted"])
        record["repair_validation"] = {"checked_utc": cached["checked_utc"], "model": cached["model"],
                                       "prompt_sha256": signature, "previous_validation": previous}
        staged = TMP / "repaired_sessions" / uid / path.name
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        results.append({"user": uid, "date": date, "changes": changes,
                        "failures": cached["failures"], "before_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
                        "after_sha256": hashlib.sha256(staged.read_bytes()).hexdigest()})
        print(uid, date, "PASS" if not cached["failures"] else cached["failures"], flush=True)

    await asyncio.gather(*(one(*key) for key in targets))
    results.sort(key=lambda r: (r["user"], r["date"]))
    (OUT / "source_repairs.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    if any(r["failures"] for r in results):
        print("Staged only: fix the remaining validation failures before publication.")
        return 1
    for uid, date in targets:
        (config.OUTPUT_DIR / uid / "sessions" / f"{date}.json").write_bytes(
            (TMP / "repaired_sessions" / uid / f"{date}.json").read_bytes())
    print(f"Published {len(EDITS)} text repairs and {len(targets)} genuine session revalidations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
