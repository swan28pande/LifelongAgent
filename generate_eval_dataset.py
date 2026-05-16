"""
Synthetic evaluation dataset for lifelong memory agents.

Design requirements:
  - Every session covers ALL preference domains (not rotating)
  - Rich evolving facts: stable facts + new facts introduced across sessions
  - Pattern difficulties:
      coffee   → SIMPLE   (strict 7-day cycle)
      clothing → MID      (alternates every 3 days)
      exercise → DIFFICULT (day-of-week nested: different for each day of the week)

QA types: factual, recall, pattern_id, prediction, transition, duration
"""

import json
import os
import time
from datetime import datetime, timedelta

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv

dotenv.load_dotenv()

def get_llm(model_name: str, temperature: float = 0.85):
    if "gemini" in model_name.lower():
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    return ChatOpenAI(model=model_name, temperature=temperature)

BASE    = datetime(2026, 3, 1)  # Day 1 = Sunday


def date_of(day: int) -> str:
    return (BASE + timedelta(days=day - 1)).strftime("%Y-%m-%d")

def dow(day: int) -> int:
    """1=Mon … 7=Sun"""
    return ((BASE + timedelta(days=day - 1)).weekday()) + 1

def week(day: int) -> int:
    return (day - 1) // 7 + 1


# ── Pattern functions ──────────────────────────────────────────────────────────

def coffee_pattern(day):
    """SIMPLE: strict 7-day cycle — oat milk latte (odd weeks) / black coffee (even weeks)."""
    return "oat milk latte" if week(day) % 2 == 1 else "black coffee"

def clothing_pattern(day):
    """MID: alternates every 3 days — fitted t-shirt / oversized hoodie."""
    return "fitted t-shirt" if (day - 1) % 6 < 3 else "oversized hoodie"

def exercise_pattern(day):
    """DIFFICULT: depends on day-of-week — different activity for each day."""
    d = dow(day)
    return {
        1: "morning yoga",    # Mon
        2: "evening run",     # Tue
        3: "morning yoga",    # Wed
        4: "evening run",     # Thu
        5: "morning yoga",    # Fri
        6: "climbing gym",    # Sat
        7: "rest day",        # Sun
    }[d]

DOMAIN_PATTERNS = {
    "coffee":   {"fn": coffee_pattern,   "difficulty": "simple",
                 "description": "Strict 7-day cycle: oat milk latte for 7 days, then black coffee for 7 days, repeating."},
    "clothing": {"fn": clothing_pattern, "difficulty": "mid",
                 "description": "Alternates every 3 days: fitted t-shirt for 3 days, then oversized hoodie for 3 days, repeating."},
    "exercise": {"fn": exercise_pattern, "difficulty": "difficult",
                 "description": "Day-of-week rule: Mon/Wed/Fri = morning yoga, Tue/Thu = evening run, Sat = climbing gym, Sun = rest day."},
}

# Natural-language phrasings used in QA questions instead of raw domain names.
DOMAIN_NATURAL = {
    "coffee": {
        "recall":     "What was {name} drinking on {date} ({day})?",
        "pattern_id": "Describe the pattern in what {name} chooses to drink over time.",
        "prediction": "What will {name} most likely be drinking on {date} ({day})?",
        "transition": "When did the choice of what {name} was drinking switch from '{prev}' to '{cur}'?",
    },
    "clothing": {
        "recall":     "What was {name} wearing on {date} ({day})?",
        "pattern_id": "Describe the pattern in how {name} chooses what to wear.",
        "prediction": "What will {name} most likely be wearing on {date} ({day})?",
        "transition": "When did the style {name} was wearing switch from '{prev}' to '{cur}'?",
    },
    "exercise": {
        "recall":     "How was {name} working out on {date} ({day})?",
        "pattern_id": "Describe {name}'s routine for working out.",
        "prediction": "How will {name} most likely be working out on {date} ({day})?",
        "transition": "When did {name}'s way of working out switch from '{prev}' to '{cur}'?",
    },
}

# ── Evolving facts ────────────────────────────────────────────────────────────
# Stable facts are always present. Evolving facts are introduced on specific days.

STABLE_FACTS = {
    "name":       "Jordan",
    "age":        28,
    "job":        "graphic designer at a small agency",
    "city":       "Austin, Texas",
    "lives_with": "roommate named Sam",
    "hobby":      "rock climbing",
    "diet":       "vegetarian",
}

# Each fact has: introduced_day (when it can first come up), must_state_day (day it MUST be
# explicitly stated if not yet covered), and keywords to detect coverage in the conversation.
ALL_FACTS = [
    # ── Stable facts — must be explicitly stated in early sessions ──
    {"id": "city",       "introduced_day": 1,  "must_state_day": 2,
     "fact": "Jordan lives in Austin, Texas.",
     "keywords": ["austin", "texas"]},
    {"id": "job",        "introduced_day": 1,  "must_state_day": 2,
     "fact": "Jordan works as a graphic designer at a small agency.",
     "keywords": ["graphic designer", "agency", "design"]},
    {"id": "diet",       "introduced_day": 1,  "must_state_day": 3,
     "fact": "Jordan is vegetarian.",
     "keywords": ["vegetarian", "vegan", "no meat", "plant"]},
    {"id": "lives_with", "introduced_day": 1,  "must_state_day": 2,
     "fact": "Jordan lives with a roommate named Sam.",
     "keywords": ["sam", "roommate"]},
    {"id": "age",        "introduced_day": 1,  "must_state_day": 4,
     "fact": "Jordan is 28 years old.",
     "keywords": ["28", "twenty-eight", "twenties"]},
    {"id": "hobby",      "introduced_day": 1,  "must_state_day": 2,
     "fact": "Jordan's main hobby is rock climbing.",
     "keywords": ["climbing", "boulder", "gym", "routes"]},

    # ── Evolving facts ──────────────────────────────────────────────
    {"id": "deadline",      "introduced_day": 1,  "must_state_day": 1,
     "fact": "Jordan has a big client presentation deadline on March 10.",
     "keywords": ["presentation", "deadline", "march 10", "client"]},
    {"id": "puppy_coming",  "introduced_day": 3,  "must_state_day": 3,
     "fact": "Jordan's roommate Sam is getting a puppy next week.",
     "keywords": ["puppy", "dog", "sam", "getting"]},
    {"id": "draft_feedback","introduced_day": 5,  "must_state_day": 5,
     "fact": "Jordan received positive client feedback on a draft design.",
     "keywords": ["feedback", "draft", "positive", "client"]},
    {"id": "presentation_done","introduced_day": 7, "must_state_day": 7,
     "fact": "Jordan finished the client presentation and it went really well.",
     "keywords": ["presentation", "went well", "finished", "done"]},
    {"id": "puppy_arrived", "introduced_day": 9,  "must_state_day": 9,
     "fact": "Sam's puppy, a golden retriever named Mango, arrived home.",
     "keywords": ["mango", "golden retriever", "arrived", "home"]},
    {"id": "competition_considering","introduced_day": 11,"must_state_day": 11,
     "fact": "Jordan is considering signing up for a beginner climbing competition in April.",
     "keywords": ["competition", "april", "signing up", "considering"]},
    {"id": "competition_registered","introduced_day": 13,"must_state_day": 13,
     "fact": "Jordan registered for the April climbing competition.",
     "keywords": ["registered", "competition", "signed up"]},
    {"id": "promotion_offered","introduced_day": 15,"must_state_day": 15,
     "fact": "Jordan's agency offered them a senior designer role.",
     "keywords": ["senior", "promotion", "offered", "role"]},
    {"id": "promotion_accepted","introduced_day": 17,"must_state_day": 17,
     "fact": "Jordan accepted the senior designer promotion.",
     "keywords": ["accepted", "senior", "promotion"]},
    {"id": "trip_planning",  "introduced_day": 19, "must_state_day": 19,
     "fact": "Jordan is planning a weekend trip to Barton Springs with Sam and Mango.",
     "keywords": ["barton springs", "trip", "weekend"]},
    {"id": "trip_done",      "introduced_day": 21, "must_state_day": 21,
     "fact": "Jordan went to Barton Springs — it was relaxing and Mango loved the water.",
     "keywords": ["barton springs", "relaxing", "mango", "water"]},
]

# Coverage tracker — updated during generation
fact_coverage: dict = {f["id"]: False for f in ALL_FACTS}


def get_known_facts(day: int) -> list:
    """All facts introduced up to this day."""
    return [f for f in ALL_FACTS if f["introduced_day"] <= day]

def get_must_state_today(day: int) -> list:
    """Facts that MUST be explicitly stated today (either newly introduced or overdue)."""
    must = []
    for f in ALL_FACTS:
        if f["introduced_day"] <= day and f["must_state_day"] <= day:
            if not fact_coverage.get(f["id"], False):
                must.append(f)
    return must

def mark_covered(day_text: str):
    """Scan conversation text and mark facts as covered."""
    text_lower = day_text.lower()
    for f in ALL_FACTS:
        if not fact_coverage.get(f["id"], False):
            if any(kw in text_lower for kw in f["keywords"]):
                fact_coverage[f["id"]] = True


# ── Conversation generation ───────────────────────────────────────────────────

CONV_SYSTEM = """\
You are writing a natural conversation between an AI assistant and a user named Jordan.
The assistant remembers everything from previous sessions and asks genuine follow-up questions.
Jordan is a real person who thinks before answering, uses natural hesitation
("Hmm...", "Actually...", "You know what—"), and doesn't volunteer everything upfront.

RULES:
1. REQUIRED FACTS must be explicitly stated by Jordan in this conversation (not just hinted at).
   Each required fact needs 2-4 turns of real discussion — not a single mention.
2. The assistant follows up on known ongoing facts with genuine curiosity.
3. ALL three preference topics (coffee, clothing, exercise) must be covered naturally.
4. Jordan reveals preferences through the conversation, not by announcing them.
5. 16-20 turns total. End naturally.

Return JSON: {{"turns": [{{"speaker": "Assistant"|"User", "text": "..."}}]}}\
"""


def generate_conversation(day: int, must_state: list, prev_ctx: str = "") -> list:
    today_values = {domain: info["fn"](day) for domain, info in DOMAIN_PATTERNS.items()}

    known_facts = get_known_facts(day)
    # Facts already covered in previous sessions (for follow-up)
    already_covered = [f for f in known_facts
                       if f["introduced_day"] < day and fact_coverage.get(f["id"], False)]

    hidden = "\n".join(f"  {d}: {v}" for d, v in today_values.items())
    prev   = f"\nPrevious session summary: {prev_ctx}" if prev_ctx else ""

    followup_text = ""
    if already_covered:
        followup_text = "\nFacts assistant knows — ask genuine follow-up questions about these:\n"
        followup_text += "\n".join(f"  - {f['fact']}" for f in already_covered[-3:])

    must_text = ""
    if must_state:
        must_text = "\n\nREQUIRED — Jordan MUST explicitly state these facts today (2-4 turns each):\n"
        must_text += "\n".join(f"  [{i+1}] {f['fact']}" for i, f in enumerate(must_state))

    msg = (
        f"Today's hidden ground truth (Jordan reveals naturally — NOT upfront):\n"
        f"{hidden}\n"
        f"{followup_text}"
        f"{must_text}\n\n"
        f"Date: {date_of(day)} "
        f"({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow(day)-1]})"
        f"{prev}\n\nGenerate the conversation."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", CONV_SYSTEM),
        ("human", "{msg}"),
    ])
    try:
        chain  = prompt | llm | JsonOutputParser()
        result = chain.invoke({"msg": msg})
        turns  = result.get("turns", result) if isinstance(result, dict) else result
        return [t for t in turns if t.get("text")]
    except Exception as e:
        print(f"    Conversation error day {day}: {e}")
        return []


# ── QA generation ─────────────────────────────────────────────────────────────

def generate_qa(train_days: int) -> list:
    name = STABLE_FACTS["name"]
    qa   = []

    # ── Stable factual questions ──────────────────────────────────────
    for q, a in [
        (f"What does {name} do for work?",           STABLE_FACTS["job"]),
        (f"Where does {name} live?",                 STABLE_FACTS["city"]),
        (f"Who does {name} live with?",              STABLE_FACTS["lives_with"]),
        (f"What is {name}'s main hobby?",            STABLE_FACTS["hobby"]),
        (f"What are {name}'s dietary preferences?",  STABLE_FACTS["diet"]),
    ]:
        qa.append({"type": "factual", "difficulty": "simple", "domain": "facts",
                   "question": q, "answer": a})

    # ── All facts (stable + evolving) within training window ──────────
    for f in ALL_FACTS:
        if f["introduced_day"] <= train_days:
            qa.append({
                "type": "factual_evolving",
                "difficulty": "simple",
                "domain": "facts",
                "question": f"What do we know about Jordan regarding: {f['id'].replace('_',' ')}?",
                "answer": f["fact"],
                "target_day": f["introduced_day"],
            })

    # ── Per-domain QA ─────────────────────────────────────────────────
    for domain, info in DOMAIN_PATTERNS.items():
        fn         = info["fn"]
        difficulty = info["difficulty"]
        desc       = info["description"]
        phrasing   = DOMAIN_NATURAL[domain]
        dow_names  = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        # Recall — sample from each week
        for w_num in range(1, (train_days // 7) + 1):
            sample_day = w_num * 7 - 3
            if sample_day > train_days:
                continue
            qa.append({
                "type": "recall", "difficulty": difficulty, "domain": domain,
                "question": phrasing["recall"].format(
                    name=name,
                    date=date_of(sample_day),
                    day=dow_names[dow(sample_day) - 1],
                ),
                "answer": fn(sample_day),
                "target_day": sample_day,
            })

        # Pattern identification
        qa.append({
            "type": "pattern_id", "difficulty": difficulty, "domain": domain,
            "question": phrasing["pattern_id"].format(name=name),
            "answer": desc,
        })

        # Prediction — 3 future days beyond training window
        for predict_day in [train_days + 1, train_days + 7, train_days + 14]:
            qa.append({
                "type": "prediction", "difficulty": difficulty, "domain": domain,
                "question": phrasing["prediction"].format(
                    name=name,
                    date=date_of(predict_day),
                    day=dow_names[dow(predict_day) - 1],
                ),
                "answer": fn(predict_day),
                "target_day": predict_day,
            })

        # Transitions (value changes within training window)
        prev_val = fn(1)
        for d in range(2, train_days + 1):
            cur_val = fn(d)
            if cur_val != prev_val:
                qa.append({
                    "type": "transition", "difficulty": difficulty, "domain": domain,
                    "question": phrasing["transition"].format(
                        name=name, prev=prev_val, cur=cur_val,
                    ),
                    "answer": date_of(d),
                    "target_day": d,
                })
            prev_val = cur_val

    return qa


# ── Main ──────────────────────────────────────────────────────────────────────

def main(num_days: int = 21, dry_run: bool = False, model: str = "gpt-4o-mini", out_dir: str = "eval_dataset"):
    global llm
    llm = get_llm(model)
    os.makedirs(out_dir, exist_ok=True)
    name = STABLE_FACTS["name"]

    print(f"\n{'='*55}")
    print(f"User: {name} | {num_days} days")
    print(f"{'='*55}")
    print("\nPatterns:")
    for domain, info in DOMAIN_PATTERNS.items():
        print(f"  [{info['difficulty'].upper():9s}] {domain}: {info['description']}")

    print(f"\nDay-by-day schedule (first {min(num_days,7)} days):")
    print(f"  {'Day':>4}  {'Date':>12}  {'DoW':>4}  {'coffee':>20}  {'clothing':>20}  {'exercise':>15}")
    for d in range(1, min(num_days, 8)):
        dw = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow(d)-1]
        print(f"  {d:>4}  {date_of(d):>12}  {dw:>4}  "
              f"{coffee_pattern(d):>20}  {clothing_pattern(d):>20}  {exercise_pattern(d):>15}")

    if dry_run:
        print("\n[dry_run=True — skipping conversation generation]")
        return

    # Reset coverage tracker for this run
    for fid in fact_coverage:
        fact_coverage[fid] = False

    sessions, prev_ctx = {}, ""
    for day in range(1, num_days + 1):
        dow_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow(day)-1]

        # Determine which facts must be stated today
        must_state = get_must_state_today(day)
        coverage_pct = sum(fact_coverage.values()) / len(fact_coverage) * 100
        overdue = [f["id"] for f in must_state]

        print(f"\nDay {day:02d} ({date_of(day)}, {dow_name}) | "
              f"coverage: {sum(fact_coverage.values())}/{len(fact_coverage)} "
              f"({coverage_pct:.0f}%)"
              + (f" | MUST STATE: {overdue}" if overdue else ""))

        turns = generate_conversation(day, must_state, prev_ctx)
        if turns:
            sessions[date_of(day)] = {"day": day, "date": date_of(day), "turns": turns}

            # Mark facts covered based on conversation text
            full_text = " ".join(t["text"] for t in turns)
            mark_covered(full_text)

            prev_ctx = " | ".join(
                f"{t['speaker']}: {t['text'][:50]}" for t in turns[-3:]
            )
            print(f"  {len(turns)} turns | "
                  f"newly covered: {[f['id'] for f in ALL_FACTS if fact_coverage.get(f['id']) and f['introduced_day'] <= day][-3:]}")
            for t in turns[:2]:
                print(f"    {t['speaker']:12s}: {t['text'][:80]}")
        time.sleep(0.4)

    # Final coverage report
    print(f"\n{'='*55}")
    print("FACT COVERAGE REPORT")
    print(f"{'='*55}")
    covered   = [f for f in ALL_FACTS if fact_coverage.get(f["id"])]
    uncovered = [f for f in ALL_FACTS if not fact_coverage.get(f["id"])]
    print(f"Covered:   {len(covered)}/{len(ALL_FACTS)}")
    for f in covered:
        print(f"  ✓ [{f['id']}] {f['fact'][:60]}")
    if uncovered:
        print(f"Uncovered: {len(uncovered)}")
        for f in uncovered:
            print(f"  ✗ [{f['id']}] {f['fact'][:60]}")

    conversations = {"user_1": {"facts": STABLE_FACTS, "sessions": sessions}}
    qa_pairs      = {"user_1": {"name": name, "qa_pairs": generate_qa(num_days)}}
    schedules     = {"user_1": {
        "facts": STABLE_FACTS,
        "all_facts": ALL_FACTS,
        "patterns": {
            domain: {
                "difficulty": info["difficulty"],
                "description": info["description"],
                "schedule": {d: info["fn"](d) for d in range(1, num_days + 15)},
            }
            for domain, info in DOMAIN_PATTERNS.items()
        }
    }}

    for fname, obj in [("conversations", conversations),
                       ("qa_pairs", qa_pairs),
                       ("schedules", schedules)]:
        path = os.path.join(out_dir, f"{fname}.json")
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"\nSaved {fname} → {path}")

    # QA summary
    qa = qa_pairs["user_1"]["qa_pairs"]
    from collections import Counter
    print(f"\nQA pairs: {len(qa)}")
    by_diff = Counter(q["difficulty"] for q in qa)
    by_type = Counter(q["type"] for q in qa)
    print(f"  By difficulty: {dict(by_diff)}")
    print(f"  By type:       {dict(by_type)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--num_days",  type=int,  default=5)
    p.add_argument("--dry_run",   action="store_true",
                   help="Print schedule without generating conversations")
    p.add_argument("--model",     type=str,  default="gpt-4o-mini")
    p.add_argument("--out_dir",   type=str,  default="eval_dataset")
    args = p.parse_args()
    main(args.num_days, args.dry_run, args.model, args.out_dir)
