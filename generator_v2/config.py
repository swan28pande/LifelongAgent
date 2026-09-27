"""Defaults for paths, seeds, tolerances, QA targets and (later phases) models."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
PERSONA_DIR = PACKAGE_DIR / "personas"
LADDER_PATH = PERSONA_DIR / "difficulty_ladder.yaml"
OUTPUT_DIR = REPO_DIR / "datasets" / "v2"

# ── Simulation ──────────────────────────────────────────────────────
MIN_REGIME_MENTIONS = 5          # every regime must be observed at least this often
MAX_STATEMENTS_PER_SESSION = 4   # required statements beyond this roll to the next session
BACKGROUND_FACTS_PER_SESSION = 3 # day-1 facts are introduced over the first sessions
EXCEPTION_BOUNDARY_GAP = 3       # no exceptions within this many days of a regime boundary
EXCEPTION_MIN_SPACING = 3        # exceptions in one domain are at least this many days apart

FORBIDDEN_INVENTIONS = [
    "new pets",
    "new cities or moves",
    "new jobs, employers or job titles",
    "new partners, family members or roommates",
    "new hobbies",
    "diet changes",
    "any fact about the user not listed in known facts",
]

# ── Checks ──────────────────────────────────────────────────────────
PROB_TOLERANCE = 0.05            # p_session, p_mention
EXCEPTION_RATE_TOLERANCE = 0.02
VISIBILITY_MIX_TOLERANCE = 0.10

# ── QA ──────────────────────────────────────────────────────────────
QA_TOTAL_TARGET = 390            # recall and prediction fill up to this total
FILLER_SPLIT = {"recall": 2, "prediction": 1}
QA_MIN_FILLER = {"recall": 60, "prediction": 30}
QA_MAX_FILLER = {"recall": 220, "prediction": 110}
PREDICTION_HORIZON = 60
BOUNDARY_WINDOW = 7
BOUNDARY_WEIGHT = 3.0
MAX_SHARED_ANSWER_FRACTION = 0.60
GUARD_MIN_GROUP_SIZE = 4

QA_TARGETS = {
    "pattern_at_time": 30,
    "fact_at_time": 40,
    "exception_vs_shift": 24,     # half exceptions (no), half real shifts (yes)
    "duration": 20,
    "abstention": 12,             # half never-mentioned, half answerable controls
}

ABSTENTION_TEMPLATES = [
    ("blood_type", "What is {name}'s blood type?"),
    ("car", "What car does {name} drive?"),
    ("university", "Which university did {name} attend?"),
    ("favorite_movie", "What is {name}'s favorite movie?"),
    ("allergy", "What is {name} allergic to?"),
    ("hometown", "Where did {name} grow up?"),
    ("phone", "What phone does {name} use?"),
    ("languages", "Which languages does {name} speak besides English?"),
    ("shoe_size", "What is {name}'s shoe size?"),
    ("favorite_color", "What is {name}'s favorite color?"),
]

# ── Conversations (Vertex AI) ───────────────────────────────────────
CONVERSATION_MODEL = "gemini-3.5-flash"
VALIDATOR_MODEL = "gemini-3.1-pro-preview"   # a stronger model checks the writer
WRITER_TEMPERATURE = 0.9
CONCURRENCY = 16
MAX_RETRIES = 5
MIN_TURNS = 12
MAX_TURNS = 20
