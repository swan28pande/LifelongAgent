"""One place for everything that must be held equal across methods."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "experiments" / "data"
V2_DATA_DIR = ROOT / "datasets" / "v2"   # canonical v2 data for every harness
RESULTS_DIR = ROOT / "results" / "experiments"

# Same backbone, embedder and judge for every method, so differences come from memory design.
MODEL = "gemini-3.5-flash"
JUDGE_MODEL = "gemini-3.1-pro-preview"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
EMBED_DIMS = 768

TOP_K = 10                       # retrieved items for single-shot methods
CHUNK_TURNS = 5                  # turns per chunk for naive RAG (matches memory_v3)
FULL_CONTEXT_MAX_TOKENS = 900_000  # oldest sessions are dropped beyond this (approx. 4 chars/token)

ANSWER_WORKERS = 8
JUDGE_CONCURRENCY = 16

# Answer prompt shared by every method that answers in one LLM call over retrieved context.
ANSWER_SYSTEM = """\
You are a personal assistant answering questions about the user's life from the memory
context provided. Items carry the date of the conversation they came from; resolve
relative dates against it and against the date the question is asked.

Give the shortest answer that is complete. If the question asks for reasoning, give the
answer first, then the reasoning.

If the context does not contain the answer, say only: "I don't know."
"""
