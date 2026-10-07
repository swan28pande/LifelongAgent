"""Helpers shared by the methods that answer with one LLM call over retrieved text."""

import contextvars
import threading
from contextlib import contextmanager
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage

from .. import config
from ..core.types import Answer, Question, Session, prompt_for


def format_session(s: Session) -> str:
    when = f"{s.date} {s.time}" if s.time else s.date
    return f"=== Conversation on {when} ===\n" + "\n".join(f"{t.speaker}: {t.text}" for t in s.turns)


def text_of(content) -> str:
    if isinstance(content, list):
        content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content).strip()


def answer_from_context(llm, q: Question, context: str, meta: dict | None = None) -> Answer:
    reply = llm.invoke([SystemMessage(content=config.ANSWER_SYSTEM),
                        HumanMessage(content=f"MEMORY CONTEXT:\n{context}\n\nQUESTION: {prompt_for(q)}")])
    return Answer(text_of(reply.content), {"context_chars": len(context), **(meta or {})})


@lru_cache(maxsize=1)
def embeddings():
    """One copy of the shared embedding model per process."""
    from memory_v3.store import PrefixedEmbeddings
    return PrefixedEmbeddings(model_name=config.EMBED_MODEL, model_kwargs={"trust_remote_code": True})


_OBSERVATION_DATE = contextvars.ContextVar("mem0_observation_date", default=None)


@contextmanager
def mem0_observation_date(day: str):
    """Ground Mem0's fact extraction in the session date.

    Mem0 OSS rejects add(timestamp=...), so its prompt's "Observation Date" (the anchor for
    "yesterday", "last week") silently becomes today's system date. Its prompt builder accepts the
    date; inject it, as the hosted platform does.
    """
    import mem0.memory.main as mem0_main

    if not getattr(mem0_main.generate_additive_extraction_prompt, "_observation_date", False):
        original = mem0_main.generate_additive_extraction_prompt

        def prompt(*args, **kwargs):
            if kwargs.get("timestamp") is None and _OBSERVATION_DATE.get() is not None:
                kwargs["timestamp"] = _OBSERVATION_DATE.get()
            return original(*args, **kwargs)

        prompt._observation_date = True
        mem0_main.generate_additive_extraction_prompt = prompt
    token = _OBSERVATION_DATE.set(day)
    try:
        yield
    finally:
        _OBSERVATION_DATE.reset(token)


_EMBED_LOCK = threading.Lock()


def serialize_embedder(embedder) -> None:
    """Concurrent answer threads sharing one nomic-embed model corrupt each other's tensors."""
    inner = embedder.embed

    def embed(*args, **kwargs):
        with _EMBED_LOCK:
            return inner(*args, **kwargs)

    embedder.embed = embed


def make_llm(model: str, usage):
    import os

    import dotenv

    dotenv.load_dotenv()
    if "gemini" not in model.lower():
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0.0, callbacks=[usage])

    from langchain_google_genai import ChatGoogleGenerativeAI

    options = {"model": model, "temperature": 0.0, "callbacks": [usage]}
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() not in ("false", "0", "no"):
        import google.auth

        options.update(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT") or google.auth.default()[1],
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        )
    return ChatGoogleGenerativeAI(**options)
