"""Thin async wrapper over chat models. Gemini goes through Vertex AI (application-default
credentials); anything named gpt-* goes through OpenAI. Every call is logged."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dotenv
from langchain_core.messages import HumanMessage, SystemMessage

dotenv.load_dotenv()


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


def _vertex_project() -> str | None:
    if os.getenv("GOOGLE_CLOUD_PROJECT"):
        return os.getenv("GOOGLE_CLOUD_PROJECT")
    import google.auth
    return google.auth.default()[1]


def make_chat_model(model: str, temperature: float):
    if model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model, temperature=temperature, vertexai=True, project=_vertex_project(),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            response_mime_type="application/json",
        )
    if model.startswith("gpt"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature,
                          model_kwargs={"response_format": {"type": "json_object"}})
    raise ValueError(f"unsupported model {model!r}")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    return json.loads(text)


class LLM:
    def __init__(self, model: str, temperature: float, log_path: Path):
        self.model = model
        self.chat = make_chat_model(model, temperature)
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    async def json(self, system: str, user: str, tag: str) -> tuple[dict, Usage]:
        start = time.time()
        usage = Usage()
        error = None
        try:
            msg = await self.chat.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
            meta = msg.usage_metadata or {}
            usage = Usage(meta.get("input_tokens", 0), meta.get("output_tokens", 0),
                          (meta.get("output_token_details") or {}).get("reasoning", 0))
            return parse_json(_text(msg.content)), usage
        except Exception as e:
            error = f"{type(e).__name__}: {e}"[:500]
            raise
        finally:
            with self.log_path.open("a") as f:
                f.write(json.dumps({
                    "ts": round(start, 3), "model": self.model, "tag": tag,
                    "latency_s": round(time.time() - start, 2),
                    "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                    "reasoning_tokens": usage.reasoning_tokens, "error": error,
                }) + "\n")
