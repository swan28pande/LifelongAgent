"""
TiMem Gemini/Vertex AI LLM Adapter

Uses langchain_google_genai to call Gemini models via Vertex AI.
Drop-in replacement for OpenAI adapter in the TiMem LLM manager.
"""

import asyncio
import os
import time
from typing import Dict, List, Optional, Any, AsyncIterator

from llm.base_llm import (
    BaseLLM, Message, MessageRole, ChatResponse, EmbeddingResponse, ModelConfig,
    ModelType, handle_llm_errors
)
from timem.utils.logging import get_logger

logger = get_logger(__name__)


def _vertex_project() -> Optional[str]:
    explicit = os.getenv("GOOGLE_CLOUD_PROJECT")
    if explicit:
        return explicit
    try:
        import google.auth
        return google.auth.default()[1]
    except Exception:
        return None


class GeminiAdapter(BaseLLM):
    """Gemini adapter using Vertex AI via langchain_google_genai."""

    def __init__(self, config: Optional[ModelConfig] = None):
        from timem.utils.config_manager import get_llm_config

        llm_config = get_llm_config()
        gemini_config = llm_config.get("providers", {}).get("gemini", {})

        if config is None:
            config = ModelConfig(
                model_name=gemini_config.get("model", "gemini-3.5-flash"),
                temperature=gemini_config.get("temperature", 0.7),
                max_tokens=gemini_config.get("max_tokens", 2048),
            )

        super().__init__(config)
        self.model_type = ModelType.CHAT
        self._llm = None
        self._lock = asyncio.Lock()

    def _get_llm(self):
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                vertexai=True,
                project=_vertex_project(),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            )
        return self._llm

    def _to_langchain_messages(self, messages: List[Message]):
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        lc = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                lc.append(SystemMessage(content=m.content))
            elif m.role == MessageRole.ASSISTANT:
                lc.append(AIMessage(content=m.content))
            else:
                lc.append(HumanMessage(content=m.content))
        return lc

    @handle_llm_errors
    async def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        start = time.time()
        llm = self._get_llm()
        lc_msgs = self._to_langchain_messages(messages)
        resp = await asyncio.to_thread(llm.invoke, lc_msgs)
        content = resp.content
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in content
            ).strip()
        elapsed = time.time() - start
        usage = {}
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            um = resp.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "input_tokens", 0) or 0,
                "completion_tokens": getattr(um, "output_tokens", 0) or 0,
                "total_tokens": getattr(um, "total_tokens", 0) or 0,
            }
        return ChatResponse(
            content=content,
            finish_reason="stop",
            model=self.config.model_name,
            usage=usage,
            response_time=elapsed,
        )

    async def chat_stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        llm = self._get_llm()
        lc_msgs = self._to_langchain_messages(messages)

        def _stream():
            return list(llm.stream(lc_msgs))

        chunks = await asyncio.to_thread(_stream)
        for chunk in chunks:
            text = chunk.content
            if isinstance(text, list):
                text = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c) for c in text
                ).strip()
            if text:
                yield text

    @handle_llm_errors
    async def complete(self, prompt: str, **kwargs) -> str:
        messages = [Message(role=MessageRole.USER, content=prompt)]
        response = await self.chat(messages, **kwargs)
        return response.content

    @handle_llm_errors
    async def embed(self, text: str, **kwargs) -> EmbeddingResponse:
        raise NotImplementedError("Use the embedding service for embeddings")

    @handle_llm_errors
    async def embed_batch(self, texts: List[str], **kwargs) -> List[EmbeddingResponse]:
        raise NotImplementedError("Use the embedding service for embeddings")

    @handle_llm_errors
    async def summarize(self, text: str, **kwargs) -> str:
        messages = [
            Message(role=MessageRole.SYSTEM, content="Summarize the following text concisely."),
            Message(role=MessageRole.USER, content=text),
        ]
        response = await self.chat(messages, **kwargs)
        return response.content

    async def validate_model(self, model_name: str) -> bool:
        return "gemini" in model_name.lower()

    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        return {
            "name": model_name,
            "provider": "google-vertexai",
            "type": "chat",
        }
