"""
TiMem LLM Adapters
LLM adapter module
"""

from .base_llm import BaseLLM
from importlib import import_module

_ADAPTERS = {
    "ClaudeAdapter": "claude_adapter",
    "OpenAIAdapter": "openai_adapter",
    "ZhipuAIAdapter": "zhipuai_adapter",
    "QwenAdapter": "qwen_adapter",
    "Qwen3EmbeddingService": "qwen_embedding_adapter",
}


def __getattr__(name):
    if name not in _ADAPTERS:
        raise AttributeError(name)
    return getattr(import_module(f".{_ADAPTERS[name]}", __name__), name)

def get_llm(provider: str, **kwargs) -> BaseLLM:
    from .llm_manager import get_llm as create
    return create(provider)

__all__ = [
    "BaseLLM",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "ZhipuAIAdapter",
    "QwenAdapter",
    "Qwen3EmbeddingService",
    "get_llm",
]
