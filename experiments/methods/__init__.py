"""Registry of runnable methods. Imports are lazy so one method's dependencies never block another."""

import importlib

METHODS = {
    "ours_v3": ("experiments.methods.ours_v3", "OursV3"),
    "full_context": ("experiments.methods.full_context", "FullContext"),
    "naive_rag": ("experiments.methods.naive_rag", "NaiveRAG"),
    "mem0": ("experiments.methods.mem0_oss", "Mem0OSS"),
}


def get(name: str):
    module, cls = METHODS[name]
    return getattr(importlib.import_module(module), cls)
