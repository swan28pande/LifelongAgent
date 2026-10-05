"""Lazy method selection; schedule previews never load models or vector stores."""

from importlib import import_module


METHODS = {
    'ours_v4d': ('experiments.methods.ours_v4d', 'OursV4D', 'summaries_and_distillation'),
    'timem': ('setup_2.baselines.TiMem.method', 'TiMem', 'timem_complexity_aware'),
    'naive_rag': ('setup_2.baselines.NaiveRAG.method', 'NaiveRAG', 'raw_chunks_dense_top_k'),
    'full_context': ('setup_2.baselines.DirectPrompting.method', 'FullContext', 'complete_observed_history'),
}


def create(name, store, instance, model, usage):
    module, class_name, _ = METHODS[name]
    return getattr(import_module(module), class_name)(store, instance, model, usage)
