"""Methods available to system_3, each paired with the live assistant that uses it."""

from experiments import config
from setup_2.baselines.NaiveRAG.method import NaiveRAG
from setup_2.baselines.TiMem.method import STRATEGIES, TiMem

from ..methods.ours_v4 import OursV4
from ..methods.ours_v4d import OursV4D
from .assistant import AgentAssistant, RetrievalAssistant


class LiveNaiveRAG(NaiveRAG):
    """setup_2's NaiveRAG plus retrieval for a live assistant turn."""

    def retrieve(self, text: str) -> str:
        if self.index is None:
            return ""
        docs = self.index.similarity_search(text, k=config.TOP_K)
        docs.sort(key=lambda d: d.metadata["date"])
        return "\n\n".join(d.page_content for d in docs)


class LiveTiMem(TiMem):
    """setup_2's TiMem plus its native recall for a live assistant turn.

    Same steps as TiMem.answer (planner, fused L1 ranking, bottom-up parents, memory refiner),
    returning the refined memories instead of an answer. setup_2's file is left untouched so
    its runs keep their implementation hash.
    """

    def retrieve(self, text: str) -> str:
        if self.pending:
            raise RuntimeError("Finalize observed sessions before retrieving")
        if not any(layer == "L1" for layer, _ in self.nodes):
            return ""
        import asyncio

        cfg = self._recall_config()
        complexity, keywords = asyncio.run(self._native_planner().analyze_query_complexity(text, max_retries=3))
        strategy_name = STRATEGIES[complexity]
        strategy = cfg["retrieval_strategies"][strategy_name]
        layers = strategy["layers"]
        l1_results = self._rank_l1(text, keywords, strategy, cfg)
        targets = [layer for layer in layers if layer != "L1"]
        parents = self._parents(l1_results, targets, strategy["final_limits"])
        candidates = self._order(l1_results + [m for layer in targets for m in parents[layer]], strategy_name, layers)
        state = asyncio.run(self._native_refiner().run(
            {"question": text, "ranked_results": candidates, "query_complexity": complexity}))
        kept = state.get("ranked_results", candidates)
        return "\n\n".join(f"[{m['level']}] ({m['start']} to {m['end']}) {m['content']}" for m in kept)


SYSTEM3_METHODS = {
    "naive_rag": (LiveNaiveRAG, RetrievalAssistant),
    "timem": (LiveTiMem, RetrievalAssistant),
    "ours_v4": (OursV4, AgentAssistant),
    "ours_v4d": (OursV4D, AgentAssistant),
}


def build(name, store_dir, instance, model, usage):
    method_cls, assistant_cls = SYSTEM3_METHODS[name]
    method = method_cls(store_dir, instance, model, usage)
    assistant = assistant_cls(method, model, usage) if assistant_cls is RetrievalAssistant else assistant_cls(method)
    return method, assistant
