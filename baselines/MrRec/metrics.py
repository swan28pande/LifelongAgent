"""Evaluation metrics for MR.Rec: Recall@K and NDCG@K."""

import numpy as np


def recall_at_k(retrieved_ids: list, ground_truth_id: str, k: int) -> float:
    return 1.0 if ground_truth_id in retrieved_ids[:k] else 0.0


def ndcg_at_k(retrieved_ids: list, ground_truth_id: str, k: int) -> float:
    for i, item_id in enumerate(retrieved_ids[:k]):
        if item_id == ground_truth_id:
            return 1.0 / np.log2(i + 2)
    return 0.0


def compute_metrics(results: list, ks: list = [10, 100, 1000]) -> dict:
    """
    Compute Recall@K and NDCG@K averaged over all queries.

    results: list of {retrieved_ids: list[str], ground_truth_id: str}
    Returns dict with R@K and N@K for each k.
    """
    scores = {f"R@{k}": [] for k in ks}
    scores.update({f"N@{k}": [] for k in ks})

    for r in results:
        retrieved = [item_id for item_id, _ in r["retrieved"]]
        gt = r["ground_truth_id"]
        for k in ks:
            scores[f"R@{k}"].append(recall_at_k(retrieved, gt, k))
            scores[f"N@{k}"].append(ndcg_at_k(retrieved, gt, k))

    return {metric: np.mean(vals) for metric, vals in scores.items()}


def print_metrics(metrics: dict, label: str = ""):
    header = f"\n{'='*50}\n{label}\n{'='*50}" if label else ""
    if header:
        print(header)
    for metric, val in sorted(metrics.items()):
        print(f"  {metric}: {val:.4f}")
