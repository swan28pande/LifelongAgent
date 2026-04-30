"""
MR.Rec Baseline Evaluation Script

Evaluates three memory settings from the paper (Table 1):
  1. w/o Memory    - GPT-4o-mini query → item profile → FAISS retrieval
  2. w/ Naive Memory - recent interaction history appended
  3. w/ Static Memory - LLM-generated user summary appended

Paper results (All categories, Table 1):
  GPT-4o w/o Memory:      R@100=0.226, R@10=0.092, N@100=0.086, N@10=0.059
  GPT-4o w/ Naive Memory: R@100=0.258, R@10=0.109, N@100=0.104, N@10=0.072
  GPT-4o w/ Static Memory:R@100=0.252, R@10=0.098, N@100=0.095, N@10=0.065
"""

import argparse
import json
import os
import time
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from data_utils import load_queries, load_items_by_category, build_user_history
from llm_recommender import generate_item_profile
from memory_builder import format_naive_memory, generate_static_memory
from metrics import compute_metrics, print_metrics
from retriever import ItemRetriever

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_CATEGORIES = ["Home", "Clothing", "Tools"]
KS = [10, 100, 1000]


def run_evaluation(
    memory_setting: str = "no_memory",
    model: str = "gpt-4o-mini",
    sample_size: int = 200,
    categories: list = None,
    force_rebuild_index: bool = False,
    seed: int = 42,
):
    """
    Run evaluation for a given memory setting.

    memory_setting: "no_memory" | "naive_memory" | "static_memory"
    """
    if categories is None:
        categories = EVAL_CATEGORIES

    print(f"\n{'='*60}")
    print(f"Memory Setting: {memory_setting} | Model: {model} | Sample: {sample_size}")
    print(f"Categories: {categories}")
    print(f"{'='*60}")

    # Load data
    df = load_queries(sample_size=sample_size, seed=seed, categories=categories)
    user_history = build_user_history(df)

    # Load items
    items_by_cat = load_items_by_category()

    # Build FAISS indices
    retriever = ItemRetriever()
    for cat in categories:
        if cat in items_by_cat:
            retriever.build_index(cat, items_by_cat[cat], force_rebuild=force_rebuild_index)

    # Pre-generate static memory summaries (expensive, do once per user)
    static_summaries = {}
    if memory_setting == "static_memory":
        print("\nGenerating static user summaries...")
        unique_users = df["user_id"].unique()
        for uid in tqdm(unique_users, desc="Static memory"):
            hist = user_history.get(uid, [])
            if hist:
                try:
                    static_summaries[uid] = generate_static_memory(hist, model=model)
                except Exception as e:
                    print(f"  Static memory error for {uid}: {e}")
                    static_summaries[uid] = ""
            else:
                static_summaries[uid] = ""

    # Run evaluation per category
    all_results = []
    category_results = defaultdict(list)

    print(f"\nGenerating item profiles and retrieving ({len(df)} queries)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        query = row["query"]
        user_id = row["user_id"]
        gt_item = row["item_id"]
        category = row["category"]

        if category not in retriever.indices:
            continue

        # Build memory context
        hist = [h for h in user_history.get(user_id, []) if h["item_id"] != gt_item]

        if memory_setting == "no_memory":
            memory_text = ""
        elif memory_setting == "naive_memory":
            memory_text = format_naive_memory(hist, max_interactions=10)
        elif memory_setting == "static_memory":
            memory_text = static_summaries.get(user_id, "")
        else:
            memory_text = ""

        # Generate ideal item profile
        try:
            profile = generate_item_profile(
                query=query,
                memory_text=memory_text,
                model=model,
                memory_setting=memory_setting,
            )
        except Exception as e:
            print(f"  Profile error: {e}")
            profile = query

        # Retrieve top-k items
        try:
            retrieved = retriever.retrieve(profile, category, top_k=max(KS))
        except Exception as e:
            print(f"  Retrieval error: {e}")
            retrieved = []

        result = {
            "qid": row["qid"],
            "query": query,
            "ground_truth_id": gt_item,
            "category": category,
            "profile": profile,
            "retrieved": retrieved,
        }
        all_results.append(result)
        category_results[category].append(result)

        time.sleep(0.05)  # respect rate limits

    # Compute metrics
    print(f"\n{'='*60}")
    print(f"RESULTS — {memory_setting} ({model})")
    print(f"{'='*60}")

    overall_metrics = compute_metrics(all_results, ks=KS)
    print_metrics(overall_metrics, label="ALL Categories")

    for cat in categories:
        if category_results[cat]:
            cat_metrics = compute_metrics(category_results[cat], ks=KS)
            print_metrics(cat_metrics, label=cat)

    # Save results
    save_path = os.path.join(RESULTS_DIR, f"{memory_setting}_{model}_results.json")
    with open(save_path, "w") as f:
        json.dump({
            "memory_setting": memory_setting,
            "model": model,
            "sample_size": sample_size,
            "overall": overall_metrics,
            "by_category": {
                cat: compute_metrics(category_results[cat], ks=KS)
                for cat in categories if category_results[cat]
            },
        }, f, indent=2)
    print(f"\nResults saved to: {save_path}")

    return overall_metrics, category_results


def compare_with_paper(results: dict):
    """Print comparison table against paper's claimed results."""
    PAPER_RESULTS = {
        "no_memory": {
            "All": {"R@100": 0.226, "R@10": 0.092, "N@100": 0.086, "N@10": 0.059},
            "Home": {"R@100": 0.227, "R@10": 0.057, "N@100": 0.090, "N@10": 0.033},
            "Clothing": {"R@100": 0.100, "R@10": 0.030, "N@100": 0.036, "N@10": 0.022},
            "Tools": {"R@100": 0.247, "R@10": 0.097, "N@100": 0.096, "N@10": 0.065},
        },
        "naive_memory": {
            "All": {"R@100": 0.258, "R@10": 0.109, "N@100": 0.104, "N@10": 0.072},
            "Home": {"R@100": 0.278, "R@10": 0.081, "N@100": 0.091, "N@10": 0.047},
            "Clothing": {"R@100": 0.125, "R@10": 0.035, "N@100": 0.040, "N@10": 0.026},
            "Tools": {"R@100": 0.299, "R@10": 0.112, "N@100": 0.119, "N@10": 0.085},
        },
        "static_memory": {
            "All": {"R@100": 0.252, "R@10": 0.098, "N@100": 0.095, "N@10": 0.065},
            "Home": {"R@100": 0.237, "R@10": 0.076, "N@100": 0.071, "N@10": 0.041},
            "Clothing": {"R@100": 0.110, "R@10": 0.035, "N@100": 0.034, "N@10": 0.020},
            "Tools": {"R@100": 0.311, "R@10": 0.107, "N@100": 0.110, "N@10": 0.069},
        },
    }

    print("\n" + "="*70)
    print("COMPARISON WITH PAPER RESULTS (GPT-4o baseline, Table 1)")
    print("="*70)
    print(f"{'Setting':<20} {'Metric':<12} {'Ours':>8} {'Paper':>8} {'Match':>8}")
    print("-"*70)

    for setting, our_metrics in results.items():
        paper = PAPER_RESULTS.get(setting, {}).get("All", {})
        for metric in ["R@100", "R@10", "N@100", "N@10"]:
            ours = our_metrics.get(metric, 0)
            paper_val = paper.get(metric, 0)
            close = abs(ours - paper_val) < 0.02
            marker = "~OK" if close else "DIFF"
            print(f"{setting:<20} {metric:<12} {ours:>8.4f} {paper_val:>8.4f} {marker:>8}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MR.Rec baselines")
    parser.add_argument("--memory", choices=["no_memory", "naive_memory", "static_memory", "all"],
                        default="no_memory", help="Memory setting to evaluate")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--sample", type=int, default=200, help="Number of queries to evaluate")
    parser.add_argument("--categories", nargs="+", default=None, help="Categories to evaluate")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild FAISS indices")
    args = parser.parse_args()

    settings = ["no_memory", "naive_memory", "static_memory"] if args.memory == "all" else [args.memory]
    all_results = {}

    for setting in settings:
        overall, _ = run_evaluation(
            memory_setting=setting,
            model=args.model,
            sample_size=args.sample,
            categories=args.categories,
            force_rebuild_index=args.force_rebuild,
        )
        all_results[setting] = overall

    if len(all_results) > 1:
        compare_with_paper(all_results)
