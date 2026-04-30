"""Data loading utilities for MR.Rec evaluation on Amazon-C4."""

import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ITEM_META_FILE = os.path.join(DATA_DIR, "sampled_item_metadata_1M.jsonl")

# Categories evaluated in the paper
PAPER_CATEGORIES = {"Home", "Clothing", "Tools"}  # plus "All"


def download_data():
    """Download Amazon-C4 data if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ITEM_META_FILE):
        print("Downloading item metadata (~614MB)...")
        hf_hub_download(
            repo_id="McAuley-Lab/Amazon-C4",
            filename="sampled_item_metadata_1M.jsonl",
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
    print(f"Item metadata: {ITEM_META_FILE}")


def load_queries(sample_size: int = None, seed: int = 42, categories: list = None):
    """
    Load Amazon-C4 test queries.

    Returns DataFrame with columns: qid, query, item_id, user_id, ori_rating, ori_review, category
    """
    ds = load_dataset("McAuley-Lab/Amazon-C4", split="test")
    df = ds.to_pandas()

    # Add category from item metadata
    item_cat = load_item_categories()
    df["category"] = df["item_id"].map(item_cat)

    if categories:
        df = df[df["category"].isin(categories)].reset_index(drop=True)

    if sample_size and sample_size < len(df):
        rng = random.Random(seed)
        # Sample proportionally across categories
        sampled = df.groupby("category", group_keys=False).apply(
            lambda g: g.sample(
                n=max(1, int(sample_size * len(g) / len(df))),
                random_state=seed,
            )
        )
        df = sampled.sample(n=min(sample_size, len(sampled)), random_state=seed).reset_index(drop=True)

    print(f"Loaded {len(df)} queries, {df['user_id'].nunique()} unique users")
    print(f"Category distribution:\n{df['category'].value_counts().to_string()}")
    return df


def load_item_categories():
    """Load item_id → category mapping."""
    item_cat = {}
    with open(ITEM_META_FILE) as f:
        for line in f:
            ex = json.loads(line)
            item_cat[ex["item_id"]] = ex["category"]
    return item_cat


def load_items_by_category():
    """
    Load all items grouped by category.
    Returns dict: category → list of {item_id, metadata}
    """
    items_by_cat = defaultdict(list)
    print("Loading item metadata...")
    with open(ITEM_META_FILE) as f:
        for line in tqdm(f, desc="Loading items", total=1058417):
            ex = json.loads(line)
            items_by_cat[ex["category"]].append({
                "item_id": ex["item_id"],
                "metadata": ex["metadata"],
            })
    return dict(items_by_cat)


def load_items_flat():
    """Load all items as flat list. Returns (item_ids list, metadatas list, categories list)."""
    item_ids, metadatas, categories = [], [], []
    with open(ITEM_META_FILE) as f:
        for line in tqdm(f, desc="Loading items", total=1058417):
            ex = json.loads(line)
            item_ids.append(ex["item_id"])
            metadatas.append(ex["metadata"])
            categories.append(ex["category"])
    return item_ids, metadatas, categories


def build_user_history(df: pd.DataFrame) -> dict:
    """
    Build per-user interaction history from the available Amazon-C4 reviews.
    Groups all reviews by user_id. For test queries, excludes the target item.
    Returns dict: user_id → list of {item_id, review, rating, category}
    """
    history = defaultdict(list)
    for _, row in df.iterrows():
        history[row["user_id"]].append({
            "item_id": row["item_id"],
            "review": row["ori_review"],
            "rating": row["ori_rating"],
            "category": row["category"],
        })
    return dict(history)


def get_candidate_pool(items_by_category: dict, query_category: str) -> list:
    """Get item candidate pool for a given query category."""
    return items_by_category.get(query_category, [])
