"""
Embedding-based retrieval for MR.Rec evaluation.

Builds FAISS indices per category for efficient top-k retrieval.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INDEX_DIR = os.path.join(os.path.dirname(__file__), "indices")
DEFAULT_MODEL = "BAAI/bge-m3"  # paper uses Qwen3-Embedding-0.6B; bge-m3 is comparable


class ItemRetriever:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 256):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.indices = {}       # category → faiss.Index
        self.item_ids = {}      # category → list of item_ids (ordered)
        os.makedirs(INDEX_DIR, exist_ok=True)

    def _index_path(self, category: str):
        safe = category.replace("/", "_").replace(" ", "_")
        return os.path.join(INDEX_DIR, f"{safe}.faiss"), os.path.join(INDEX_DIR, f"{safe}_ids.pkl")

    def build_index(self, category: str, items: list, force_rebuild: bool = False):
        """
        Build FAISS index for a category's items.

        items: list of {item_id, metadata}
        """
        idx_path, ids_path = self._index_path(category)

        if not force_rebuild and os.path.exists(idx_path) and os.path.exists(ids_path):
            print(f"Loading cached index for {category} ({len(items)} items)")
            self.indices[category] = faiss.read_index(idx_path)
            with open(ids_path, "rb") as f:
                self.item_ids[category] = pickle.load(f)
            return

        print(f"Building index for {category} ({len(items)} items)...")
        texts = [item["metadata"][:512] for item in items]
        ids = [item["item_id"] for item in items]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity (normalized)
        index.add(embeddings)

        faiss.write_index(index, idx_path)
        with open(ids_path, "wb") as f:
            pickle.dump(ids, f)

        self.indices[category] = index
        self.item_ids[category] = ids
        print(f"Index built: {len(ids)} items, dim={dim}")

    def retrieve(self, query_text: str, category: str, top_k: int = 1000) -> list:
        """
        Retrieve top-k item_ids for a query in a given category.
        Returns list of (item_id, score) tuples.
        """
        if category not in self.indices:
            raise ValueError(f"No index for category '{category}'. Call build_index first.")

        q_emb = self.model.encode(
            [query_text],
            normalize_embeddings=True,
        ).astype(np.float32)

        k = min(top_k, self.indices[category].ntotal)
        scores, indices = self.indices[category].search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.item_ids[category][idx], float(score)))
        return results

    def batch_retrieve(self, query_texts: list, category: str, top_k: int = 1000) -> list:
        """Retrieve for multiple queries at once. Returns list of result lists."""
        if category not in self.indices:
            raise ValueError(f"No index for category '{category}'.")

        q_embs = self.model.encode(
            query_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        k = min(top_k, self.indices[category].ntotal)
        scores, indices = self.indices[category].search(q_embs, k)

        all_results = []
        for s_row, i_row in zip(scores, indices):
            row = [(self.item_ids[category][idx], float(s)) for s, idx in zip(s_row, i_row) if idx >= 0]
            all_results.append(row)
        return all_results
