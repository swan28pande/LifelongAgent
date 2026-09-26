# baselines

Third-party systems and datasets. Most are upstream code with light local additions
(evaluation scripts, result files); see each folder's own README where there is one.

| Folder | What it is | Used by |
|---|---|---|
| `locomo/` | LoCoMo benchmark repo (Snap Research, ACL 2024). **`data/locomo10.json` is the dataset every LoCoMo runner reads.** | `benchmarks/locomo/`, `evaluation/evaluate_locomo.py` |
| `Rsum/` | Recursive summarisation for long-term dialogue (a git submodule). `results/` holds its MSC predictions. | `evaluation/evaluate_msc.py`, `evaluation/evaluate_synthetic.py` |
| `MemGPT/` | MemGPT/Letta evaluated on MSC in two ways (Letta SDK, and a manual reimplementation), with result files. | `evaluation/evaluate_synthetic.py` |
| `MemoryBank-SiliconFriend/` | MemoryBank (forgetting-curve memory) and SiliconFriend, plus a probing evaluation. | `evaluation/evaluate_synthetic.py` |
| `MrRec/` | Memory-augmented LLM recommender: memory builder, retriever, metrics. | reference |
| `NaiveRAG/` | Plain retrieval-augmented baseline (`naive_rag.py`). | `evaluation/evaluate_synthetic.py` |

mem0 is not vendored here. It is installed as a package (`pip install mem0ai`) and run
by `benchmarks/locomo/run_mem0.py`. Its source: github.com/mem0ai/mem0; its own
benchmark suite: github.com/mem0ai/memory-benchmarks.
