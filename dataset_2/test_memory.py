import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")

import re
from datetime import datetime

# Allow importing MemoryManager from the shared memory/ module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))
from memory_manager import MemoryManager

def test_query(question: str, n_results: int = 2, hybrid: bool = False):
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "faiss_index")
    tests_dir  = os.path.join(base_dir, "tests")
    os.makedirs(tests_dir, exist_ok=True)

    output = []
    mode_str = "Hybrid (Relevance+Recency)" if hybrid else "Standard (Relevance)"
    header = f"Querying dataset_2 [{mode_str}]: {question}"
    print(f"\n{header}")
    output.append(header)
    
    manager = MemoryManager(index_path=index_path)
    if hybrid:
        results = manager.query_hybrid(question, k=n_results)
    else:
        results = manager.query(question, n_results=n_results)

    if not results:
        msg = "No results found. Have you run ingest_memory.py yet?"
        print(msg)
        output.append(msg)
    else:
        for i, doc in enumerate(results):
            res_header = f"\n--- Result {i+1} (Date: {doc.metadata.get('date')}, Time: {doc.metadata.get('time_of_day')}) ---"
            print(res_header)
            print(doc.page_content)
            output.append(res_header)
            output.append(doc.page_content)

    # Save to file
    slug = re.sub(r'[^a-z0-9]+', '_', question.lower()).strip('_')[:50]
    filename = f"test_{slug}.txt"
    filepath = os.path.join(tests_dir, filename)
    
    with open(filepath, "w") as f:
        f.write("\n".join(output))
    print(f"\nResults saved to: {filepath}")

if __name__ == "__main__":
    query = "List Down the dates when the user preferred pasta?"
    # Compare both
    test_query(query, hybrid=False)
    test_query(query, hybrid=True)
