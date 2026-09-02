import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")

import re
from datetime import datetime

# Allow importing MemoryManager from the shared memory/ module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))
from memory_manager import MemoryManager

def run_test_suite(question: str, n_results: int = 2):
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    tests_dir  = os.path.join(base_dir, "tests")
    os.makedirs(tests_dir, exist_ok=True)

    combined_output = []
    
    # 1. Standard Query
    combined_output.extend(test_query(question, n_results, mode="standard"))
    combined_output.append("\n" + "="*50 + "\n")
    
    # 2. Hybrid Query
    combined_output.extend(test_query(question, n_results, mode="hybrid"))
    combined_output.append("\n" + "="*50 + "\n")

    # 3. Hierarchical Query
    combined_output.extend(test_query(question, n_results, mode="hierarchical"))

    # Save to file
    slug = re.sub(r'[^a-z0-9]+', '_', question.lower()).strip('_')[:50]
    filename = f"test_{slug}.txt"
    filepath = os.path.join(tests_dir, filename)
    
    with open(filepath, "w") as f:
        f.write("\n".join(combined_output))
    print(f"\nFinal consolidated results saved to: {filepath}")

def test_query(question: str, n_results: int = 2, mode: str = "standard"):
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "faiss_index")

    output = []
    mode_display = mode.capitalize()
    header = f"Querying dataset_1 [{mode_display}]: {question}"
    print(f"\n{header}")
    output.append(header)
    
    manager = MemoryManager(index_path=index_path)
    if mode == "hierarchical":
        results = manager.query_hierarchical(question, k_days=n_results)
    elif mode == "hybrid":
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
    
    return output

if __name__ == "__main__":
    query = "What did the user prefer in the afternoon of 2026-02-19?"
    run_test_suite(query)
