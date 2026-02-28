from memory_manager import MemoryManager

def test_query(question):
    print(f"\nQuerying: {question}")
    manager = MemoryManager(index_path="memory/faiss_index")
    results = manager.query(question, n_results=2)
    
    if not results:
        print("No results found.")
        return
        
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} (Date: {doc.metadata.get('date')}, Time: {doc.metadata.get('time_of_day')}) ---")
        print(doc.page_content)

if __name__ == "__main__":
    # Test with a question about a specific activity
    test_query("What did I do on 2026-02-15?")
    test_query("When did I wear my fitness band for the first time?")
