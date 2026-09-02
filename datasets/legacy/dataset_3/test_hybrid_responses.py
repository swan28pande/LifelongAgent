import os
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Add memory/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))
from memory_manager import MemoryManager
from structured_memory_pref import StructuredMemoryManagerPref
from hybrid_synthesizer import HybridMemorySynthesizer

def main():
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    db_path      = os.path.join(base_dir, "lifelong_memory.db")
    index_path   = os.path.join(base_dir, "faiss_index")
    
    # Initialize Managers
    sql_manager  = StructuredMemoryManagerPref(db_path=db_path)
    rag_manager  = MemoryManager(index_path=index_path)
    synthesizer  = HybridMemorySynthesizer(sql_manager, rag_manager)
    
    # Standard AI Assistant model
    assistant_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # Test Queries
    queries = [
        "What kind of coffee should I get today? Any recommendations?",
        "I was thinking about my shoes... what did I like recently?",
        "I used to love espresso, why am I drinking latte now? Can you check my past thoughts?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}\n[USER QUERY]: {query}\n{'='*60}")
        
        # 1. Synthesize Hybrid Context
        print("  ...Synthesizing memory context (SQL + RAG)...")
        context = synthesizer.get_assistant_context(query)
        
        if context:
            print(f"  [CONTEXT RETRIEVED]:\n{context}")
        else:
            print("  [NO RELEVANT MEMORY FOUND]")
        
        # 2. Generate Assistant Response
        print("  ...Generating personalized response...")
        assistant_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a personalized AI assistant. Use the provided user memory briefs to give a detailed, supportive, and fact-based response."),
            ("human", "CONTEXT:\n{context}\n\nUSER MESSAGE: {query}")
        ])
        
        chain = assistant_prompt | assistant_llm
        response = chain.invoke({"context": context, "query": query})
        
        print(f"\n[ASSISTANT]: {response.content}")

if __name__ == "__main__":
    main()
