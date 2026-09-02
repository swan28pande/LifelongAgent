import os
import sys
import json
import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")

# Add memory/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))
from structured_memory import StructuredMemoryManager

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "lifelong_memory.db")
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}.")
        return

    manager = StructuredMemoryManager(db_path=db_path)
    
    print("--- STEP 1: Generating Insightful Questions ---")
    questions = manager.generate_insight_questions()
    if not questions:
        print("Failed to generate questions.")
        return
        
    for i, q in enumerate(questions):
        print(f"{i+1}. {q}")

    print("\n--- STEP 2: Executing AI-Driven SQL Queries & Summarizing ---")
    final_insights = []
    for q in questions[:10]: # Limit to 10
        print(f"\nQuestion: {q}")
        res = manager.execute_ai_sql(q)
        if "error" in res:
            print(f"  Error: {res['error']}")
            continue
            
        print(f"  SQL: {res['sql']}")
        print(f"  Found {len(res['results'])} results.")
        
        # Generate a specific summary for this question's data
        system_prompt = (
            "You are a helpful assistant analyzing a specific part of a user's memory database. "
            "Given a question and the data retrieved from the database, provide a concise, "
            "one-sentence summary of the finding."
        )
        
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Question: {question}\nData: {data}")
        ])
        
        try:
            chain = prompt | manager.llm
            summary = chain.invoke({
                "question": q, 
                "data": json.dumps(res['results'][:20]) # Limit data to avoid context overflow
            }).content.strip()
            print(f"  Summary: {summary}")
            
            final_insights.append({
                "question": q,
                "sql": res['sql'],
                "summary": summary
            })
        except Exception as e:
            print(f"  Error generating summary: {e}")

    print("\n--- STEP 3: Saving Structured Insights to JSON ---")
    if not final_insights:
        print("No insights to save.")
        return

    output_path = os.path.join(base_dir, "structured_insights.json")
    with open(output_path, "w") as f:
        json.dump(final_insights, f, indent=2)
    
    print(f"Successfully saved {len(final_insights)} insights to {output_path}")

if __name__ == "__main__":
    main()
