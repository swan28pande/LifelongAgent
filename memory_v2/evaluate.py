import argparse
import json
import os
import string
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate

# Ensure we can import from memory_v2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_v2.agent import LifelongAgent

def token_f1(pred: str, true: str) -> float:
    pred_tokens = pred.lower().translate(str.maketrans('', '', string.punctuation)).split()
    true_tokens = true.lower().translate(str.maketrans('', '', string.punctuation)).split()
    common = set(pred_tokens).intersection(set(true_tokens))
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(true_tokens)
    return 2 * (prec * rec) / (prec + rec)

def llm_judge(agent, query: str, pred: str, ref: str) -> float:
    prompt_text = f"""
Query: {query}
Ground Truth: {ref}
System Prediction: {pred}

Is the System Prediction semantically correct and factually consistent with the Ground Truth? 
Ignore minor phrasing differences or verbosity. 

EVALUATION GUIDELINES:
- Temporal Evolution: Facts can evolve over time. If the Ground Truth describes a state as current/upcoming but the System Prediction describes it as past/completed (or vice-versa), treat it as consistent if they refer to the same event.
- State Changes: If the Ground Truth describes a historical preference or state that subsequently changed, the prediction is correct if it lists either the relevant state or mentions the transition.
- List/Group Overlap: If the Ground Truth is a single item and the System Prediction provides a category, list, or set of options that clearly contains the Ground Truth item, it is correct.
- Extra Details: Do not penalize the System Prediction for including extra reasoning, context, or details, as long as the correct core information is present.

If the System says "I don't know" but the Ground Truth has a correct answer, it is WRONG.
If the System provides the correct core information (even if embedded in a larger correct context or list), it is RIGHT.

Return only "RIGHT" or "WRONG".
"""
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a strict grading assistant."),
            ("human", prompt_text)
        ])
        chain = prompt | agent.chat_llm
        response = chain.invoke({})
        
        if isinstance(response.content, list):
            result = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in response.content]).strip().upper()
        else:
            result = str(response.content).strip().upper()
            
        return 1.0 if "RIGHT" in result else 0.0
    except Exception as e:
        print(f"LLM Judge error: {e}")
        return 0.0

def run_evaluation(store_path: str, qa_path: str, model: str):
    if not os.path.exists(qa_path):
        print(f"Error: QA file not found at {qa_path}")
        sys.exit(1)

    print(f"\n--- PHASE 3: EVALUATING QA PAIRS ---")
    print(f"Loading QA dataset from {qa_path}...")
    with open(qa_path) as f:
        qa_data = json.load(f)["user_1"]
        qa_pairs = qa_data.get("qa_pairs", [])

    print(f"Initializing LifelongAgent with store at {store_path} using model {model}...")
    agent = LifelongAgent(base_dir=store_path, chat_model=model, extract_model=model)

    print(f"Answering {len(qa_pairs)} questions...")

    all_f1_scores = []
    all_llm_scores = []
    diff_f1_scores = defaultdict(list)
    diff_llm_scores = defaultdict(list)
    type_f1_scores = defaultdict(list)
    type_llm_scores = defaultdict(list)
    records = []

    for qa in tqdm(qa_pairs, desc="Evaluating QA"):
        q    = qa["question"]
        a    = qa["answer"]
        diff = qa["difficulty"]
        qtype = qa["type"]

        resp  = agent.chat(q)
        f1_score = token_f1(resp, a)
        llm_score = llm_judge(agent, q, resp, a)

        all_f1_scores.append(f1_score)
        all_llm_scores.append(llm_score)
        
        diff_f1_scores[diff].append(f1_score)
        diff_llm_scores[diff].append(llm_score)
        
        type_f1_scores[qtype].append(f1_score)
        type_llm_scores[qtype].append(llm_score)

        records.append({
            "question": q, "answer": a,
            "difficulty": diff, "type": qtype,
            "response": resp, "f1_score": f1_score, "llm_score": llm_score,
        })
        
    print(f"\n{'='*75}")
    print(f"{'QA EVALUATION RESULTS':<25} | {'F1 SCORE':<15} | {'LLM JUDGE':<15}")
    print(f"{'='*75}")
    print(f"  {'Overall (n='+str(len(all_f1_scores))+')':<23} | {np.mean(all_f1_scores):<15.3f} | {np.mean(all_llm_scores):<15.3f}")
    
    print("\n  By difficulty:")
    for diff in ("simple", "mid", "difficult"):
        vals_f1 = diff_f1_scores[diff]
        vals_llm = diff_llm_scores[diff]
        if vals_f1:
            print(f"    {diff:<21} | {np.mean(vals_f1):<15.3f} | {np.mean(vals_llm):<15.3f}")
            
    print("\n  By type:")
    for qtype in ("factual", "factual_evolving", "recall", "pattern_id", "prediction", "transition"):
        vals_f1 = type_f1_scores[qtype]
        vals_llm = type_llm_scores[qtype]
        if vals_f1:
            print(f"    {qtype:<21} | {np.mean(vals_f1):<15.3f} | {np.mean(vals_llm):<15.3f}")
            
    run_dir = os.path.dirname(store_path)
    qa_output = os.path.join(run_dir, "qa_results.json")
    with open(qa_output, "w") as f:
        json.dump({
            "summary": {
                "overall_f1": np.mean(all_f1_scores),
                "overall_llm": np.mean(all_llm_scores)
            }, 
            "records": records
        }, f, indent=2)
    print(f"\nDetailed QA results saved to {qa_output}")

def evaluate_cli():
    parser = argparse.ArgumentParser(description="Evaluate memory_v2 QA performance")
    parser.add_argument("--store_dir", type=str, default="../results/gemini_extraction_run/store", help="Path to the agent's store directory")
    parser.add_argument("--qa_file", type=str, default="../datasets/eval/qa_pairs.json", help="Path to the QA pairs JSON file")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite", help="Model to use for chatting and judging")
    
    args = parser.parse_args()

    # Resolve absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qa_path = os.path.join(base_dir, args.qa_file.lstrip("../"))
    store_path = os.path.join(base_dir, args.store_dir.lstrip("../"))

    run_evaluation(store_path, qa_path, args.model)

if __name__ == "__main__":
    evaluate_cli()
