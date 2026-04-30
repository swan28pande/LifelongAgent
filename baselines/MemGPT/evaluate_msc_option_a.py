"""
Option A: MemGPT evaluation on MSC-Self-Instruct using the official Letta SDK.

Requires:
  - Letta server running: docker run -d -p 8283:8283 -e OPENAI_API_KEY=<key> letta/letta:latest
  - pip install letta-client

Run: python evaluate_msc_option_a.py [--limit N]
"""

import os
import json
import time
import argparse
import openai
from dotenv import load_dotenv
from rouge_score import rouge_scorer

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
JUDGE_MODEL = "gpt-4o"

MSC_PATH    = os.path.join(os.path.dirname(__file__), "msc_data", "msc_self_instruct.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "eval_results_msc_option_a_gpt4o_500.json")
LETTA_BASE_URL = "http://localhost:8283"
LLM_MODEL      = "gpt-4o-mini"

RATE_LIMIT_SLEEP = 30  # seconds to sleep on 429
INTER_EXAMPLE_SLEEP = 5  # seconds between examples to avoid rate limits


def build_session_messages(example: dict) -> list[dict]:
    """
    Convert previous_dialogs into alternating user/assistant messages.
    Loaded one-by-one so Letta's queue manager can evict older messages
    to recall storage as the 8k FIFO window fills up.
    """
    messages = []
    for s_idx, session in enumerate(example["previous_dialogs"], start=1):
        for turn in session.get("dialog", []):
            speaker = turn.get("id", "")
            text    = turn.get("text", "").strip()
            if not text:
                continue
            role = "assistant" if "1" in speaker else "user"
            messages.append({"role": role, "content": f"[Session {s_idx}] {text}"})
    return messages


def llm_judge(question: str, prediction: str, ground_truth: str) -> int:
    """GPT-4o judge: returns 1 if prediction is consistent with ground truth, else 0."""
    prompt = (
        f"Question: {question}\n"
        f"Gold answer: {ground_truth}\n"
        f"Agent response: {prediction}\n\n"
        "Is the agent response consistent with the gold answer? "
        "Answer only 'yes' or 'no'."
    )
    try:
        resp = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        verdict = resp.choices[0].message.content.strip().lower()
        return 1 if verdict.startswith("yes") else 0
    except Exception:
        return 0


def score(question: str, prediction: str, ground_truth: str) -> dict:
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = rs.score(ground_truth, prediction)["rougeL"].recall
    accuracy = llm_judge(question, prediction, ground_truth)
    return {"accuracy": accuracy, "rouge_l_recall": round(rouge_l, 4)}


def call_with_retry(fn, *args, max_retries=8, **kwargs):
    """Call fn(*args, **kwargs), retrying on rate limit (429) with backoff."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = RATE_LIMIT_SLEEP * (attempt + 1)
                print(f"    [Rate limit] sleeping {wait}s (attempt {attempt+1}): {str(e)[:200]}")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Max retries exceeded for {fn}")


def evaluate(limit: int = None, output_path: str = OUTPUT_PATH):
    from letta_client import Letta

    with open(MSC_PATH) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]

    client = Letta(base_url=LETTA_BASE_URL)
    print(f"Connected to Letta server at {LETTA_BASE_URL}")
    print(f"Evaluating on {len(data)} examples with {LLM_MODEL}...")

    all_tools = client.tools.list()
    tool_map  = {t.name: t.id for t in all_tools}
    tool_ids  = [
        tool_map["archival_memory_search"],
        tool_map["archival_memory_insert"],
        tool_map["send_message"],
        tool_map["conversation_search"],
    ]

    results = []
    total_acc, total_rouge = 0.0, 0.0

    for i, example in enumerate(data):
        question = example["self_instruct"]["B"]
        gt       = example["self_instruct"]["A"]

        personas    = example.get("init_personas", [])
        persona_str = " ".join(personas[0]) if personas else "I am a conversational agent."

        agent = call_with_retry(
            client.agents.create,
            name=f"msc_agent_{i}",
            model=f"openai/{LLM_MODEL}",
            embedding="openai/text-embedding-ada-002",
            tool_ids=tool_ids,
            context_window_limit=8192,  # simulate paper's 8k window — forces FIFO overflow to recall storage
            memory_blocks=[
                {"label": "human",   "value": "Speaker 2 (the user). Context: multi-session conversation."},
                {"label": "persona", "value": f"You are Speaker 1. Your persona: {persona_str}"},
            ],
        )
        agent_id = agent.id

        session_messages = build_session_messages(example)
        if session_messages:
            call_with_retry(
                client.agents.messages.create,
                agent_id=agent_id,
                messages=session_messages,
            )

        response = call_with_retry(
            client.agents.messages.create,
            agent_id=agent_id,
            messages=[{"role": "user", "content": question}],
        )

        prediction = ""
        for msg in response.messages:
            msg_type = getattr(msg, "message_type", None) or getattr(msg, "role", None)
            if msg_type in ("assistant_message", "assistant"):
                content = getattr(msg, "content", "") or ""
                if isinstance(content, str) and content.strip():
                    prediction = content.strip()
                    break

        metrics = score(question, prediction, gt)
        total_acc   += metrics["accuracy"]
        total_rouge += metrics["rouge_l_recall"]

        results.append({
            "id": i, "question": question, "ground_truth": gt,
            "prediction": prediction, **metrics,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(data)}] Acc: {total_acc/(i+1):.3f} | ROUGE-L R: {total_rouge/(i+1):.3f}")
        print(f"    Q: {question[:70]}...")
        print(f"    GT: {gt[:50]} | Pred: {prediction[:50]}...")

        call_with_retry(client.agents.delete, agent_id=agent_id)
        time.sleep(INTER_EXAMPLE_SLEEP)

        # Save checkpoint every 10 examples
        if (i + 1) % 10 == 0:
            n = len(results)
            checkpoint = {
                "aggregate": {
                    "n": n,
                    "accuracy": round(total_acc / n, 4),
                    "rouge_l_recall": round(total_rouge / n, 4),
                },
                "results": results,
            }
            with open(output_path, "w") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    n = len(results)
    aggregate = {
        "n": n,
        "accuracy": round(total_acc / n, 4),
        "rouge_l_recall": round(total_rouge / n, 4),
    }
    output = {"aggregate": aggregate, "results": results}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Accuracy:    {aggregate['accuracy']:.4f}")
    print(f"ROUGE-L R:   {aggregate['rouge_l_recall']:.4f}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    evaluate(limit=args.limit, output_path=args.output)
