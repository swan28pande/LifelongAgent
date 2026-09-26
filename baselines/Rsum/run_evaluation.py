"""
Streamlined evaluation for ChatGPT-Rsum on MSC session 5.

Runs:
  1. ChatGPT-Rsum   (recursive memory → response)
  2. ChatGPT-only   (context-only, no memory)

Uses ThreadPoolExecutor to parallelize memory-update API calls.
Computes F1 and BLEU-1/2. Compares to paper Table 3.
"""

import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from openai import OpenAI
from tqdm import tqdm

# Paper results — Table 3, MSC dataset
PAPER_RESULTS = {
    "ChatGPT (context-only)": {"F1": 19.41, "BLEU1": 21.23, "BLEU2": 12.24},
    "ChatGPT-BM25 (k=3)":    {"F1": 19.56, "BLEU1": 21.60, "BLEU2": 12.46},
    "ChatGPT-MemoryBank":    {"F1": 20.28, "BLEU1": 21.82, "BLEU2": 12.58},
    "ChatGPT-Rsum (Ours)":   {"F1": 20.48, "BLEU1": 21.83, "BLEU2": 12.59},
}

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client


def call_llm(prompt, model, system, max_retries=5):
    client = get_client()
    for i in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(3)
    return ""


def build_memory_for_dialogue(prev_list, model, prompts):
    """
    Recursively build memory M_N from N past sessions.
    M_0 = "Empty"
    M_i = LLM(S_i, M_{i-1})
    """
    instruction = prompts["msc"]["gpt-3.5-turbo"]["update_memory"]
    system = "You are an advanced AI language model with the ability to keep track of dialog information between speakers."
    memory = "Empty"
    for session_text in prev_list:
        prompt = (
            f"**Instruction** {instruction}\n"
            f"**Test** [Previous Memory] {memory} "
            f"[Dialogue Context] {session_text} [Updated Memory]"
        )
        result = call_llm(prompt, model, system)
        for prefix in ["Updated memory:", "[Updated Memory]"]:
            if prefix in result:
                result = result.split(prefix)[-1]
        memory = result.replace("\n", " ").strip() or memory
    return memory


def generate_response(memory, context, model, prompts):
    instruction = prompts["msc"]["gpt-3.5-turbo"]["update_response"]
    system = "You are an advanced AI language model designed to engage in personality-based conversations."
    prompt = (
        f"**Instruction** {instruction}\n"
        f"**Test** [Previous Memory] {memory} [Dialogue Context] {context} [Response]\n"
    )
    result = call_llm(prompt, model, system)
    return result.replace("\n", " ").replace("System:", "").strip()


def generate_direct_response(context, model, prompts):
    instruction = prompts["msc"]["gpt-3.5-turbo"]["direct_response"]
    system = "You are an advanced AI language model designed to engage in personality-based conversations."
    prompt = (
        f"**Instruction** {instruction}\n"
        f"**Test** [Dialogue Context] {context} [Response]"
    )
    result = call_llm(prompt, model, system)
    return result.replace("\n", " ").replace("System:", "").strip()


def group_by_dialogue(test_data, max_dialogs):
    """Group test turns by dialogue, return first max_dialogs."""
    dialogs = {}
    order = []
    for item in test_data:
        dial_id = item["dial_id"].split("-")[0]
        if dial_id not in dialogs:
            if len(dialogs) >= max_dialogs:
                break
            dialogs[dial_id] = {"prev_list": item["prev_list"], "turns": []}
            order.append(dial_id)
        dialogs[dial_id]["turns"].append(item)
    return [(did, dialogs[did]) for did in order]


def normalize(s):
    import re
    s = s.lower()
    s = re.sub(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']', ' ', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def compute_f1(preds, refs):
    total = 0
    for pred, ref in zip(preds, refs):
        p_toks = word_tokenize(normalize(pred))
        r_toks = word_tokenize(normalize(ref))
        common = collections.Counter(p_toks) & collections.Counter(r_toks)
        n = sum(common.values())
        if not p_toks or not r_toks or n == 0:
            continue
        prec, rec = n / len(p_toks), n / len(r_toks)
        total += 2 * prec * rec / (prec + rec)
    return total / len(preds) * 100


def compute_bleu(preds, refs):
    smooth = SmoothingFunction()
    hyps = [word_tokenize(p.lower()) for p in preds]
    ref_toks = [[word_tokenize(r.lower())] for r in refs]
    b1 = corpus_bleu(ref_toks, hyps, weights=(1,0,0,0), smoothing_function=smooth.method7) * 100
    b2 = corpus_bleu(ref_toks, hyps, weights=(0.5,0.5,0,0), smoothing_function=smooth.method7) * 100
    return b1, b2


def run_rsum(dialogs, model, prompts, n_workers=4):
    """Build memory in parallel, then generate responses."""
    print(f"\n[Rsum] Building memory for {len(dialogs)} dialogues (parallel, {n_workers} workers)...")

    # Step 1: build memory for each dialogue in parallel
    memories = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(build_memory_for_dialogue, d["prev_list"], model, prompts): did
            for did, d in dialogs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building memory"):
            did = futures[future]
            memories[did] = future.result()

    # Step 2: generate responses for all turns
    print(f"[Rsum] Generating responses for all turns...")
    preds, refs = [], []
    turn_tasks = []
    for did, d in dialogs:
        for turn in d["turns"]:
            turn_tasks.append((memories[did], turn["window"], turn["label"]))

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(generate_response, mem, ctx, model, prompts)
            for mem, ctx, _ in turn_tasks
        ]
        results = []
        for f in tqdm(as_completed(futures), total=len(futures), desc="Rsum responses"):
            results.append(f.result())

    # futures don't preserve order when using as_completed — use map instead
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(lambda t: generate_response(t[0], t[1], model, prompts), turn_tasks),
            total=len(turn_tasks), desc="Rsum responses"
        ))

    preds = results
    refs = [t[2] for t in turn_tasks]
    return preds, refs


def run_context_only(dialogs, model, prompts, n_workers=4):
    print(f"\n[Context-only] Generating responses for all turns...")
    turn_tasks = [(turn["window"], turn["label"]) for _, d in dialogs for turn in d["turns"]]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(lambda t: generate_direct_response(t[0], model, prompts), turn_tasks),
            total=len(turn_tasks), desc="Context-only responses"
        ))
    preds = results
    refs = [t[1] for t in turn_tasks]
    return preds, refs


def report(name, preds, refs):
    f1 = compute_f1(preds, refs)
    b1, b2 = compute_bleu(preds, refs)
    paper = PAPER_RESULTS.get(name, {})
    print(f"\n{'─'*60}")
    print(f"  {name}   (n={len(preds)} turns)")
    print(f"{'─'*60}")
    print(f"  F1:     {f1:6.2f}   (paper: {paper.get('F1','?')})")
    print(f"  BLEU-1: {b1:6.2f}   (paper: {paper.get('BLEU1','?')})")
    print(f"  BLEU-2: {b2:6.2f}   (paper: {paper.get('BLEU2','?')})")
    return {"name": name, "F1": f1, "BLEU1": b1, "BLEU2": b2, "n": len(preds)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", type=int, default=5)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_dialogs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--mode", choices=["rsum", "context_only", "both"], default="both")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Load API key from .env
    env_path = os.path.join(os.path.dirname(__file__), "../../.env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip('"')

    # Load data
    sys.path.insert(0, ".")
    import argparse as ap2
    from chatgpt.data_loader import prepare_test_data
    cfg = ap2.Namespace(
        dataset="msc", session_id=args.session_id, mode="rsum",
        model_name=args.model, n_shot=1, random_seed=42, do_ict=False,
        summary_size=200, window_size=2000, target_size=200,
        resp_temp=0, summ_temp=0, max_seq_length=4000,
        eval_file="", summary_file="", operation="infer",
        do_rag=False, do_sample=False, retrieval="b25",
        saving_dir="save_msc", test_num=300, topk=5,
        summary_model_name="gpt-3.5-turbo-0301",
        train_batch_size=2, dev_batch_size=2, test_batch_size=4,
        nopersona_subsampling_weight=0, summary_num_turns=7,
        learning_rate=2e-5, gradient_accumulation_steps=8,
        n_epoch=3, load_path="", trainer="summarizer", summary_type="pred",
    )
    setattr(cfg, "local-rank", -1)
    print(f"Loading MSC session {args.session_id} test data...")
    test_data, _ = prepare_test_data(cfg)
    print(f"Loaded {len(test_data)} turns total")

    prompts = json.load(open("prompt.json"))
    dialogs = group_by_dialogue(test_data, args.max_dialogs)
    total_turns = sum(len(d["turns"]) for _, d in dialogs)
    print(f"Evaluating {len(dialogs)} dialogues, {total_turns} turns")

    all_results = []

    if args.mode in ("rsum", "both"):
        preds, refs = run_rsum(dialogs, args.model, prompts, args.workers)
        all_results.append(report("ChatGPT-Rsum (Ours)", preds, refs))
        with open("results/rsum_predictions.json", "w") as f:
            json.dump({"predictions": preds, "references": refs}, f, indent=2)

    if args.mode in ("context_only", "both"):
        preds, refs = run_context_only(dialogs, args.model, prompts, args.workers)
        all_results.append(report("ChatGPT (context-only)", preds, refs))
        with open("results/context_only_predictions.json", "w") as f:
            json.dump({"predictions": preds, "references": refs}, f, indent=2)

    print(f"\n{'='*60}")
    print("PAPER TABLE 3 REFERENCE (ChatGPT backbone, MSC session 5):")
    print(f"{'='*60}")
    for name, vals in PAPER_RESULTS.items():
        print(f"  {name:<35} F1={vals['F1']:.2f}  B1={vals['BLEU1']:.2f}  B2={vals['BLEU2']:.2f}")

    with open("results/evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/evaluation_results.json")
