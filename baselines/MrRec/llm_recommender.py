"""
LLM-based ideal item profile generation for MR.Rec baselines.

Implements the Rec-R1 style prompt: given a query (+ optional memory),
generate an ideal item profile for dense retrieval.
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# From paper Appendix B.2 (Rec-R1 framework prompt)
BASE_SYSTEM_PROMPT = (
    "You are an expert in generating queries for dense retrieval. "
    "Given a customer query, your task is to retain the original query while "
    "expanding it with additional semantically relevant information, retrieve the "
    "most relevant products, ensuring they best meet customer needs. "
    "If no useful expansion is needed, return the original query as is. "
    "# Below is the product search query: {query}"
)

# Full ideal-item-profile prompt used by LLM baselines
PROFILE_SYSTEM = (
    "You are an expert in generating ideal item profiles for product recommendation. "
    "Given a user query and optional user memory, generate a detailed ideal item profile "
    "that captures what the ideal product should look like. The profile should include "
    "key attributes, features, and characteristics that would make the product a perfect match."
)


def generate_item_profile(
    query: str,
    memory_text: str = "",
    model: str = "gpt-4o-mini",
    memory_setting: str = "no_memory",
) -> str:
    """
    Generate ideal item profile for retrieval.

    memory_setting: "no_memory" | "naive_memory" | "static_memory" | "local_memory"
    """
    if memory_setting == "no_memory" or not memory_text:
        user_msg = (
            f"User Query: {query}\n\n"
            "Generate an ideal item profile for this query. "
            "Describe the key attributes and features of the perfect product:"
        )
    else:
        user_msg = (
            f"User Query: {query}\n\n"
            f"User Memory:\n{memory_text}\n\n"
            "Based on the user query and their memory/preferences, generate an ideal item profile. "
            "The profile should reflect both the query requirements and the user's known preferences:"
        )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROFILE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


def simplify_query(query: str, model: str = "gpt-4o-mini") -> str:
    """
    Simplify an Amazon-C4 query by removing overly specific details.
    Mimics the GPT-o3-mini query simplification from Appendix A.
    """
    prompt = (
        'Rewrite the following product query into a single, casual, conversational question. '
        'Rules:\n'
        '- Keep: 1 core item + 1 core use case (e.g., "a laptop for gaming").\n'
        '- Delete: All other details (including brand, specs, price, personal preferences, etc.).\n'
        '- Tone: Natural and conversational.\n'
        '- Format: Must be a single sentence.\n\n'
        f'Original Query: {query}\n\n'
        'Rewritten Query:'
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()


def batch_generate_profiles(
    queries: list,
    memories: list = None,
    model: str = "gpt-4o-mini",
    memory_setting: str = "no_memory",
) -> list:
    """Generate profiles for a batch of queries. memories must match len(queries) if provided."""
    if memories is None:
        memories = [""] * len(queries)

    profiles = []
    for q, m in zip(queries, memories):
        try:
            profile = generate_item_profile(q, m, model=model, memory_setting=memory_setting)
        except Exception as e:
            print(f"  Profile generation error: {e}")
            profile = q  # fallback to raw query
        profiles.append(profile)
    return profiles
