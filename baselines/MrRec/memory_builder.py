"""
Memory building for MR.Rec baselines.

Implements three memory settings from the paper:
  - w/o Memory: no user history
  - w/ Naive Memory: raw recent interaction history
  - w/ Static Memory: LLM-generated user summary
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def format_naive_memory(user_history: list, max_interactions: int = 10) -> str:
    """Format raw interaction history as naive memory string (recent N interactions)."""
    if not user_history:
        return ""
    recent = user_history[-max_interactions:]
    lines = []
    for h in recent:
        cat = h.get("category", "Unknown")
        rating = h.get("rating", "?")
        review = h.get("review", "")[:300]
        lines.append(f"[{cat}] Rating: {rating}/5 - {review}")
    return "\n".join(lines)


def generate_static_memory(user_history: list, model: str = "gpt-4o-mini") -> str:
    """Generate a static user summary from interaction history using LLM."""
    if not user_history:
        return ""

    history_text = format_naive_memory(user_history, max_interactions=20)
    prompt = f"""Based on the following user interaction history (product reviews and ratings), generate a concise summary of the user's preferences and characteristics. Focus on:
- Product categories they like
- Price sensitivity
- Quality vs. value preferences
- Specific features they care about
- Any notable patterns

User Interaction History:
{history_text}

Generate a concise user preference summary (3-5 sentences):"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def build_local_memory_components(user_history: list, category: str, model: str = "gpt-4o-mini") -> dict:
    """
    Build user-specific local memory components (MR.Rec Section 3.1.1).

    Returns dict with:
      - behavior_records: raw interaction data
      - preference_patterns: category-specific preference summaries
      - user_profile: high-level cross-category profile
    """
    if not user_history:
        return {
            "behavior_records": [],
            "preference_patterns": "",
            "user_profile": "",
        }

    # Behavior Records: raw interaction data
    behavior_records = [
        {
            "item_id": h["item_id"],
            "category": h.get("category", "Unknown"),
            "rating": h.get("rating", 0),
            "review": h.get("review", "")[:500],
        }
        for h in user_history
    ]

    # Preference Patterns: category-specific summaries
    cat_history = [h for h in user_history if h.get("category") == category]
    if cat_history:
        cat_text = "\n".join(
            f"- Rating {h['rating']}/5: {h['review'][:300]}"
            for h in cat_history[-5:]
        )
        pref_prompt = f"""Analyze these user reviews in the {category} category and extract their key preference dimensions (style, quality, price, features, etc.):

{cat_text}

Summarize the user's specific preferences in this category (2-3 sentences):"""
        pref_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": pref_prompt}],
            temperature=0,
            max_tokens=200,
        )
        preference_patterns = pref_resp.choices[0].message.content.strip()
    else:
        preference_patterns = ""

    # User Profile: high-level cross-category summary
    all_text = "\n".join(
        f"[{h.get('category', '?')}] Rating {h['rating']}/5: {h['review'][:200]}"
        for h in user_history[-10:]
    )
    profile_prompt = f"""Based on these product interactions across categories, create a high-level user profile:

{all_text}

Summarize the user's overall shopping preferences and characteristics (2-3 sentences):"""
    profile_resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": profile_prompt}],
        temperature=0,
        max_tokens=200,
    )
    user_profile = profile_resp.choices[0].message.content.strip()

    return {
        "behavior_records": behavior_records,
        "preference_patterns": preference_patterns,
        "user_profile": user_profile,
    }


def format_local_memory_for_prompt(local_memory: dict) -> str:
    """Format local memory components into a prompt-ready string."""
    parts = []
    if local_memory.get("user_profile"):
        parts.append(f"User Profile: {local_memory['user_profile']}")
    if local_memory.get("preference_patterns"):
        parts.append(f"Category Preferences: {local_memory['preference_patterns']}")
    if local_memory.get("behavior_records"):
        recent = local_memory["behavior_records"][-3:]
        recs = "\n".join(
            f"  - [{r['category']}] Rating {r['rating']}/5: {r['review'][:200]}"
            for r in recent
        )
        parts.append(f"Recent Interactions:\n{recs}")
    return "\n\n".join(parts)
