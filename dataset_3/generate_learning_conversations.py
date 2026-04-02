import csv
import warnings
# Suppress Pydantic V1 warnings for Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1 functionality.*")
import json
import os
import argparse
from typing import List, Dict, Optional, Any
import dotenv
import datetime
import random
import time
import re

dotenv.load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# Pydantic models for structured output
class ConversationTurn(BaseModel):
    speaker: str = Field(description="Either 'AI' or 'User'")
    text: str = Field(description="The text spoken by the speaker")

class DailyConversation(BaseModel):
    time_of_day: str = Field(description="E.g., Morning, Afternoon, Evening, Late Night")
    turns: List[ConversationTurn] = Field(description="The turns in this specific interaction")

class ConversationThread:
    """Tracks a topic over multiple interactions or days."""
    def __init__(self, topic: str, expiry_days: int = 3):
        self.topic = topic
        self.created_at = None # Will be set
        self.expiry_days = expiry_days
        self.resolved = False
        self.mentions = 0

class PreferenceState:
    """Simulates the AI's internal belief about a preference per time of day."""
    def __init__(self):
        # time_of_day -> belief string
        self.beliefs: Dict[str, Optional[str]] = {}
        # time_of_day -> days_consistent count
        self.days_consistent: Dict[str, int] = {}

    def get_status(self, time_of_day: str) -> str:
        belief = self.beliefs.get(time_of_day)
        consistent = self.days_consistent.get(time_of_day, 0)
        
        if belief is None:
            return "UNKNOWN: You have no idea what the user likes. You must ask."
        if consistent < 3:
            return f"TENTATIVE: You think they like {belief} (seen for {consistent} days), but you should verify or ask."
        return f"CONFIDENT: You are sure they like {belief}. Suggest it proactively."

def parse_csv(csv_filepath: str):
    """Parses tasks2.csv for preferences."""
    preferences = {}
    
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_str = row['Task']
            # Match "Name Preference (Option1->Option2->...)"
            match = re.match(r"(.*) Preference \((.*)\)", task_str)
            if match:
                name = match.group(1).strip()
                options = [opt.strip() for opt in match.group(2).split('->')]
                frequency = row['Frequency']
                
                # Determine cycle days from frequency string
                cycle_days = 7 # Default weekly
                if "Bi-weekly" in frequency:
                    cycle_days = 14
                elif "Daily" in frequency:
                    cycle_days = 1
                elif "Monthly" in frequency:
                    cycle_days = 30
                    
                preferences[name] = {
                    "options": options,
                    "cycle_days": cycle_days
                }
                
    return preferences

def save_data(data: Dict, filepath: str):
    """Safely saves data to the JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def generate_interaction(llm: ChatGoogleGenerativeAI, context: Dict, max_retries: int = 4) -> dict:
    system_prompt = (
        "You are an AI assistant interacting with a User. You are learning their preferences over time.\n"
        "Time of day: {time_of_day}\n"
        "ACTIVE THREADS: {active_threads}\n"
        "GUIDELINES:\n"
        "{custom_guidelines}\n"
        "CRITICAL: The AI speaker MUST NOT know the User's actual preferences unless stated as KNOWN or CONFIDENT in the guidelines. If a preference is UNKNOWN or TENTATIVE, the AI MUST ask an open question without guessing or verify it. The User speaker MUST then reply with their actual preference as stated in the context. The User speaker SHOULD be proactive in mentioning their preferences if the AI invites them to share or asks an open question.\n"
        "Generate a natural, short interaction (2-6 turns). The conversation MUST conclude logically. It MUST NOT end with the AI asking a question that the User does not answer. The User MUST state their actual preferences during the interaction for any items mentioned or asked about.\n"
        "Use structured output."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Date: {date} ({day_name})\nContext: {human_context}")
    ])
    
    try:
        chain = prompt | llm.with_structured_output(DailyConversation)
        result = chain.invoke(context)
        # Add a small delay after a successful call to avoid immediate 429 on next call
        time.sleep(2)
        return result.model_dump()
    except Exception as e:
        if "429" in str(e):
            print(f"    Rate limit hit. Breaking as requested. Exact error: {str(e)}")
            # We raise the exception so the caller can handle the stop efficiently
            raise e
        else:
            raise e

def main():
    parser = argparse.ArgumentParser(description="Generate learning conversations")
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate")
    parser.add_argument("--output", type=str, default="learning_conversations.json", help="Output file")
    args = parser.parse_args()

    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY not set.")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)
    
    # Path handling for tasks2.csv or tasks.csv
    search_paths = [
        "tasks.csv",
    ]
    csv_path = None
    for p in search_paths:
        full_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), p)
        if os.path.exists(full_p):
            csv_path = full_p
            break
        if os.path.exists(p):
            csv_path = p
            break
            
    if not csv_path:
        print("Error: Could not find tasks2.csv or tasks.csv.")
        return
        
    print(f"Using configuration from: {csv_path}")
    prefs_config = parse_csv(csv_path)
    
    # Change start date to March 1st
    base_date = datetime.date(2026, 3, 1)
    all_data = {}
    
    # Load existing data
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r') as f:
                all_data = json.load(f)
            print(f"Loaded {len(all_data)} days of existing data.")
        except Exception as e:
            print(f"Warning: Could not load existing data: {e}")

    # Global AI Knowledge
    preference_knowledge: Dict[str, PreferenceState] = {name: PreferenceState() for name in prefs_config.keys()}
    
    # Context and Threads
    active_threads: List[ConversationThread] = []

    # Reconstruct state from existing encounters
    sorted_dates = sorted(all_data.keys())
    for date_str in sorted_dates:
        day_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        day_idx = (day_date - base_date).days
        day_data = all_data[date_str]
        interactions = day_data.get("interactions", [])
        
        # Re-simulate Preference Learning
        # Determine all relevant preferences for this day from the start_date
        # We categorise them into parts of the day
        day_prefs = {
            "Morning": ["Coffee", "Breakfast", "T-Shirt Color", "Shoes", "Watch", "Workout Style"],
            "Afternoon": ["Lunch"],
            "Evening": ["Dinner"]
        }
        
        for time_of_day, pref_names in day_prefs.items():
            for pref_name in pref_names:
                if pref_name not in prefs_config: continue
                p_cfg = prefs_config[pref_name]
                
                # Check if preference has started based on Date
                try:
                    start_date = datetime.datetime.strptime(p_cfg['start_date'], "%Y-%m-%d").date()
                except (ValueError, KeyError):
                    start_date = base_date
                    
                if day_date < start_date: continue
                
                actual_val = p_cfg['options'][(day_idx // p_cfg['cycle_days']) % len(p_cfg['options'])]
                
                interactions_count = len(interactions)
                anchor_idx = {"Morning": 1, "Afternoon": 2, "Evening": 3}
                
                if interactions_count >= anchor_idx[time_of_day]:
                    interaction_text = " ".join([turn.get("text", "") for turn in interactions[anchor_idx[time_of_day]-1].get("turns", [])]).lower()
                    p_state = preference_knowledge[pref_name]
                    
                    # Check if the preference value was actually mentioned in the text
                    if actual_val.lower() in interaction_text:
                        current_belief = p_state.beliefs.get(time_of_day)
                        if current_belief != actual_val:
                            p_state.beliefs[time_of_day] = actual_val
                            p_state.days_consistent[time_of_day] = 1
                        else:
                            p_state.days_consistent[time_of_day] += 1



    # Determine where to start
    if sorted_dates:
        last_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date()
        last_interactions = all_data[sorted_dates[-1]].get("interactions", [])
        
        if len(last_interactions) < 3: # Not all 3 planned interactions done
            start_day_idx = (last_date - base_date).days
            print(f"Resuming partial day: {sorted_dates[-1]} ({len(last_interactions)}/3 interactions done)")
        else:
            start_day_idx = (last_date - base_date).days + 1
            print(f"Resuming from next day: {base_date + datetime.timedelta(days=start_day_idx)}")
    else:
        start_day_idx = 0
        print(f"Starting fresh from {base_date}")

    for i in range(start_day_idx, start_day_idx + args.days):
        current_date = base_date + datetime.timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        day_name = current_date.strftime("%A")
        
        if date_str not in all_data:
            all_data[date_str] = {"day": day_name, "interactions": []}
            
        print(f"--- Day {i+1}: {date_str} ({day_name}) ---")
        day_ref = all_data[date_str]
        interactions_done = len(day_ref["interactions"])

        # Randomize interactions for the day
        if interactions_done == 0:
            num_interactions = random.randint(2, 4)
            # Probability-based times
            available_times = ["Morning", "Afternoon", "Evening", "Late Night"]
            interaction_times = sorted(random.sample(available_times, num_interactions), 
                                     key=lambda x: available_times.index(x))
            day_ref["planned_times"] = interaction_times
        else:
            interaction_times = day_ref.get("planned_times", ["Morning", "Afternoon", "Evening"])

        try:
            for time_idx, time_of_day in enumerate(interaction_times):
                if interactions_done > time_idx:
                    continue
                

                # Active threads for prompt
                thread_texts = [f"- {t.topic}" for t in active_threads if not t.resolved]
                active_threads_str = "\n".join(thread_texts) if thread_texts else "None"

                # Preferences logic
                active_prefs = []
                pref_time_map = {
                    "Morning": ["Coffee", "Breakfast", "T-Shirt Color", "Shoes", "Watch", "Workout Style"],
                    "Afternoon": ["Lunch"],
                    "Evening": ["Dinner"],
                    "Late Night": []
                }
                
                potential_prefs = pref_time_map.get(time_of_day, [])
                for pref_name in potential_prefs:
                    if pref_name not in prefs_config: continue
                    p_cfg = prefs_config[pref_name]
                    
                    try:
                        start_date = datetime.datetime.strptime(p_cfg['start_date'], "%Y-%m-%d").date()
                    except (ValueError, KeyError):
                        start_date = base_date
                        
                    if current_date < start_date: continue
                    
                    actual_val = p_cfg['options'][(i // p_cfg['cycle_days']) % len(p_cfg['options'])]
                    active_prefs.append((pref_name, actual_val))

                custom_guidelines = ""
                human_context = ""
                
                for pref_name, actual_val in active_prefs:
                    p_state = preference_knowledge[pref_name]
                    custom_guidelines += f"- Preference '{pref_name}' Knowledge: {p_state.get_status(time_of_day)}\n"
                    human_context += f"- The User's actual {pref_name} preference today is {actual_val}.\n"
                
                if custom_guidelines:
                    custom_guidelines = "Guidelines for Learning:\n" + custom_guidelines
                else:
                    custom_guidelines = "Focus on natural interaction exploring the preferences."

                pref_context = {
                    "time_of_day": time_of_day,
                    "active_threads": active_threads_str,
                    "custom_guidelines": custom_guidelines,
                    "date": date_str,
                    "day_name": day_name,
                    "human_context": human_context
                }
                
                try:
                    interaction = generate_interaction(llm, pref_context)
                    
                    day_ref["interactions"].append(interaction)
                    save_data(all_data, args.output)
                    
                    # AI Knowledge Updates
                    interaction_text = " ".join([turn["text"] for turn in interaction.get("turns", [])]).lower()
                    
                    # Check for new threads in User text (simplified logic)
                    if "plan" in interaction_text or "tomorrow" in interaction_text:
                        # Extract some hint of a thread
                        match = re.search(r"(planning to .*\.|going to .*\.)", interaction_text)
                        if match:
                            new_topic = match.group(1)
                            if not any(t.topic == new_topic for t in active_threads):
                                active_threads.append(ConversationThread(new_topic))
                                print(f"    New Thread: {new_topic}")

                    for pref_name, actual_val in active_prefs:
                        p_state = preference_knowledge[pref_name]
                        if actual_val.lower() in interaction_text:
                            current_belief = p_state.beliefs.get(time_of_day)
                            if current_belief != actual_val:
                                p_state.beliefs[time_of_day] = actual_val
                                p_state.days_consistent[time_of_day] = 1
                            else:
                                p_state.days_consistent[time_of_day] += 1
                except Exception as e:
                    if "429" in str(e): raise e
                    print(f"    Error in interaction for {time_of_day}: {e}")
                
                interactions_done = len(day_ref["interactions"])
        except Exception as e:
            if "429" in str(e):
                print(f"Stopping execution due to rate limit at {date_str}")
                break
            else:
                print(f"Error on {date_str}: {e}")


    print(f"\nSimulation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
