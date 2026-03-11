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
    time_of_day: str = Field(description="E.g., Morning, Afternoon, Evening")
    turns: List[ConversationTurn] = Field(description="The turns in this specific interaction")

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

class TaskState:
    """Simulates the AI's internal belief about a task's regularity."""
    def __init__(self):
        self.frequency: Optional[str] = None # e.g., "Weekly on Monday"
        self.confirmed: bool = False

    def get_status(self, task_name: str, day_name: str) -> str:
        if not self.confirmed:
            if self.frequency:
                return f"LEARNING: You suspect this task happens {self.frequency}. Ask the user to confirm the pattern."
            return f"UNKNOWN: You don't know if the user has '{task_name}' today. Ask if they have any tasks."
        return f"KNOWN: You know '{task_name}' happens {self.frequency}. Mention it proactively as if you've already scheduled it or are ready for it."

def parse_csv(csv_filepath: str):
    """Parses tasks2.csv for both preferences and recurrent tasks."""
    preferences = {}
    tasks = []
    
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
            elif row['Frequency'] != "Random":
                tasks.append({
                    "name": task_str,
                    "frequency": row['Frequency'],
                    "start_date": row['Start Date']
                })
                
    return preferences, tasks

def is_task_today(task: Dict, current_date: datetime.date) -> bool:
    """Simple check if a recurrent task occurs today."""
    try:
        start_date = datetime.datetime.strptime(task['start_date'], "%Y-%m-%d").date()
    except ValueError:
        return False
        
    if current_date < start_date:
        return False
        
    delta = (current_date - start_date).days
    freq = task['frequency']
    
    if freq == "Daily":
        return True
    if freq == "Weekly":
        return delta % 7 == 0
    if freq == "Bi-weekly":
        return delta % 14 == 0
    if freq == "Monthly":
        return current_date.day == start_date.day
        
    return False

def save_data(data: Dict, filepath: str):
    """Safely saves data to the JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def generate_interaction(llm: ChatGoogleGenerativeAI, context: Dict, max_retries: int = 4) -> dict:
    system_prompt = (
        "You are an AI assistant interacting with a User. You are learning their habits and schedules over time.\n"
        "Time of day: {time_of_day}\n"
        "GUIDELINES:\n"
        "{custom_guidelines}\n"
        "CRITICAL: The AI speaker MUST NOT know the User's actual preferences or tasks unless stated as KNOWN or CONFIDENT in the guidelines. If a preference/task is UNKNOWN, the AI MUST ask an open question without guessing (e.g. 'What would you like?'). The User speaker MUST then reply with their actual preference.\n"
        "Generate a natural, short interaction (2-6 turns). The conversation MUST conclude logically. It MUST NOT end with the AI asking a question that the User does not answer. The User MUST state their actual preferences during the interaction.\n"
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
    parser.add_argument("--output", type=str, default="dataset_2/learning_conversations.json", help="Output file")
    args = parser.parse_args()

    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY not set.")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)
    
    # Path handling for tasks2.csv or tasks.csv
    search_paths = [
        "dataset_2/tasks2.csv",
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
    prefs_config, tasks_config = parse_csv(csv_path)
    
    # Change start date to March 1st so tasks are active right away
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
    task_knowledge: Dict[str, TaskState] = {t['name']: TaskState() for t in tasks_config}

    # Reconstruct state from existing encounters
    sorted_dates = sorted(all_data.keys())
    for date_str in sorted_dates:
        day_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        day_idx = (day_date - base_date).days
        day_data = all_data[date_str]
        interactions = day_data.get("interactions", [])
        
        # 1. Re-simulate Task Learning
        tasks_today = [t['name'] for t in tasks_config if is_task_today(t, day_date)]
        # We assume if there's any interaction on a day where a task was due, the AI "saw" it
        if interactions:
            for t_name in tasks_today:
                if not task_knowledge[t_name].confirmed:
                    task_knowledge[t_name].frequency = next(t['frequency'] for t in tasks_config if t['name'] == t_name)
                    task_knowledge[t_name].confirmed = True

        # 2. Re-simulate Preference Learning
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
                    p_state = preference_knowledge[pref_name]
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

        try:
            # Generate interactions
            day_prefs = {
                "Morning": ["Coffee", "Breakfast", "T-Shirt Color", "Shoes", "Watch", "Workout Style"],
                "Afternoon": ["Lunch"],
                "Evening": ["Dinner"]
            }
            times = ["Morning", "Afternoon", "Evening"]
            anchor_indices = {"Morning": 0, "Afternoon": 1, "Evening": 2}
            
            for time_of_day in times:
                if interactions_done > anchor_indices[time_of_day]:
                    continue # Already done
                    
                # Collect active preferences for this time block
                active_prefs = []
                for pref_name in day_prefs[time_of_day]:
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
                    custom_guidelines = "Preference Knowledge Status:\n" + custom_guidelines + "\nCRITICAL: If a preference is UNKNOWN, the AI MUST ask an open question and not guess it."
                else:
                    custom_guidelines = "No specific preference knowledge needed for this time."
                    
                # If Morning, also add task logic
                if time_of_day == "Morning":
                    tasks_today = [t['name'] for t in tasks_config if is_task_today(t, current_date)]
                    task_guidelines = [f"- Task '{tn}': {task_knowledge[tn].get_status(tn, day_name)}" for tn in [t['name'] for t in tasks_config]]
                    custom_guidelines += "\n\nTask Knowledge Status:\n" + "\n".join(task_guidelines) + "\n\nIf the user mentions a recurrent task, ask 'Is this every [Day]?' to learn the frequency if unknown."
                    
                    tasks_with_freq = [f"{t} ({next(tx['frequency'] for tx in tasks_config if tx['name'] == t)})" for t in tasks_today]
                    human_context += f"\nThe user actually has these tasks today: {', '.join(tasks_with_freq) if tasks_with_freq else 'None'}."
                
                pref_context = {
                    "time_of_day": time_of_day,
                    "custom_guidelines": custom_guidelines,
                    "date": date_str,
                    "day_name": day_name,
                    "human_context": human_context
                }
                
                try:
                    interaction = generate_interaction(llm, pref_context)
                    day_ref["interactions"].append(interaction)
                    save_data(all_data, args.output)
                    
                    if time_of_day == "Morning":
                        for t_name in tasks_today:
                            if not task_knowledge[t_name].confirmed:
                                task_knowledge[t_name].frequency = next(t['frequency'] for t in tasks_config if t['name'] == t_name)
                                task_knowledge[t_name].confirmed = True
                                print(f"    AI LEARNED task: {t_name}")

                    for pref_name, actual_val in active_prefs:
                        p_state = preference_knowledge[pref_name]
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
