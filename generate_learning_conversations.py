import csv
import json
import os
import argparse
from typing import List, Dict, Optional
import dotenv
import datetime
import random
import time

dotenv.load_dotenv()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
except ImportError:
    print("Please install requirements: pip install langchain langchain-google-genai langchain-core pydantic")
    exit(1)

# Pydantic models for structured output
class ConversationTurn(BaseModel):
    speaker: str = Field(description="Either 'AI' or 'User'")
    text: str = Field(description="The text spoken by the speaker")

class DailyConversation(BaseModel):
    time_of_day: str = Field(description="E.g., Morning, Afternoon, Evening")
    turns: List[ConversationTurn] = Field(description="The turns in this specific interaction")

class DailyConversations(BaseModel):
    interactions: List[DailyConversation] = Field(description="Conversations throughout the day")

class LearningState:
    """Simulates the AI's internal belief about a preference."""
    def __init__(self):
        self.belief: Optional[str] = None
        self.days_seen: int = 0
        self.last_corrected_to: Optional[str] = None

    def get_status(self) -> str:
        if self.belief is None:
            return "UNKNOWN: You have no idea what the user likes. You must ask."
        if self.days_seen < 3:
            return f"TENTATIVE: You think they like {self.belief} (seen for {self.days_seen} days), but you should verify or ask if they want the same."
        return f"CONFIDENT: You are sure they like {self.belief}. Suggest it proactively or say you've prepared it."

def parse_preferences(csv_filepath: str) -> Dict:
    """Parses tasks2.csv specifically for the Coffee Preference cycle."""
    # Hardcoded for the specific requested format in tasks2.csv
    # Coffee Preference (Espresso->Latte->Cappuccino->Americano->Macchiato->Flat White)
    coffee_prefs = ["Espresso", "Latte", "Cappuccino", "Americano", "Macchiato", "Flat White"]
    return {
        "Coffee": coffee_prefs,
        "CycleDays": 14 # Bi-weekly as per CSV
    }

def generate_conversations_for_day(llm: ChatGoogleGenerativeAI, day_data: Dict, belief_state: str) -> dict:
    date = day_data['Date']
    day_name = day_data['Day']
    actual_pref = day_data['Actual_Coffee']
    
    system_prompt = (
        "You are an AI assistant interacting with a User. You are learning their habits over time.\n"
        "Your current knowledge status is: {belief_status}\n\n"
        "GUIDELINES:\n"
        "1. If status is UNKNOWN, ask the user what they want.\n"
        "2. If status is TENTATIVE, mention you noticed they had it recently and ask if they want it again.\n"
        "3. If status is CONFIDENT, proactively suggest or provide it.\n"
        "4. IMPORTANT: If your belief (e.g., Espresso) differs from the User's Actual Preference ({actual_pref}), you MUST suggest your belief first. "
        "The User will then CORRECT you. You must apologize and update your mental model in the dialogue.\n\n"
        "Generate 3 short interactions for this day."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Date: {date} ({day_name})\nUser's Actual Preference for Coffee today: {actual_pref}")
    ])
    
    chain = prompt | llm.with_structured_output(DailyConversations)
    result = chain.invoke({
        "belief_status": belief_state,
        "date": date,
        "day_name": day_name,
        "actual_pref": actual_pref
    })
    return result.model_dump()

def main():
    parser = argparse.ArgumentParser(description="Generate learning conversations")
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate")
    parser.add_argument("--output", type=str, default="learning_conversations.json", help="Output file")
    args = parser.parse_args()

    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY not set.")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    
    prefs_config = parse_preferences("tasks2.csv")
    coffee_list = prefs_config["Coffee"]
    cycle = prefs_config["CycleDays"]
    
    start_date = datetime.date(2026, 2, 15)
    all_data = {}
    
    # AI's internal state (simulated memory)
    coffee_state = LearningState()

    for i in range(args.days):
        current_date = start_date + datetime.timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        day_name = current_date.strftime("%A")
        
        # Determine Ground Truth (User's actual current preference)
        # Changes every 'cycle' days
        actual_coffee = coffee_list[(i // cycle) % len(coffee_list)]
        
        # Get AI's current belief status for the prompt
        belief_status_str = coffee_state.get_status()
        
        print(f"--- Day {i+1}: {date_str} ---")
        print(f"AI Belief Status: {belief_status_str}")
        print(f"User Ground Truth: {actual_coffee}")
        
        day_data = {
            "Date": date_str,
            "Day": day_name,
            "Actual_Coffee": actual_coffee
        }
        
        try:
            conversations = generate_conversations_for_day(llm, day_data, belief_status_str)
            
            # Simulated "Learning" after the day is over
            # We look at the conversations to see if the user corrected the AI
            # In this simulation, we'll just check if the AI's belief matched the truth
            
            if coffee_state.belief != actual_coffee:
                # This is either a new start or a shift
                # The AI "learned" the truth today because the user corrected it
                if coffee_state.belief is not None:
                    print(f"AI was CORRECTED from {coffee_state.belief} to {actual_coffee}")
                
                coffee_state.belief = actual_coffee
                coffee_state.days_seen = 1 # Just started learning new pref
            else:
                # AI was right or confirmed it
                coffee_state.days_seen += 1
                
            all_data[date_str] = {
                "day": day_name,
                "ai_belief_at_start": belief_status_str,
                "ground_truth": actual_coffee,
                "interactions": conversations['interactions']
            }
            
            # Save progress
            with open(args.output, 'w') as f:
                json.dump(all_data, f, indent=2)
                
        except Exception as e:
            print(f"Error on day {i+1}: {e}")
            time.sleep(2)

    print(f"\nSimulation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
