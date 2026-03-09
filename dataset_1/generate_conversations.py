import csv
import json
import os
import argparse
from typing import List, Dict
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

# Define Pydantic models for structured output
class ConversationTurn(BaseModel):
    speaker: str = Field(description="Either 'AI' or 'User'")
    text: str = Field(description="The text spoken by the speaker")

class DailyConversation(BaseModel):
    time_of_day: str = Field(description="E.g., Morning, Afternoon, Evening")
    turns: List[ConversationTurn] = Field(description="The turns in this specific interaction")

class DailyConversations(BaseModel):
    interactions: List[DailyConversation] = Field(description="5 separate conversations throughout the day")

def parse_activities(csv_filepath: str, num_days: int, start_offset: int = 0) -> List[Dict]:
    """Parse the tasks CSV and return a list of dictionaries for each day, starting from an offset."""
    tasks = []
    with open(csv_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)

    start_date = datetime.date(2026, 2, 15)
    days_data = []

    coffee_prefs = ["Espresso", "Latte", "Cappuccino", "Americano", "Macchiato", "Flat White"]
    shirt_prefs = ["Black", "White", "Navy", "Grey", "Olive"]
    breakfast_prefs = ["Oatmeal", "Eggs & Toast", "Pancakes", "Fruit Bowl", "Cereal"]
    shoe_prefs = ["Sneakers", "Boots", "Loafers", "Oxfords", "Running Shoes"]
    lunch_prefs = ["Salad", "Sandwich", "Sushi", "Pasta", "Burrito"]
    dinner_prefs = ["Pizza", "Steak", "Curry", "Stir Fry", "Soup"]
    watch_prefs = ["Smartwatch", "Analog", "Digital", "Chronograph", "Fitness Band"]
    workout_prefs = ["Cardio", "Weights", "Yoga", "Pilates", "HIIT"]
    random.seed(42)

    for i in range(start_offset, start_offset + num_days):
        current_date = start_date + datetime.timedelta(days=i)
        day_name = current_date.strftime("%A")
        
        daily_activities = []
        user_actual_prefs = []
        
        # Helper to process preference with disagreement probability
        def process_pref(name, index, prefs_list, user_change_prob=0.3):
            
            sched = prefs_list[index]
            daily_activities.append(f"{name}: {sched}")
            
            # User changing mind (hidden from AI initially)
            if random.random() < user_change_prob:
                other_options = [p for p in prefs_list if p != sched]
                actual = random.choice(other_options)
                user_actual_prefs.append(f"{name}: {actual} (Changed mind)")
            else:
                user_actual_prefs.append(f"{name}: {sched}")

        process_pref("Coffee Preference", (i // 15) % len(coffee_prefs), coffee_prefs, 0.30)
        process_pref("T-Shirt Preference", (i // 7) % len(shirt_prefs), shirt_prefs, 0.3)
        process_pref("Breakfast Preference", (i // 7) % len(breakfast_prefs), breakfast_prefs, 0.3)
        process_pref("Shoes Preference", (i // 7) % len(shoe_prefs), shoe_prefs, 0.3)
        process_pref("Lunch Preference", i % len(lunch_prefs), lunch_prefs, 0.3)
        process_pref("Dinner Preference", i % len(dinner_prefs), dinner_prefs, 0.3)
        process_pref("Watch Preference", (i // 7) % len(watch_prefs), watch_prefs, 0.3)
        process_pref("Workout Style Preference", (i // 7) % len(workout_prefs), workout_prefs, 0.3)
            
        for task in tasks:
            task_name = task['Task']
            if "Preference" in task_name or "Deep Learning" in task_name:
                continue
                
            try:
                task_start = datetime.datetime.strptime(task['Start Date'], "%Y-%m-%d").date()
                if task['End Date'] != 'Random':
                    task_end = datetime.datetime.strptime(task['End Date'], "%Y-%m-%d").date()
                else:
                    task_end = datetime.date(2030, 1, 1) # Arbitrary far date
            except ValueError:
                continue
                
            if current_date < task_start or current_date > task_end:
                continue
                
            freq = task['Frequency']
            days_since_start = (current_date - task_start).days
            
            if freq == 'Daily':
                daily_activities.append(task_name)
                user_actual_prefs.append(task_name)
            elif freq == 'Weekly' and days_since_start % 7 == 0:
                daily_activities.append(task_name)
                user_actual_prefs.append(task_name)
            elif freq == 'Bi-weekly' and days_since_start % 14 == 0:
                daily_activities.append(task_name)
                user_actual_prefs.append(task_name)
            elif freq == 'Monthly' and current_date.day == task_start.day:
                daily_activities.append(task_name)
                user_actual_prefs.append(task_name)
                
        activity_str = "; ".join(daily_activities) if daily_activities else "nothing"
        actual_str = "; ".join(user_actual_prefs) if user_actual_prefs else "nothing"
        
        days_data.append({
            'Date': current_date.strftime("%Y-%m-%d"),
            'Day': day_name,
            'Activities': activity_str,
            'Actual': actual_str
        })
        
    return days_data

def generate_conversations_for_day(llm: ChatGoogleGenerativeAI, day_data: Dict) -> dict:
    """Uses Langchain to generate conversations for a specific day based on its activities."""
    date = day_data['Date']
    day_name = day_data['Day']
    activities = day_data['Activities']
    user_actual = day_data['Actual']
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an author writing 5 separate short conversational interactions that occur throughout the day between an AI assistant and a User.\n"
         "The AI assistant is proactive and helps the user manage their daily tasks. The AI ONLY knows the 'AI Scheduled Activities' below.\n"
         "However, today the User actually has 'User's Actual Preferences' in mind instead. If the User's actual preference differs from the schedule (it says 'Changed mind'), "
         "the AI must initially suggest the scheduled preference. The User MUST then correct the AI and state their actual preference instead. The AI should politely adapt.\n"
         "If the User has NOT changed their mind, the dialogue proceeds normally. Make the interactions realistic, interesting, and cover all tasks throughout the day."),
        ("human", 
         "Date: {date}\nDay: {day_name}\nAI Scheduled Activities: {activities}\nUser's Actual Preferences: {user_actual}\n\nPlease generate the 3-5 daily conversations.")
    ])
    
    max_retries = 3
    retry_delay = 2 # Initial delay in seconds
    
    # Try with initial requirement
    interaction_count_str = "3-5"
    
    for attempt in range(max_retries + 1):
        try:
            chain = prompt | llm.with_structured_output(DailyConversations)
            
            print(f"Generating {interaction_count_str} conversations for {date} ({day_name}) (Attempt {attempt + 1})")
            print(f"AI Scheduled: {activities}\nUser Actual: {user_actual}")
            
            result = chain.invoke({
                "date": date,
                "day_name": day_name,
                "activities": activities,
                "user_actual": user_actual
            })
            
            # Convert Pydantic object to dict for JSON serialization
            return result.model_dump()
            
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            
            # If we've reached max retries, or if it's a specific token limit error, 
            # we could try to reduce the scope for the last attempt
            if attempt < max_retries:
                # Exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
                # If we've failed twice, try reducing the scope as a fallback
                if attempt == 1:
                    print("Fallback: Reducing requested interactions to 2 to save tokens.")
                    interaction_count_str = "2"
                    # Update prompt for fallback
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", 
                         "You are an author writing exactly 2 separate short conversational interactions that occur throughout the day between an AI assistant and a User.\n"
                         "The AI assistant is proactive and helps the user manage their daily tasks. The AI ONLY knows the 'AI Scheduled Activities' below.\n"
                         "However, today the User actually has 'User's Actual Preferences' in mind instead. If the User's actual preference differs from the schedule (it says 'Changed mind'), "
                         "the AI must initially suggest the scheduled preference. The User MUST then correct the AI and state their actual preference instead. The AI should politely adapt.\n"
                         "If the User has NOT changed their mind, the dialogue proceeds normally. Make the interactions realistic and interesting."),
                        ("human", 
                         "Date: {date}\nDay: {day_name}\nAI Scheduled Activities: {activities}\nUser's Actual Preferences: {user_actual}\n\nPlease generate exactly 2 daily conversations.")
                    ])
            else:
                # Re-raise the last exception if all retries fail
                raise e

def main():
    parser = argparse.ArgumentParser(description="Generate daily conversations using Gemini via Langchain")
    parser.add_argument("--input", type=str, default="tasks.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default="daily_conversations.json", help="Output JSON file")
    parser.add_argument("--days", type=int, default=3, help="Number of days to process (to limit API usage)")
    
    args = parser.parse_args()
    
    # Ensure API key is set
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set your Gemini API key in your terminal before running this script.")
        print("Example: export GOOGLE_API_KEY='AIzaSy...'")
        return
        
    print(f"Initializing Gemini via LangChain...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    # Use absolute paths if relative paths were not provided correctly
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_path)
        
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_path)
    
    print(f"Reading tasks from {input_path}")
    
    # Load existing data if file exists
    all_conversations = {}
    last_date = None
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                all_conversations = json.load(f)
            print(f"Loaded {len(all_conversations)} days of existing conversations from {output_path}")
            
            # Find the latest date
            if all_conversations:
                dates = sorted([datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in all_conversations.keys()])
                last_date = dates[-1]
                print(f"Latest conversation date found: {last_date}")
        except Exception as e:
            print(f"Warning: Could not load existing data: {e}")

    # Calculate start offset
    base_date = datetime.date(2026, 2, 15)
    if last_date:
        start_offset = (last_date - base_date).days + 1
        print(f"Resuming from offset {start_offset} (Next day: {base_date + datetime.timedelta(days=start_offset)})")
    else:
        start_offset = 0
        print(f"Starting fresh from {base_date}")

    days_data = parse_activities(input_path, args.days, start_offset)
    
    for day in days_data:
        # Skip if already generated (optional, based on your preference, but safer for reruns)
        if day['Date'] in all_conversations:
            print(f"Skipping {day['Date']} as it already exists in the output.")
            continue

        try:
            day_convos = generate_conversations_for_day(llm, day)
            all_conversations[day['Date']] = {
                "day": day['Day'],
                "ai_scheduled": day['Activities'],
                "user_actual": day['Actual'],
                "conversations": day_convos['interactions']
            }
            
            # Instant save after each day
            with open(output_path, 'w') as f:
                json.dump(all_conversations, f, indent=2)
            print(f"Successfully saved {day['Date']} to {output_path}")

        except Exception as e:
            print(f"Error generating conversations for {day['Date']}: {e}")
            
    print(f"\nFinalized generated conversations in {output_path}")

if __name__ == "__main__":
    main()
