import datetime

def generate_activities():
    # Set the range for 1 year
    start_date = datetime.date(2026, 2, 15)
    end_date = datetime.date(2027, 2, 15)
    
    dataset = {}
    current = start_date
    while current <= end_date:
        dataset[current] = []
        current += datetime.timedelta(days=1)

    
    # 1. Gym: Every Monday for 2 weeks
    for i in range(2):
        d = datetime.date(2026, 2, 16) + datetime.timedelta(weeks=i)
        if d in dataset:
            dataset[d].append("Gym")
            
    # 2. Meeting: Every second Saturday for 3 months
    for i in range(6): 
        d = datetime.date(2026, 2, 21) + datetime.timedelta(weeks=2*i)
        if d in dataset:
            dataset[d].append("Meeting")
            
    # 3. Project Report: 15th of every month for 4 months
    for i in range(4):
        month = (start_date.month + i - 1) % 12 + 1
        year = start_date.year + (start_date.month + i - 1) // 12
        try:
            d = datetime.date(year, month, 15)
            if d in dataset:
                dataset[d].append("Project report")
        except ValueError:
            pass
            
    # 4. Call Parents: Every Sunday for 6 weeks
    for i in range(6):
        d = datetime.date(2026, 2, 15) + datetime.timedelta(weeks=i)
        if d in dataset:
            dataset[d].append("Call parents")
            
    # 5. Pay Rent: 1st of every month for 3 months
    for i in range(3):
        # Starts from March 1st as requested
        month = (start_date.month + i) % 12 + 1 
        year = start_date.year + (start_date.month + i) // 12
        try:
            d = datetime.date(year, month, 1)
            if d in dataset:
                dataset[d].append("Pay rent")
        except ValueError:
            pass
            
    # 6. Team Sync: Every Wednesday for 5 weeks
    for i in range(5):
        d = datetime.date(2026, 2, 18) + datetime.timedelta(weeks=i)
        if d in dataset:
            dataset[d].append("Team sync")
            
    # 7. Doctor appointment: Every 3rd Friday for 2 months
    doctor_dates = [datetime.date(2026, 2, 20), datetime.date(2026, 3, 20)]
    for d in doctor_dates:
        if d in dataset:
            dataset[d].append("Doctor appointment")
            
    # 8. Grocery shopping: every alternate Thursday for 8 weeks
    for i in range(4):
        d = datetime.date(2026, 2, 19) + datetime.timedelta(weeks=2*i)
        if d in dataset:
            dataset[d].append("Grocery shopping")

    # 9. Library visit: Every Tuesday for 4 months
    for i in range(16):
        d = datetime.date(2026, 2, 17) + datetime.timedelta(weeks=i)
        if d in dataset:
            dataset[d].append("Library visit")

    # 10. Yoga session: Every Friday for 6 months
    for i in range(26):
        d = datetime.date(2026, 2, 20) + datetime.timedelta(weeks=i)
        if d in dataset:
            dataset[d].append("Yoga session")

    # 11. Visit grandparents: Every last Sunday of the month for 1 year
    current_iter = start_date
    while current_iter <= end_date:
        # Find last day of current month
        next_month = current_iter.replace(day=28) + datetime.timedelta(days=4)
        last_day = next_month - datetime.timedelta(days=next_month.day)
        # Find last Sunday
        last_sunday = last_day - datetime.timedelta(days=(last_day.weekday() + 1) % 7)
        if last_sunday in dataset:
            dataset[last_sunday].append("Visit grandparents")
        current_iter = (last_day + datetime.timedelta(days=1))

    # 12. Car wash: Every 1st Saturday of the month for 1 year
    current_iter = start_date
    while current_iter <= end_date:
        first_day = current_iter.replace(day=1)
        # Find 1st Saturday
        first_sat = first_day + datetime.timedelta(days=(5 - first_day.weekday() + 7) % 7)
        if first_sat in dataset:
            dataset[first_sat].append("Car wash")
        # Move to 1st of next month
        next_month = first_day.replace(day=28) + datetime.timedelta(days=4)
        current_iter = next_month.replace(day=1)

    # 13. Coffee Preference: Changes every 15 days, with 30% chance of "nostalgic" preference
    import random
    random.seed(42) # For reproducibility
    coffee_prefs = ["Espresso", "Latte", "Cappuccino", "Americano", "Macchiato", "Flat White"]
    current_dt = start_date
    while current_dt <= end_date:
        # Calculate day difference from start
        days_diff = (current_dt - start_date).days
        coffee_index = (days_diff // 15) % len(coffee_prefs)
        
        # 30% chance to pick the PREVIOUS coffee in the cycle (if available)
        if coffee_index > 0 and random.random() < 0.30:
            chosen_coffee = coffee_prefs[coffee_index - 1]
            dataset[current_dt].insert(0, f"Coffee: {chosen_coffee} (Nostalgic choice)")
        else:
            dataset[current_dt].insert(0, f"Coffee: {coffee_prefs[coffee_index]}")
            
        current_dt += datetime.timedelta(days=1)
            
    # 14. Deep Learning Project: 5 random 3-day sequences per year
    num_sprints = 5
    sprint_starts = []
    
    # Generate 5 random start dates at least 5 days apart
    all_days = list(dataset.keys())
    # Exclude last 3 days to avoid overflow
    possible_starts = all_days[:-3]
    
    random.seed(42) # Keeping same seed for consistency
    while len(sprint_starts) < num_sprints:
        d = random.choice(possible_starts)
        # Check for overlap (ensure no start date is within 4 days of another)
        if not any(abs((d - existing).days) < 5 for existing in sprint_starts):
            sprint_starts.append(d)
            
    for start_d in sorted(sprint_starts):
        dataset[start_d].append("Dataset collection and cleaning")
        dataset[start_d + datetime.timedelta(days=1)].append("Model training and optimization")
        dataset[start_d + datetime.timedelta(days=2)].append("Result analysis and report")
            
    # --- Output Generation ---
    
    filename = "/Users/swanandpande/Documents/Coding/Projects/LifelongAgent/activities_dataset_full.csv"
    with open(filename, "w") as f:
        f.write("Date,Day,Activities\n")
        current_dt = start_date
        while current_dt <= end_date:
            day_name = current_dt.strftime("%A")
            activities = dataset[current_dt]
            activity_str = "; ".join(activities) if activities else "nothing"
            f.write(f"{current_dt},{day_name},{activity_str}\n")
            current_dt += datetime.timedelta(days=1)
    
    print(f"Dataset generated: {filename}")

if __name__ == "__main__":
    generate_activities()
