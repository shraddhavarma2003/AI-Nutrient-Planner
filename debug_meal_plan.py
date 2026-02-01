
import sys
import os
import json
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from auth.database import init_database, UserRepository, MedicalProfileRepository
from services.meal_planner import WeeklyPlanGenerator

def debug_meal_plan():
    print("Initializing database...")
    init_database()
    
    # 1. Find a user with a profile
    user_id = None
    with open("data/nutrition.db", "rb") as f:
        # Just checking if file exists, the init_database should have handled it
        pass
        
    print("Checking for users with profiles...")
    # Inspect DB manually to find a user
    import sqlite3
    conn = sqlite3.connect("data/nutrition.db")
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM medical_profiles LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    
    if row:
        user_id = row[0]
        print(f"Found user with profile: {user_id}")
    else:
        print("No user with profile found. creating 'debug_user'...")
        user_id = "debug_user"
        UserRepository.create(user_id, "debug@example.com", "hash", "Debug User")
        MedicalProfileRepository.create(
            user_id, 
            user_id, 
            conditions=["None"], 
            allergens=["None"], 
            daily_targets={"calories": 2000}
        )
        print("Created debug user and profile.")

    # 2. Try to generate plan
    print(f"\nAttempting to generate meal plan for {user_id}...")
    generator = WeeklyPlanGenerator()
    
    try:
        plan = generator.generate_plan(user_id)
        print("\nSUCCESS: Meal plan generated!")
        print(json.dumps(plan, indent=2)[:500] + "...")
    except Exception as e:
        print(f"\nFAILURE: {e}")
        traceback.print_exc()
        
        # We can't easily access the raw response here because the exception doesn't carry it
        # unless we modify the code. But the traceback will help.

if __name__ == "__main__":
    debug_meal_plan()
