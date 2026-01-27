import pytest
from datetime import datetime, date
from auth.database import DailyLogRepository, MealRepository, init_database
import os
import sqlite3

@pytest.fixture(autouse=True)
def setup_db():
    """Ensure DB is initialized before tests."""
    init_database()
    yield

def test_daily_log_creation():
    user_id = "test_user_persistence"
    today = date.today().strftime("%Y-%m-%d")
    
    # Get or create
    log = DailyLogRepository.get_or_create(user_id, today)
    assert log["user_id"] == user_id
    assert log["date"] == today
    assert log["calories_consumed"] == 0

def test_nutrition_increment():
    user_id = "test_user_nutrition"
    today = date.today().strftime("%Y-%m-%d")
    
    nutrition = {
        "calories": 500,
        "protein_g": 20,
        "carbs_g": 50,
        "fat_g": 10
    }
    
    DailyLogRepository.update_nutrition(user_id, today, nutrition)
    
    log = DailyLogRepository.get_or_create(user_id, today)
    assert log["calories_consumed"] == 500
    assert log["protein_g"] == 20
    
    # Increment again
    DailyLogRepository.update_nutrition(user_id, today, nutrition)
    log = DailyLogRepository.get_or_create(user_id, today)
    assert log["calories_consumed"] == 1000

def test_water_increment():
    user_id = "test_user_water"
    today = date.today().strftime("%Y-%m-%d")
    
    DailyLogRepository.update_water(user_id, today, 1) # Add 1
    log = DailyLogRepository.get_or_create(user_id, today)
    assert log["water_cups"] == 1
    
    DailyLogRepository.update_water(user_id, today, -1) # Remove 1
    log = DailyLogRepository.get_or_create(user_id, today)
    assert log["water_cups"] == 0
    
    DailyLogRepository.update_water(user_id, today, -1) # Below zero test
    log = DailyLogRepository.get_or_create(user_id, today)
    assert log["water_cups"] == 0

def test_exercise_log():
    user_id = "test_user_exercise"
    today = date.today().strftime("%Y-%m-%d")
    
    DailyLogRepository.log_exercise(user_id, today, 300)
    log = DailyLogRepository.get_or_create(user_id, today)
    assert log["calories_burned"] == 300

def test_meal_logging_sync():
    """Test that creating a meal entry automatically updates the daily log."""
    user_id = "test_user_sync"
    today = datetime.now()
    
    nutrition = {
        "calories": 250,
        "protein_g": 10,
        "carbs_g": 30,
        "fat_g": 5
    }
    
    # This should trigger DailyLogRepository.update_nutrition
    MealRepository.create(
        user_id=user_id,
        food_name="Sync Test Food",
        nutrition=nutrition,
        timestamp=today
    )
    
    log = DailyLogRepository.get_or_create(user_id, today.strftime("%Y-%m-%d"))
    assert log["calories_consumed"] == 250
