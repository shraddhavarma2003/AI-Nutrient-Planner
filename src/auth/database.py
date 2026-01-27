"""
Database Setup for Authentication

SQLite database for users and medical profiles.
Simple, file-based, no external dependencies.
"""

import sqlite3
import os
import uuid
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


# Database file location
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nutrition.db")


def init_database():
    """Initialize database with required tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Medical profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medical_profiles (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conditions TEXT,
                allergens TEXT,
                medications TEXT,
                daily_targets TEXT,
                raw_ocr_text TEXT,
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Upload history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Meal logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meal_logs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                food_name TEXT,
                nutrition TEXT,  -- JSON
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                confidence REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Daily logs table for performance and persistence
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_logs (
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                calories_consumed REAL DEFAULT 0,
                calories_burned REAL DEFAULT 0,
                calories_target REAL DEFAULT 2000,
                protein_g REAL DEFAULT 0,
                carbs_g REAL DEFAULT 0,
                fat_g REAL DEFAULT 0,
                water_cups INTEGER DEFAULT 0,
                water_target INTEGER DEFAULT 8,
                PRIMARY KEY (user_id, date),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Migrations: Add new columns if they don't exist
        new_columns = [
            ("age", "INTEGER"),
            ("gender", "TEXT"),
            ("weight_kg", "REAL"),
            ("height_cm", "REAL"),
            ("activity_level", "TEXT"),
            ("fitness_goal", "TEXT")
        ]
        
        for col_name, col_type in new_columns:
            try:
                cursor.execute(f"ALTER TABLE medical_profiles ADD COLUMN {col_name} {col_type}")
                print(f"[DB] Added column {col_name} to medical_profiles")
            except sqlite3.OperationalError:
                # Column likely already exists
                pass

        conn.commit()


@contextmanager
def get_connection():
    """Get database connection with context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class UserRepository:
    """Repository for user operations."""
    
    @staticmethod
    def create(user_id: str, email: str, password_hash: str, name: str) -> bool:
        """Create a new user."""
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (id, email, password_hash, name) VALUES (?, ?, ?, ?)",
                    (user_id, email.lower(), password_hash, name)
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False  # Email already exists
    
    @staticmethod
    def get_by_email(email: str) -> Optional[Dict[str, Any]]:
        """Get user by email."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    @staticmethod
    def get_by_id(user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None


class MedicalProfileRepository:
    """Repository for medical profile operations."""
    
    @staticmethod
    def create(
        profile_id: str,
        user_id: str,
        conditions: List[str],
        allergens: List[str],
        medications: List[str] = None,
        daily_targets: Dict[str, float] = None,
        raw_ocr_text: str = None,
        source_file: str = None,
        **kwargs
    ) -> bool:
        """Create a medical profile."""
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO medical_profiles 
                    (id, user_id, conditions, allergens, medications, daily_targets, raw_ocr_text, source_file, age, gender, weight_kg, height_cm, activity_level, fitness_goal)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile_id,
                    user_id,
                    json.dumps(conditions),
                    json.dumps(allergens),
                    json.dumps(medications or []),
                    json.dumps(daily_targets or {}),
                    raw_ocr_text,
                    source_file,
                    kwargs.get('age'),
                    kwargs.get('gender'),
                    kwargs.get('weight_kg'),
                    kwargs.get('height_cm'),
                    kwargs.get('activity_level'),
                    kwargs.get('fitness_goal'),
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error creating profile: {e}")
            return False
    
    @staticmethod
    def get_by_user_id(user_id: str) -> Optional[Dict[str, Any]]:
        """Get medical profile for a user."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM medical_profiles WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                profile = dict(row)
                # Parse JSON fields
                profile['conditions'] = json.loads(profile['conditions'] or '[]')
                profile['allergens'] = json.loads(profile['allergens'] or '[]')
                profile['medications'] = json.loads(profile['medications'] or '[]')
                profile['daily_targets'] = json.loads(profile['daily_targets'] or '{}')
                return profile
            return None
    
    @staticmethod
    def update(
        user_id: str,
        conditions: List[str] = None,
        allergens: List[str] = None,
        medications: List[str] = None,
        **kwargs
    ) -> bool:
        """Update an existing profile."""
        with get_connection() as conn:
            cursor = conn.cursor()
            updates = []
            values = []
            
            if conditions is not None:
                updates.append("conditions = ?")
                values.append(json.dumps(conditions))
            if allergens is not None:
                updates.append("allergens = ?")
                values.append(json.dumps(allergens))
            if medications is not None:
                updates.append("medications = ?")
                values.append(json.dumps(medications))
            
            # New fields
            for field in ['age', 'gender', 'weight_kg', 'height_cm', 'activity_level', 'fitness_goal']:
                if field in kwargs:
                    updates.append(f"{field} = ?")
                    values.append(kwargs[field])
            
            if updates:
                updates.append("updated_at = ?")
                values.append(datetime.now().isoformat())
                values.append(user_id)
                
                cursor.execute(
                    f"UPDATE medical_profiles SET {', '.join(updates)} WHERE user_id = ?",
                    values
                )
                conn.commit()
                return cursor.rowcount > 0
            return False


class UploadRepository:
    """Repository for upload tracking."""
    
    @staticmethod
    def create(upload_id: str, user_id: str, filename: str, file_type: str) -> bool:
        """Create upload record."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO uploads (id, user_id, filename, file_type) VALUES (?, ?, ?, ?)",
                (upload_id, user_id, filename, file_type)
            )
            conn.commit()
            return True
    
    @staticmethod
    def update_status(upload_id: str, status: str, error_message: str = None):
        """Update upload status."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE uploads SET status = ?, error_message = ? WHERE id = ?",
                (status, error_message, upload_id)
            )
            conn.commit()


class MealRepository:
    """Repository for meal log operations."""
    
    @staticmethod
    def _log_debug(msg: str):
        try:
            with open("db_ops.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} - {msg}\n")
        except: pass

    @staticmethod
    def create(
        user_id: str,
        food_name: str,
        nutrition: Dict[str, float],
        source: str = "manual",
        confidence: float = 1.0,
        timestamp: datetime = None
    ) -> bool:
        """Log a new meal."""
        try:
            MealRepository._log_debug(f"CREATE START: {food_name} for {user_id}")
            with get_connection() as conn:
                cursor = conn.cursor()
                log_id = str(uuid.uuid4())
                ts = timestamp or datetime.now()
                
                print(f"[DB DEBUG] Creating meal: {food_name} at {ts.isoformat()} for {user_id}")
                
                cursor.execute("""
                    INSERT INTO meal_logs (id, user_id, food_name, nutrition, source, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_id,
                    user_id,
                    food_name,
                    json.dumps(nutrition),
                    source,
                    confidence,
                    ts.isoformat()
                ))
                conn.commit()
                MealRepository._log_debug("CREATE SUCCESS")
                print("[DB DEBUG] Meal saved successfully")
                
                # Update daily log summary
                try:
                    date_str = ts.strftime("%Y-%m-%d")
                    DailyLogRepository.update_nutrition(user_id, date_str, nutrition)
                except Exception as de:
                    print(f"[DB] Error updating daily summary: {de}")
                    
                return True
        except Exception as e:
            MealRepository._log_debug(f"CREATE ERROR: {e}")
            print(f"[DB] Error logging meal: {e}")
            return False

    @staticmethod
    def get_meals_by_date(user_id: str, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get meals logged on a specific date (defaults to today)."""
        target_date_prefix = date_str or datetime.now().strftime("%Y-%m-%d")
        print(f"[DB DEBUG] Fetching meals for {user_id}. Target prefix: {target_date_prefix}")
        MealRepository._log_debug(f"GET START: {user_id} prefix={target_date_prefix}")
        
        with get_connection() as conn:
            cursor = conn.cursor()
            # Just get recent meals
            cursor.execute("""
                SELECT * FROM meal_logs 
                WHERE user_id = ? 
                ORDER BY timestamp DESC
                LIMIT 100
            """, (user_id,))
            
            rows = cursor.fetchall()
            print(f"[DB DEBUG] Found {len(rows)} raw candidate rows")
            MealRepository._log_debug(f"GET RAW: Found {len(rows)} rows")
            
            results = []
            for row in rows:
                # Check if timestamp (string) starts with selected date
                ts = row['timestamp'] # e.g. "2024-05-21T12:00:00.000"
                
                if ts and ts.startswith(target_date_prefix):
                    item = dict(row)
                    # Safe JSON parse
                    try:
                        item['nutrition'] = json.loads(item['nutrition']) if item['nutrition'] else {}
                    except:
                        item['nutrition'] = {}
                    results.append(item)
            
            print(f"[DB DEBUG] Returning {len(results)} valid matches for {target_date_prefix}")
            MealRepository._log_debug(f"GET DONE: Returning {len(results)} matches")
            return results

    @staticmethod
    def clear_meals(user_id: str):
        """Clear meals for a user."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM meal_logs WHERE user_id = ?", (user_id,))
            conn.commit()


class DailyLogRepository:
    """Repository for daily log operations."""

    @staticmethod
    def get_or_create(user_id: str, date_str: str) -> Dict[str, Any]:
        """Get or create a daily log for a specific date."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM daily_logs WHERE user_id = ? AND date = ?",
                (user_id, date_str)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            
            # Create if not exists
            # Try to get target from profile
            profile = MedicalProfileRepository.get_by_user_id(user_id)
            target_cals = 2000
            if profile and profile.get('daily_targets'):
                target_cals = profile['daily_targets'].get('calories', 2000)

            cursor.execute("""
                INSERT INTO daily_logs (user_id, date, calories_target)
                VALUES (?, ?, ?)
            """, (user_id, date_str, target_cals))
            conn.commit()
            
            return {
                "user_id": user_id,
                "date": date_str,
                "calories_consumed": 0,
                "calories_burned": 0,
                "calories_target": target_cals,
                "protein_g": 0,
                "carbs_g": 0,
                "fat_g": 0,
                "water_cups": 0,
                "water_target": 8
            }

    @staticmethod
    def update_nutrition(user_id: str, date_str: str, nutrition: Dict[str, float]):
        """Increment nutrition values for a day."""
        with get_connection() as conn:
            cursor = conn.cursor()
            # Ensure it exists
            DailyLogRepository.get_or_create(user_id, date_str)
            
            cursor.execute("""
                UPDATE daily_logs SET
                    calories_consumed = calories_consumed + ?,
                    protein_g = protein_g + ?,
                    carbs_g = carbs_g + ?,
                    fat_g = fat_g + ?
                WHERE user_id = ? AND date = ?
            """, (
                nutrition.get('calories', 0),
                nutrition.get('protein_g', 0),
                nutrition.get('carbs_g', 0),
                nutrition.get('fat_g', 0),
                user_id,
                date_str
            ))
            conn.commit()

    @staticmethod
    def update_water(user_id: str, date_str: str, delta: int):
        """Update water cups for a day."""
        with get_connection() as conn:
            cursor = conn.cursor()
            DailyLogRepository.get_or_create(user_id, date_str)
            cursor.execute("""
                UPDATE daily_logs SET
                    water_cups = MAX(0, water_cups + ?)
                WHERE user_id = ? AND date = ?
            """, (delta, user_id, date_str))
            conn.commit()

    @staticmethod
    def log_exercise(user_id: str, date_str: str, calories: float):
        """Record burned calories."""
        with get_connection() as conn:
            cursor = conn.cursor()
            DailyLogRepository.get_or_create(user_id, date_str)
            cursor.execute("""
                UPDATE daily_logs SET
                    calories_burned = calories_burned + ?
                WHERE user_id = ? AND date = ?
            """, (calories, user_id, date_str))
            conn.commit()


# Initialize database on import
init_database()
