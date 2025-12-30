"""
Database Setup for Authentication

SQLite database for users and medical profiles.
Simple, file-based, no external dependencies.
"""

import sqlite3
import os
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
    ) -> bool:
        """Create a medical profile."""
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO medical_profiles 
                    (id, user_id, conditions, allergens, medications, daily_targets, raw_ocr_text, source_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile_id,
                    user_id,
                    json.dumps(conditions),
                    json.dumps(allergens),
                    json.dumps(medications or []),
                    json.dumps(daily_targets or {}),
                    raw_ocr_text,
                    source_file,
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


# Initialize database on import
init_database()
