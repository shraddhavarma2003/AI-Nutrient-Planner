"""
RAG (Retrieval-Augmented Generation) Service

Retrieves relevant user data and food information to augment LLM prompts.
This enables personalized AI responses based on:
- User's medical profile (conditions, allergens, medications)
- User's meal history
- Food nutrition database
"""

import os
import csv
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta


class RAGService:
    """
    Retrieval-Augmented Generation service for nutrition AI.
    
    Retrieves and formats context from:
    1. User medical profiles (conditions, allergens, medications)
    2. User meal logs (recent meals)
    3. Food database (healthy_eating_dataset.csv)
    """
    
    def __init__(self):
        """Initialize RAG service with food database."""
        self.food_db = self._load_food_database()
        print(f"[RAG] Loaded {len(self.food_db)} meals from healthy_eating_dataset.csv")
    
    def _load_food_database(self) -> Dict[str, Dict[str, Any]]:
        """Load food database from Indian_Food_Nutrition_Processed.csv."""
        food_db = {}
        
        # Try Indian food nutrition dataset first
        csv_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "data", "Indian_Food_Nutrition_Processed.csv"
        )
        
        # Fallback options
        if not os.path.exists(csv_path):
            csv_path = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "data", "healthy_eating_dataset.csv"
            )
        if not os.path.exists(csv_path):
            csv_path = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "data", "sample_foods.csv"
            )
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Support multiple column name formats
                    dish_name = (
                        row.get('Dish Name') or  # Indian Food dataset
                        row.get('meal_name') or  # healthy_eating_dataset
                        row.get('name')          # sample_foods
                    )
                    if dish_name:
                        food_db[dish_name.lower()] = {
                            'name': dish_name,
                            # Indian food nutrition columns
                            'calories': float(row.get('Calories (kcal)', 0) or row.get('calories', 0) or 0),
                            'protein_g': float(row.get('Protein (g)', 0) or row.get('protein_g', 0) or 0),
                            'carbs_g': float(row.get('Carbohydrates (g)', 0) or row.get('carbs_g', 0) or 0),
                            'fat_g': float(row.get('Fats (g)', 0) or row.get('fat_g', 0) or 0),
                            'fiber_g': float(row.get('Fibre (g)', 0) or row.get('fiber_g', 0) or 0),
                            'sugar_g': float(row.get('Free Sugar (g)', 0) or row.get('sugar_g', 0) or 0),
                            'sodium_mg': float(row.get('Sodium (mg)', 0) or row.get('sodium_mg', 0) or 0),
                            'calcium_mg': float(row.get('Calcium (mg)', 0) or 0),
                            'iron_mg': float(row.get('Iron (mg)', 0) or 0),
                            'vitamin_c_mg': float(row.get('Vitamin C (mg)', 0) or 0),
                            'folate_ug': float(row.get('Folate (µg)', 0) or 0),
                        }
        except Exception as e:
            print(f"[RAG] Warning: Could not load food database: {e}")
        
        return food_db
    
    def get_medical_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user's medical profile from database.
        
        Returns:
            dict with conditions, allergens, medications, or demo profile if none exists
        """
        try:
            from auth.database import MedicalProfileRepository
            profile = MedicalProfileRepository.get_by_user_id(user_id)
            if profile:
                print(f"[RAG] Found user profile: {profile.get('conditions')}")
                return {
                    'conditions': profile.get('conditions', []),
                    'allergens': profile.get('allergens', []),
                    'medications': profile.get('medications', []),
                    'daily_targets': profile.get('daily_targets', {}),
                }
        except Exception as e:
            print(f"[RAG] Error getting medical profile: {e}")
        
        # Return demo profile if no profile found (for testing)
        print(f"[RAG] No profile found for {user_id}, using demo profile")
        return {
            'conditions': ['Diabetes (Type 2)', 'Hypertension'],
            'allergens': ['Peanuts'],
            'medications': ['Metformin'],
            'daily_targets': {},
            '_is_demo': True
        }
    
    def get_meal_history(
        self, 
        user_id: str, 
        meal_logs: Dict[str, List[Dict]], 
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve user's recent meal history.
        
        Args:
            user_id: The user's ID
            meal_logs: In-memory meal logs dict
            days: Number of days to look back
            
        Returns:
            List of recent meals with nutrition info
        """
        if user_id not in meal_logs:
            return []
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_meals = []
        
        for meal in meal_logs.get(user_id, []):
            try:
                timestamp = meal.get('timestamp')
                if isinstance(timestamp, str):
                    meal_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elif isinstance(timestamp, datetime):
                    meal_time = timestamp
                else:
                    meal_time = datetime.now()
                
                if meal_time.replace(tzinfo=None) >= cutoff:
                    recent_meals.append({
                        'food_name': meal.get('food_name', 'Unknown'),
                        'calories': meal.get('nutrition', {}).get('calories', 0),
                        'timestamp': meal_time.strftime('%Y-%m-%d %H:%M'),
                    })
            except Exception:
                # Include meal even if timestamp parsing fails
                recent_meals.append({
                    'food_name': meal.get('food_name', 'Unknown'),
                    'calories': meal.get('nutrition', {}).get('calories', 0),
                    'timestamp': 'Recent',
                })
        
        return recent_meals[-10:]  # Last 10 meals
    
    def search_foods(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search food database by name.
        
        Args:
            query: Food name to search for
            limit: Maximum results to return
            
        Returns:
            List of matching foods with nutrition info
        """
        query_lower = query.lower()
        matches = []
        
        for name, food in self.food_db.items():
            if query_lower in name:
                matches.append(food)
                if len(matches) >= limit:
                    break
        
        return matches
    
    def get_food_info(self, food_name: str) -> Optional[Dict[str, Any]]:
        """Get nutrition info for a specific food."""
        food_lower = food_name.lower()
        
        # Exact match
        if food_lower in self.food_db:
            return self.food_db[food_lower]
        
        # Partial match
        for name, food in self.food_db.items():
            if food_lower in name or name in food_lower:
                return food
        
        return None
    
    def build_context(
        self,
        user_id: str,
        meal_logs: Dict[str, List[Dict]],
        current_food: Optional[Dict[str, Any]] = None,
        user_question: str = ""
    ) -> str:
        """
        Build comprehensive context for LLM prompt.
        
        Args:
            user_id: The user's ID
            meal_logs: In-memory meal logs
            current_food: Currently scanned food data
            user_question: The user's question
            
        Returns:
            Formatted context string for LLM
        """
        context_parts = []
        
        # 1. Medical Profile
        profile = self.get_medical_profile(user_id)
        if profile:
            context_parts.append("=== USER HEALTH PROFILE ===")
            if profile['conditions']:
                context_parts.append(f"Health Conditions: {', '.join(profile['conditions'])}")
            if profile['allergens']:
                context_parts.append(f"Allergens: {', '.join(profile['allergens'])}")
            if profile['medications']:
                context_parts.append(f"Medications: {', '.join(profile['medications'])}")
        else:
            context_parts.append("=== USER HEALTH PROFILE ===")
            context_parts.append("No medical profile on file.")
        
        # 2. Recent Meal History
        meals = self.get_meal_history(user_id, meal_logs)
        if meals:
            context_parts.append("\n=== RECENT MEALS (Last 7 days) ===")
            total_calories = sum(m.get('calories', 0) for m in meals)
            context_parts.append(f"Total meals logged: {len(meals)}")
            context_parts.append(f"Total calories: {total_calories:.0f}")
            context_parts.append("Recent items:")
            for meal in meals[-5:]:  # Last 5 meals
                context_parts.append(f"  - {meal['food_name']} ({meal['calories']:.0f} cal)")
        
        # 3. Current Food Context
        if current_food:
            context_parts.append("\n=== CURRENT FOOD (Just Scanned) ===")
            context_parts.append(f"Food: {current_food.get('name', 'Unknown')}")
            context_parts.append(f"Calories: {current_food.get('calories', 0)}")
            context_parts.append(f"Carbs: {current_food.get('carbs_g', 0)}g")
            context_parts.append(f"Sugar: {current_food.get('sugar_g', 0)}g")
            context_parts.append(f"Protein: {current_food.get('protein_g', 0)}g")
            context_parts.append(f"Fat: {current_food.get('fat_g', 0)}g")
            context_parts.append(f"Sodium: {current_food.get('sodium_mg', 0)}mg")
            
            # Check for relevant foods in database
            db_food = self.get_food_info(current_food.get('name', ''))
            if db_food:
                context_parts.append(f"Glycemic Index: {db_food.get('glycemic_index', 'N/A')}")
                if db_food.get('allergens'):
                    context_parts.append(f"Known Allergens: {db_food['allergens']}")
        
        return "\n".join(context_parts)


# Global singleton
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create the global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
