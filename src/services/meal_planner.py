"""
Weekly Meal Planning & Grocery List Service
Uses AI (Ollama/Gemma) to generate personalized meal plans.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from services.llm_service import get_llm_service
from auth.database import WeeklyPlanRepository, MedicalProfileRepository

class WeeklyPlanGenerator:
    """Generates 7-day meal plans using AI based on user health profile."""
    
    def __init__(self):
        self.llm = get_llm_service()

    def generate_plan(self, user_id: str) -> Dict[str, Any]:
        """Generate a personalized 7-day meal plan."""
        # 1. Fetch user profile for constraints
        profile = MedicalProfileRepository.get_by_user_id(user_id)
        if not profile:
            raise ValueError("User profile not found. Please complete medical profile first.")

        # 2. Extract profile constraints
        conditions = profile.get("conditions", [])
        allergens = profile.get("allergens", [])
        targets = profile.get("daily_targets", {})
        calorie_target = targets.get("calories", 2000)

        # 3. Construct AI Prompt
        prompt = self._build_prompt(conditions, allergens, calorie_target)

        # 4. Call LLM
        response = self.llm.chat(prompt, system_prompt="meal_planner")
        
        plan = None
        
        # 5. Try to parse and structure the plan form LLM
        if response.success:
            try:
                plan = self._parse_llm_response(response.content)
            except Exception as e:
                print(f"[MealPlanner] Parsing error for user {user_id}: {e}")
                print(f"[MealPlanner] Raw content: {response.content}")
                # Fallthrough to fallback
        else:
            print(f"[MealPlanner] AI Generation failed: {response.error}")
            # Fallthrough to fallback
            
        # 6. Fallback if AI failed
        if not plan:
            print(f"[MealPlanner] engaging fallback plan generator for {user_id}")
            plan = self._generate_fallback_plan(conditions, calorie_target)

        # Save to DB
        start_date = datetime.now().strftime("%Y-%m-%d")
        WeeklyPlanRepository.create(user_id, start_date, plan)
        
        return plan

    def _generate_fallback_plan(self, conditions: List[str], calorie_target: float) -> Dict[str, Any]:
        """Generate a healthy default plan if AI fails."""
        # Adjust portion sizes based on calorie target roughly
        is_high_cal = calorie_target > 2200
        
        base_plan = {
            "Day 1": {
                "Breakfast": {"name": "Oatmeal with Berries", "ingredients": ["oats", "milk", "berries", "honey"], "calories": 400},
                "Lunch": {"name": "Grilled Chicken Salad", "ingredients": ["chicken breast", "mixed greens", "olive oil", "tomato"], "calories": 600},
                "Dinner": {"name": "Baked Salmon with Quinoa", "ingredients": ["salmon fillet", "quinoa", "asparagus", "lemon"], "calories": 500}
            },
            "Day 2": {
                "Breakfast": {"name": "Greek Yogurt Parfait", "ingredients": ["greek yogurt", "granola", "banana"], "calories": 350},
                "Lunch": {"name": "Turkey Wrap", "ingredients": ["whole wheat tortilla", "turkey slice", "lettuce", "hummus"], "calories": 550},
                "Dinner": {"name": "Vegetable Stir Fry", "ingredients": ["tofu", "broccoli", "bell pepper", "soy sauce", "brown rice"], "calories": 450}
            },
            "Day 3": {
                "Breakfast": {"name": "Scrambled Eggs on Toast", "ingredients": ["eggs", "whole wheat bread", "spinach"], "calories": 400},
                "Lunch": {"name": "Lentil Soup", "ingredients": ["lentils", "carrots", "onion", "celery", "broth"], "calories": 450},
                "Dinner": {"name": "Lean Beef Tacos", "ingredients": ["corn tortilla", "lean ground beef", "salsa", "avocado"], "calories": 600}
            },
            "Day 4": {
                "Breakfast": {"name": "Smoothie Bowl", "ingredients": ["spinach", "banana", "protein powder", "almond milk"], "calories": 350},
                "Lunch": {"name": "Quinoa Salad", "ingredients": ["quinoa", "chickpeas", "cucumber", "feta cheese"], "calories": 500},
                "Dinner": {"name": "Roast Chicken & Veggies", "ingredients": ["chicken thigh", "sweet potato", "brussels sprouts"], "calories": 650}
            },
            "Day 5": {
                "Breakfast": {"name": "Avocado Toast", "ingredients": ["whole grain bread", "avocado", "poached egg"], "calories": 400},
                "Lunch": {"name": "Tuna Salad Sandwich", "ingredients": ["canned tuna", "mayo", "lettuce", "whole wheat bread"], "calories": 500},
                "Dinner": {"name": "Pasta Primavera", "ingredients": ["whole wheat pasta", "zucchini", "tomato sauce", "parmesan"], "calories": 550}
            },
            "Day 6": {
                "Breakfast": {"name": "Cottage Cheese Bowl", "ingredients": ["cottage cheese", "pineapple", "walnuts"], "calories": 350},
                "Lunch": {"name": "Chicken Burrito Bowl", "ingredients": ["grilled chicken", "black beans", "corn", "rice"], "calories": 650},
                "Dinner": {"name": "Grilled Fish Tacos", "ingredients": ["white fish", "cabbage slaw", "lime", "corn tortillas"], "calories": 500}
            },
            "Day 7": {
                "Breakfast": {"name": "Pancakes (Whole Wheat)", "ingredients": ["flour", "milk", "egg", "maple syrup"], "calories": 450},
                "Lunch": {"name": "Mediterranean Salad", "ingredients": ["cucumber", "tomato", "olives", "feta", "olive oil"], "calories": 400},
                "Dinner": {"name": "Roast Beef with Potatoes", "ingredients": ["roast beef", "potatoes", "carrots", "gravy"], "calories": 700}
            }
        }
        
        return {"weekly_plan": base_plan, "fallback_mode": True}

    def _build_prompt(self, conditions: List[str], allergens: List[str], calorie_target: float) -> str:
        return f"""Generate a structured 7-day meal plan for a user with the following profile:
- Health Conditions: {', '.join(conditions) if conditions else 'None'}
- Allergens: {', '.join(allergens) if allergens else 'None'}
- Daily Calorie Target: {calorie_target} kcal

The plan must include Breakfast, Lunch, and Dinner for each day (Day 1 to Day 7).
Each meal should list:
1. Meal name
2. Principal ingredients (list)
3. Estimated calories

FORMAT: YOU MUST RETURN A RAW JSON OBJECT. NO MARKDOWN. NO PREAMBLE.
Schema:
{{
  "weekly_plan": {{
    "Day 1": {{
      "Breakfast": {{ "name": "...", "ingredients": ["...", "..."], "calories": 400 }},
      "Lunch": {{ "name": "...", "ingredients": ["...", "..."], "calories": 600 }},
      "Dinner": {{ "name": "...", "ingredients": ["...", "..."], "calories": 500 }}
    }},
    ... (continue for Day 2 through Day 7)
  }}
}}
"""

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response more robustly."""
        content = content.strip()
        
        # 1. Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 2. Look for JSON blocks ```json ... ```
        import re
        block_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        if block_match:
            try:
                return json.loads(block_match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Last ditch: try to find anything between the first { and last }
        loose_match = re.search(r"(\{.*\})", content, re.DOTALL)
        if loose_match:
            try:
                return json.loads(loose_match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not find a valid JSON object in the AI response.")


class GroceryGenerator:
    """Generates a categorized grocery list from a weekly meal plan."""

    @staticmethod
    def generate_list(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Aggregate ingredients from a meal plan into a categorized list.
        """
        raw_items = []
        weekly_plan = plan.get("weekly_plan", {})
        
        if not isinstance(weekly_plan, dict):
            print(f"[GroceryGen] Error: weekly_plan is not a dict, it is {type(weekly_plan)}")
            return []
            
        for day, meals in weekly_plan.items():
            if not isinstance(meals, dict): continue
            for meal_type, details in meals.items():
                if not isinstance(details, dict): continue
                
                ingredients = details.get("ingredients", [])
                if isinstance(ingredients, list):
                    # Only add non-empty strings, and ensure they are strings
                    for item in ingredients:
                        if item and isinstance(item, str):
                            raw_items.append(item.strip())
                        elif item:
                            raw_items.append(str(item).strip())
                elif isinstance(ingredients, str):
                    # AI returned a single string or comma-separated list
                    if "," in ingredients:
                        parts = [p.strip() for p in ingredients.split(",") if p.strip()]
                        raw_items.extend(parts)
                    else:
                        if ingredients.strip():
                            raw_items.append(ingredients.strip())

        # Clean raw items: Filter out single characters which are usually parsing artifacts
        raw_items = [i for i in raw_items if len(i) > 1]

        # Basic deduplication and simple categorization
        # In a real app, this would use an ingredient database/AI
        # For now, we'll return a simple unique list grouped by basic types
        unique_items = sorted(list(set(raw_items)))
        
        # Categorize (Mock logic)
        categories = {
            "Produce": ["apple", "banana", "spinach", "kale", "onion", "garlic", "tomato", "salad", "vegetables", "fruit"],
            "Protein": ["chicken", "fish", "tofu", "beans", "lentils", "egg", "meat", "beef", "turkey", "paneer"],
            "Dairy": ["milk", "yogurt", "cheese", "butter", "cream"],
            "Pantry": ["rice", "quinoa", "oat", "oil", "spice", "salt", "pepper", "bread", "flour", "pasta", "honey"]
        }

        categorized_list = []
        assigned_items = set()

        for cat_name, keywords in categories.items():
            items = []
            for item in unique_items:
                if any(kw in item.lower() for kw in keywords):
                    items.append(item)
                    assigned_items.add(item)
            if items:
                categorized_list.append({"category": cat_name, "items": items})

        # Add remaining as 'Other'
        others = [i for i in unique_items if i not in assigned_items]
        if others:
            categorized_list.append({"category": "Other", "items": others})

        return categorized_list
