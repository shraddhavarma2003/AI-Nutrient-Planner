"""
USDA FoodData Central API Integration

Fetches real nutrition data from the official USDA database.
API Documentation: https://fdc.nal.usda.gov/api-guide.html

To use:
1. Get free API key: https://fdc.nal.usda.gov/api-key-signup.html
2. Set environment variable: USDA_API_KEY=your_key
   Or add to .env file
"""

import os
import json
import requests
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

# USDA API Configuration
USDA_API_BASE = "https://api.nal.usda.gov/fdc/v1"
USDA_API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")  # DEMO_KEY has rate limits


@dataclass
class USDANutrition:
    """Nutrition data from USDA."""
    food_name: str
    fdc_id: int
    calories: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    sugar_g: float = 0.0
    fiber_g: float = 0.0
    sodium_mg: float = 0.0
    serving_size: float = 100.0
    serving_unit: str = "g"
    data_type: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "food_name": self.food_name,
            "fdc_id": self.fdc_id,
            "calories": self.calories,
            "protein_g": self.protein_g,
            "carbs_g": self.carbs_g,
            "fat_g": self.fat_g,
            "sugar_g": self.sugar_g,
            "fiber_g": self.fiber_g,
            "sodium_mg": self.sodium_mg,
            "serving_size": self.serving_size,
            "serving_unit": self.serving_unit,
        }


class USDAService:
    """
    Service for fetching nutrition data from USDA FoodData Central.
    
    Nutrient IDs (USDA):
    - 1008: Energy (kcal)
    - 1003: Protein (g)
    - 1005: Carbohydrate (g)
    - 1004: Total Fat (g)
    - 2000: Total Sugars (g)
    - 1079: Fiber (g)
    - 1093: Sodium (mg)
    """
    
    NUTRIENT_IDS = {
        1008: "calories",
        1003: "protein_g",
        1005: "carbs_g", 
        1004: "fat_g",
        2000: "sugar_g",
        1079: "fiber_g",
        1093: "sodium_mg",
    }
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or USDA_API_KEY
        self.session = requests.Session()
        self.session.params = {"api_key": self.api_key}
    
    def search_food(self, query: str, page_size: int = 5) -> List[Dict]:
        """
        Search for foods by name.
        
        Args:
            query: Food name to search
            page_size: Number of results to return
            
        Returns:
            List of food results with fdc_id and description
        """
        url = f"{USDA_API_BASE}/foods/search"
        params = {
            "query": query,
            "pageSize": page_size,
            "dataType": ["Survey (FNDDS)", "Foundation", "SR Legacy"],
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            foods = []
            for food in data.get("foods", []):
                foods.append({
                    "fdc_id": food.get("fdcId"),
                    "description": food.get("description"),
                    "data_type": food.get("dataType"),
                    "brand": food.get("brandOwner", ""),
                })
            return foods
            
        except requests.RequestException as e:
            print(f"[USDA] Search failed: {e}")
            return []
    
    def get_food_nutrients(self, fdc_id: int) -> Optional[USDANutrition]:
        """
        Get nutrition data for a specific food by FDC ID.
        
        Args:
            fdc_id: USDA Food Data Central ID
            
        Returns:
            USDANutrition object with nutrition values
        """
        url = f"{USDA_API_BASE}/food/{fdc_id}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            nutrition = USDANutrition(
                food_name=data.get("description", "Unknown"),
                fdc_id=fdc_id,
                data_type=data.get("dataType", ""),
            )
            
            # Extract nutrients
            for nutrient in data.get("foodNutrients", []):
                nutrient_id = nutrient.get("nutrient", {}).get("id")
                value = nutrient.get("amount", 0)
                
                if nutrient_id in self.NUTRIENT_IDS:
                    attr = self.NUTRIENT_IDS[nutrient_id]
                    setattr(nutrition, attr, float(value) if value else 0.0)
            
            # Get serving size if available
            portions = data.get("foodPortions", [])
            if portions:
                portion = portions[0]
                nutrition.serving_size = portion.get("gramWeight", 100)
                nutrition.serving_unit = portion.get("modifier", "g")
            
            return nutrition
            
        except requests.RequestException as e:
            print(f"[USDA] Get food failed: {e}")
            return None
    
    def get_nutrition_by_name(self, food_name: str) -> Optional[USDANutrition]:
        """
        Search for a food and get its nutrition data.
        
        Args:
            food_name: Name of food to look up
            
        Returns:
            USDANutrition object or None if not found
        """
        # Clean up food name for better search
        search_term = food_name.replace("_", " ").strip()
        
        # Search for the food
        results = self.search_food(search_term, page_size=1)
        
        if not results:
            print(f"[USDA] No results for: {search_term}")
            return None
        
        # Get nutrition for top result
        fdc_id = results[0]["fdc_id"]
        return self.get_food_nutrients(fdc_id)


# Singleton instance
usda_service = USDAService()


def fetch_usda_nutrition(food_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetch nutrition data from USDA for a food name.
    
    This is the main function to call from other modules.
    
    Args:
        food_name: Name of the food
        
    Returns:
        Dict with nutrition data or None
    """
    result = usda_service.get_nutrition_by_name(food_name)
    if result:
        return result.to_dict()
    return None


def bulk_fetch_nutrition(food_names: List[str], output_file: str = None) -> List[Dict]:
    """
    Fetch nutrition for multiple foods.
    
    Args:
        food_names: List of food names
        output_file: Optional path to save results as JSON
        
    Returns:
        List of nutrition data dicts
    """
    results = []
    
    for i, name in enumerate(food_names):
        print(f"[USDA] Fetching {i+1}/{len(food_names)}: {name}")
        
        nutrition = fetch_usda_nutrition(name)
        if nutrition:
            nutrition["original_name"] = name
            results.append(nutrition)
        else:
            # Add placeholder for missing foods
            results.append({
                "original_name": name,
                "food_name": name,
                "error": "Not found in USDA",
            })
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[USDA] Saved results to {output_file}")
    
    return results


# Test function
if __name__ == "__main__":
    # Test with a few foods
    test_foods = ["pizza", "hamburger", "apple pie", "sushi"]
    
    print("Testing USDA API...")
    print(f"API Key: {USDA_API_KEY[:10]}...")
    
    for food in test_foods:
        print(f"\n--- {food} ---")
        result = fetch_usda_nutrition(food)
        if result:
            print(f"  Calories: {result['calories']}")
            print(f"  Protein: {result['protein_g']}g")
            print(f"  Carbs: {result['carbs_g']}g")
            print(f"  Fat: {result['fat_g']}g")
        else:
            print("  Not found")
