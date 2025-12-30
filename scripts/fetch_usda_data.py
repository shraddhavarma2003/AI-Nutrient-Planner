"""
Script to fetch USDA nutrition data for all Food-101 classes.

Usage:
    1. Get USDA API key: https://fdc.nal.usda.gov/api-key-signup.html
    2. Set environment variable: set USDA_API_KEY=your_key_here
    3. Run: python scripts/fetch_usda_data.py

This will update data/sample_foods.csv with real USDA values.
"""

import os
import sys
import json
import csv
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.usda_service import USDAService, USDA_API_KEY


# Food-101 classes with better search terms for USDA
FOOD_101_SEARCH_TERMS = {
    "apple_pie": "apple pie",
    "baby_back_ribs": "pork ribs barbecue",
    "baklava": "baklava pastry",
    "beef_carpaccio": "beef raw",
    "beef_tartare": "beef tartare raw",
    "beet_salad": "beet salad",
    "beignets": "beignet fried dough",
    "bibimbap": "bibimbap korean rice",
    "bread_pudding": "bread pudding",
    "breakfast_burrito": "breakfast burrito",
    "bruschetta": "bruschetta tomato",
    "caesar_salad": "caesar salad",
    "cannoli": "cannoli pastry",
    "caprese_salad": "caprese salad mozzarella tomato",
    "carrot_cake": "carrot cake",
    "ceviche": "ceviche fish",
    "cheesecake": "cheesecake",
    "cheese_plate": "cheese cheddar",
    "chicken_curry": "chicken curry",
    "chicken_quesadilla": "chicken quesadilla",
    "chicken_wings": "chicken wings",
    "chocolate_cake": "chocolate cake",
    "chocolate_mousse": "chocolate mousse",
    "churros": "churros fried",
    "clam_chowder": "clam chowder soup",
    "club_sandwich": "club sandwich",
    "crab_cakes": "crab cake",
    "creme_brulee": "creme brulee custard",
    "croque_madame": "croque madame sandwich",
    "cup_cakes": "cupcake",
    "deviled_eggs": "deviled eggs",
    "donuts": "donut glazed",
    "dumplings": "dumpling pork",
    "edamame": "edamame",
    "eggs_benedict": "eggs benedict",
    "escargots": "escargot snail",
    "falafel": "falafel",
    "filet_mignon": "filet mignon beef",
    "fish_and_chips": "fish and chips fried",
    "foie_gras": "foie gras",
    "french_fries": "french fries",
    "french_onion_soup": "french onion soup",
    "french_toast": "french toast",
    "fried_calamari": "fried calamari squid",
    "fried_rice": "fried rice",
    "frozen_yogurt": "frozen yogurt",
    "garlic_bread": "garlic bread",
    "gnocchi": "gnocchi potato",
    "greek_salad": "greek salad feta",
    "grilled_cheese_sandwich": "grilled cheese sandwich",
    "grilled_salmon": "salmon grilled",
    "guacamole": "guacamole avocado",
    "gyoza": "gyoza dumpling",
    "hamburger": "hamburger beef",
    "hot_and_sour_soup": "hot and sour soup",
    "hot_dog": "hot dog frankfurter",
    "huevos_rancheros": "huevos rancheros",
    "hummus": "hummus",
    "ice_cream": "ice cream vanilla",
    "lasagna": "lasagna",
    "lobster_bisque": "lobster bisque soup",
    "lobster_roll_sandwich": "lobster roll sandwich",
    "macaroni_and_cheese": "macaroni and cheese",
    "macarons": "macaron cookie",
    "miso_soup": "miso soup",
    "mussels": "mussels steamed",
    "nachos": "nachos cheese",
    "omelette": "omelette egg",
    "onion_rings": "onion rings fried",
    "oysters": "oysters raw",
    "pad_thai": "pad thai noodles",
    "paella": "paella rice seafood",
    "pancakes": "pancakes",
    "panna_cotta": "panna cotta cream",
    "peking_duck": "peking duck roasted",
    "pho": "pho vietnamese soup",
    "pizza": "pizza cheese",
    "pork_chop": "pork chop",
    "poutine": "poutine fries gravy",
    "prime_rib": "prime rib beef",
    "pulled_pork_sandwich": "pulled pork sandwich",
    "ramen": "ramen noodle soup",
    "ravioli": "ravioli cheese",
    "red_velvet_cake": "red velvet cake",
    "risotto": "risotto rice",
    "samosa": "samosa fried",
    "sashimi": "sashimi fish raw",
    "scallops": "scallops",
    "seaweed_salad": "seaweed salad",
    "shrimp_and_grits": "shrimp and grits",
    "spaghetti_bolognese": "spaghetti bolognese meat sauce",
    "spaghetti_carbonara": "spaghetti carbonara",
    "spring_rolls": "spring roll",
    "steak": "steak beef sirloin",
    "strawberry_shortcake": "strawberry shortcake",
    "sushi": "sushi roll",
    "tacos": "taco beef",
    "takoyaki": "takoyaki octopus",
    "tiramisu": "tiramisu",
    "tuna_tartare": "tuna tartare raw",
    "waffles": "waffle",
}


def fetch_all_foods(api_key: str = None):
    """Fetch nutrition data for all Food-101 classes."""
    
    service = USDAService(api_key)
    results = []
    
    for i, (food_id, search_term) in enumerate(FOOD_101_SEARCH_TERMS.items()):
        print(f"[{i+1}/101] Fetching: {food_id} -> '{search_term}'")
        
        try:
            nutrition = service.get_nutrition_by_name(search_term)
            
            if nutrition:
                results.append({
                    "food_id": food_id,
                    "name": food_id.replace("_", " ").title(),
                    "usda_name": nutrition.food_name,
                    "fdc_id": nutrition.fdc_id,
                    "calories": nutrition.calories,
                    "protein_g": nutrition.protein_g,
                    "carbs_g": nutrition.carbs_g,
                    "fat_g": nutrition.fat_g,
                    "sugar_g": nutrition.sugar_g,
                    "fiber_g": nutrition.fiber_g,
                    "sodium_mg": nutrition.sodium_mg,
                    "serving_size": nutrition.serving_size,
                    "serving_unit": nutrition.serving_unit,
                    "data_source": "usda",
                })
                print(f"    ✓ Found: {nutrition.calories} kcal")
            else:
                print(f"    ✗ Not found, using estimate")
                results.append({
                    "food_id": food_id,
                    "name": food_id.replace("_", " ").title(),
                    "usda_name": "",
                    "fdc_id": 0,
                    "calories": 200,
                    "protein_g": 8,
                    "carbs_g": 25,
                    "fat_g": 8,
                    "sugar_g": 5,
                    "fiber_g": 2,
                    "sodium_mg": 300,
                    "serving_size": 100,
                    "serving_unit": "g",
                    "data_source": "estimated",
                })
            
            # Rate limiting - USDA has limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results.append({
                "food_id": food_id,
                "name": food_id.replace("_", " ").title(),
                "error": str(e),
                "data_source": "error",
            })
    
    return results


def save_to_csv(results: list, output_path: str):
    """Save results to CSV file."""
    
    fieldnames = [
        "food_id", "name", "brand", "serving_size", "serving_unit",
        "calories", "protein_g", "carbs_g", "sugar_g", "fiber_g",
        "fat_g", "saturated_fat_g", "sodium_mg", "potassium_mg",
        "cholesterol_mg", "glycemic_index", "allergens", "category", "data_source"
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for item in results:
            # Fill in missing fields with defaults
            row = {
                "brand": "null",
                "saturated_fat_g": 0,
                "potassium_mg": 0,
                "cholesterol_mg": 0,
                "glycemic_index": 50,
                "allergens": "",
                "category": "prepared_meal",
                **item
            }
            writer.writerow(row)
    
    print(f"\n✓ Saved {len(results)} foods to {output_path}")


def main():
    print("=" * 60)
    print("USDA FoodData Central - Food-101 Nutrition Fetcher")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("USDA_API_KEY", "DEMO_KEY")
    
    if api_key == "DEMO_KEY":
        print("\n⚠ WARNING: Using DEMO_KEY (rate limited)")
        print("Get your free API key at: https://fdc.nal.usda.gov/api-key-signup.html")
        print("Then set: set USDA_API_KEY=your_key_here")
        print()
        
        response = input("Continue with DEMO_KEY? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Set your API key and try again.")
            return
    else:
        print(f"✓ Using API key: {api_key[:8]}...")
    
    print("\n" + "=" * 60)
    print("Fetching nutrition data for 101 foods...")
    print("This may take a few minutes due to rate limiting.")
    print("=" * 60 + "\n")
    
    # Fetch all foods
    results = fetch_all_foods(api_key)
    
    # Save to CSV
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_foods.csv"
    )
    save_to_csv(results, output_path)
    
    # Also save as JSON for reference
    json_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "usda_nutrition.json"
    )
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved JSON backup to {json_path}")
    
    # Summary
    found = sum(1 for r in results if r.get("data_source") == "usda")
    print(f"\n✓ Complete! Found {found}/101 foods in USDA database.")


if __name__ == "__main__":
    main()
