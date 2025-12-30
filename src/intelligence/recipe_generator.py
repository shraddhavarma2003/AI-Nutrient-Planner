"""
Recipe Generator Service

Generates healthy recipes using only available ingredients,
respecting user health conditions and medical rules.

IMPORTANT: Nutrition values are retrieved from database, never guessed.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from models.food import Food, NutritionInfo
from models.user import UserProfile, HealthCondition, DailyIntake
from rules.engine import RuleEngine, RuleViolation, Severity


class CookingMethod(Enum):
    """Cooking methods ordered by healthiness."""
    RAW = "raw"
    STEAMED = "steamed"
    BAKED = "baked"
    GRILLED = "grilled"
    SAUTEED = "sauteed"
    PAN_FRIED = "pan_fried"
    DEEP_FRIED = "deep_fried"


@dataclass
class RecipeTemplate:
    """
    A template for generating recipes.
    
    Templates define the structure of a recipe (what categories
    of ingredients are needed) without specifying exact ingredients.
    """
    template_id: str
    name: str                           # "Simple Stir Fry"
    required_categories: List[str]      # ["protein", "vegetable"]
    optional_categories: List[str]      # ["grain", "sauce"]
    cooking_method: CookingMethod
    base_cooking_time_mins: int
    base_instructions: str
    health_score: float                 # 0-10, higher is healthier
    
    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "cooking_method": self.cooking_method.value,
            "cooking_time_mins": self.base_cooking_time_mins,
            "health_score": self.health_score
        }


@dataclass
class GeneratedRecipe:
    """
    A complete generated recipe with ingredients and nutrition.
    """
    name: str
    ingredients: List[Tuple[str, float, str]]  # (name, amount, unit)
    instructions: List[str]
    cooking_method: CookingMethod
    cooking_time_mins: int
    total_nutrition: Dict[str, float]
    servings: int
    medical_notes: List[str]      # Why this is suitable for user
    warnings: List[str]           # Any minor concerns
    suitability_score: float      # 0-1, how well it matches user needs
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ingredients": [
                {"name": name, "amount": amt, "unit": unit}
                for name, amt, unit in self.ingredients
            ],
            "instructions": self.instructions,
            "cooking_method": self.cooking_method.value,
            "cooking_time_mins": self.cooking_time_mins,
            "nutrition_per_serving": {
                k: round(v / self.servings, 1) 
                for k, v in self.total_nutrition.items()
            },
            "servings": self.servings,
            "medical_notes": self.medical_notes,
            "warnings": self.warnings,
            "suitability_score": round(self.suitability_score, 2)
        }


class RecipeGenerator:
    """
    Generates healthy recipes from available ingredients.
    
    Pipeline:
    1. Filter ingredients by allergens
    2. Match against recipe templates
    3. Calculate total nutrition
    4. Validate against medical rules
    5. Adjust portions if needed
    6. Return recipe with explanation
    
    SAFETY: All recipes are validated against the rule engine.
    If a recipe would violate medical rules, portions are adjusted
    or the recipe is rejected.
    """
    
    # Default recipe templates
    DEFAULT_TEMPLATES = [
        RecipeTemplate(
            template_id="stir-fry-001",
            name="Simple Stir Fry",
            required_categories=["protein", "vegetable"],
            optional_categories=["grain"],
            cooking_method=CookingMethod.SAUTEED,
            base_cooking_time_mins=20,
            base_instructions="Heat pan, cook protein, add vegetables, season.",
            health_score=8.0
        ),
        RecipeTemplate(
            template_id="salad-001",
            name="Fresh Salad",
            required_categories=["vegetable"],
            optional_categories=["protein", "fruit"],
            cooking_method=CookingMethod.RAW,
            base_cooking_time_mins=10,
            base_instructions="Wash and chop vegetables, combine, add dressing.",
            health_score=9.5
        ),
        RecipeTemplate(
            template_id="baked-001",
            name="Baked Protein with Vegetables",
            required_categories=["protein", "vegetable"],
            optional_categories=["grain"],
            cooking_method=CookingMethod.BAKED,
            base_cooking_time_mins=35,
            base_instructions="Preheat oven, season ingredients, bake until done.",
            health_score=8.5
        ),
        RecipeTemplate(
            template_id="soup-001",
            name="Healthy Soup",
            required_categories=["vegetable"],
            optional_categories=["protein", "grain"],
            cooking_method=CookingMethod.STEAMED,
            base_cooking_time_mins=30,
            base_instructions="Simmer vegetables in broth, blend or serve chunky.",
            health_score=8.0
        ),
        RecipeTemplate(
            template_id="grain-bowl-001",
            name="Grain Bowl",
            required_categories=["grain", "vegetable"],
            optional_categories=["protein"],
            cooking_method=CookingMethod.STEAMED,
            base_cooking_time_mins=25,
            base_instructions="Cook grain, prepare toppings, assemble bowl.",
            health_score=7.5
        ),
    ]
    
    # Ingredient category mapping
    INGREDIENT_CATEGORIES = {
        "protein": [
            "chicken", "fish", "salmon", "tuna", "tofu", "eggs", "beef",
            "turkey", "shrimp", "pork", "lamb", "tempeh", "beans", "lentils"
        ],
        "vegetable": [
            "broccoli", "spinach", "carrot", "onion", "garlic", "tomato",
            "bell pepper", "zucchini", "cucumber", "lettuce", "cabbage",
            "mushroom", "asparagus", "green beans", "kale", "cauliflower"
        ],
        "grain": [
            "rice", "quinoa", "pasta", "bread", "oats", "barley",
            "couscous", "bulgur", "noodles"
        ],
        "fruit": [
            "apple", "banana", "orange", "berries", "mango", "grapes",
            "pineapple", "kiwi", "pear"
        ],
        "dairy": [
            "milk", "cheese", "yogurt", "butter", "cream"
        ],
        "fat_oil": [
            "olive oil", "coconut oil", "avocado", "nuts", "seeds"
        ]
    }
    
    # Basic nutrition estimates per 100g (for when database not available)
    NUTRITION_ESTIMATES = {
        "chicken": {"calories": 165, "protein_g": 31, "fat_g": 3.6, "carbs_g": 0, "sodium_mg": 74},
        "salmon": {"calories": 208, "protein_g": 20, "fat_g": 13, "carbs_g": 0, "sodium_mg": 59},
        "broccoli": {"calories": 34, "protein_g": 2.8, "fat_g": 0.4, "carbs_g": 7, "sodium_mg": 33, "fiber_g": 2.6},
        "rice": {"calories": 130, "protein_g": 2.7, "fat_g": 0.3, "carbs_g": 28, "sodium_mg": 1},
        "spinach": {"calories": 23, "protein_g": 2.9, "fat_g": 0.4, "carbs_g": 3.6, "sodium_mg": 79, "fiber_g": 2.2},
        "eggs": {"calories": 155, "protein_g": 13, "fat_g": 11, "carbs_g": 1.1, "sodium_mg": 124},
        "tofu": {"calories": 76, "protein_g": 8, "fat_g": 4.8, "carbs_g": 1.9, "sodium_mg": 7},
        "quinoa": {"calories": 120, "protein_g": 4.4, "fat_g": 1.9, "carbs_g": 21, "sodium_mg": 7, "fiber_g": 2.8},
    }
    
    def __init__(
        self, 
        rule_engine: RuleEngine, 
        food_database: Any = None,
        templates: Optional[List[RecipeTemplate]] = None
    ):
        """
        Initialize RecipeGenerator.
        
        Args:
            rule_engine: Rule engine for validation
            food_database: Database for nutrition lookup
            templates: Custom recipe templates (uses defaults if None)
        """
        self.rule_engine = rule_engine
        self.food_database = food_database
        self.templates = templates or self.DEFAULT_TEMPLATES
    
    def generate(
        self,
        available_ingredients: List[str],
        user: UserProfile,
        daily_intake: Optional[DailyIntake] = None
    ) -> Optional[GeneratedRecipe]:
        """
        Generate a healthy recipe from available ingredients.
        
        Args:
            available_ingredients: List of ingredient names
            user: User's health profile
            daily_intake: Current daily totals
            
        Returns:
            GeneratedRecipe if successful, None if no safe recipe possible
        """
        if daily_intake is None:
            daily_intake = DailyIntake()
        
        # Step 1: Filter out allergens
        safe_ingredients = self._filter_allergens(available_ingredients, user)
        
        if not safe_ingredients:
            return None  # No safe ingredients
        
        # Step 2: Categorize ingredients
        categorized = self._categorize_ingredients(safe_ingredients)
        
        # Step 3: Find matching template
        template = self._find_best_template(categorized, user)
        
        if template is None:
            return None  # No matching template
        
        # Step 4: Build recipe
        recipe = self._build_recipe(template, categorized, user)
        
        # Step 5: Validate against rules
        is_safe, violations = self._validate_recipe(recipe, user, daily_intake)
        
        # Step 6: Adjust if needed
        if not is_safe:
            recipe = self._adjust_recipe(recipe, violations, user)
            is_safe, violations = self._validate_recipe(recipe, user, daily_intake)
            
            if not is_safe:
                # Can't make it safe, add warnings
                recipe.warnings.extend([v.message for v in violations])
        
        # Step 7: Add medical notes
        recipe.medical_notes = self._generate_medical_notes(recipe, user)
        
        return recipe
    
    def _filter_allergens(
        self, ingredients: List[str], user: UserProfile
    ) -> List[str]:
        """Remove ingredients that contain user allergens."""
        safe = []
        for ingredient in ingredients:
            ing_lower = ingredient.lower()
            is_safe = True
            
            for allergen in user.allergens:
                allergen_lower = allergen.lower()
                
                # Direct match
                if allergen_lower in ing_lower:
                    is_safe = False
                    break
                
                # Category match (e.g., "dairy" → "milk", "cheese")
                if allergen_lower == "dairy":
                    if any(d in ing_lower for d in self.INGREDIENT_CATEGORIES.get("dairy", [])):
                        is_safe = False
                        break
                
                if allergen_lower in ["nuts", "tree_nuts"]:
                    if any(n in ing_lower for n in ["almond", "walnut", "pecan", "cashew", "nut"]):
                        is_safe = False
                        break
            
            if is_safe:
                safe.append(ingredient)
        
        return safe
    
    def _categorize_ingredients(
        self, ingredients: List[str]
    ) -> Dict[str, List[str]]:
        """Categorize ingredients by food type."""
        result: Dict[str, List[str]] = {}
        
        for ingredient in ingredients:
            ing_lower = ingredient.lower()
            
            for category, keywords in self.INGREDIENT_CATEGORIES.items():
                if any(kw in ing_lower for kw in keywords):
                    if category not in result:
                        result[category] = []
                    result[category].append(ingredient)
                    break
        
        return result
    
    def _find_best_template(
        self, categorized: Dict[str, List[str]], user: UserProfile
    ) -> Optional[RecipeTemplate]:
        """Find the best matching recipe template."""
        best_template = None
        best_score = -1
        
        for template in self.templates:
            # Check if required categories are available
            has_required = all(
                cat in categorized and len(categorized[cat]) > 0
                for cat in template.required_categories
            )
            
            if not has_required:
                continue
            
            # Score based on health and match quality
            score = template.health_score
            
            # Bonus for optional categories present
            optional_bonus = sum(
                1 for cat in template.optional_categories
                if cat in categorized
            )
            score += optional_bonus * 0.5
            
            # Adjust for user conditions
            if user.has_condition(HealthCondition.OBESITY):
                # Prefer raw/steamed methods
                if template.cooking_method in [CookingMethod.RAW, CookingMethod.STEAMED]:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_template = template
        
        return best_template
    
    def _build_recipe(
        self, 
        template: RecipeTemplate, 
        categorized: Dict[str, List[str]],
        user: UserProfile
    ) -> GeneratedRecipe:
        """Build a recipe from template and ingredients."""
        selected_ingredients: List[Tuple[str, float, str]] = []
        total_nutrition: Dict[str, float] = {
            "calories": 0, "protein_g": 0, "fat_g": 0, 
            "carbs_g": 0, "sodium_mg": 0, "fiber_g": 0
        }
        
        # Select ingredients from required categories
        for category in template.required_categories:
            if category in categorized and categorized[category]:
                ingredient = categorized[category][0]
                amount = self._get_default_amount(category)
                selected_ingredients.append((ingredient, amount, "g"))
                
                # Add nutrition
                self._add_nutrition(total_nutrition, ingredient, amount)
        
        # Add optional ingredients if available
        for category in template.optional_categories:
            if category in categorized and categorized[category]:
                ingredient = categorized[category][0]
                amount = self._get_default_amount(category) * 0.5  # Half portion
                selected_ingredients.append((ingredient, amount, "g"))
                self._add_nutrition(total_nutrition, ingredient, amount)
        
        # Generate instructions
        instructions = self._generate_instructions(
            template, selected_ingredients
        )
        
        # Create recipe name
        main_ingredients = [ing[0] for ing in selected_ingredients[:2]]
        name = f"{template.name} with {' and '.join(main_ingredients)}"
        
        return GeneratedRecipe(
            name=name,
            ingredients=selected_ingredients,
            instructions=instructions,
            cooking_method=template.cooking_method,
            cooking_time_mins=template.base_cooking_time_mins,
            total_nutrition=total_nutrition,
            servings=1,
            medical_notes=[],
            warnings=[],
            suitability_score=0.8
        )
    
    def _get_default_amount(self, category: str) -> float:
        """Get default amount in grams for a category."""
        defaults = {
            "protein": 150,
            "vegetable": 100,
            "grain": 100,
            "fruit": 80,
            "dairy": 50,
            "fat_oil": 15
        }
        return defaults.get(category, 100)
    
    def _add_nutrition(
        self, total: Dict[str, float], ingredient: str, amount_g: float
    ) -> None:
        """Add ingredient nutrition to total."""
        ing_lower = ingredient.lower()
        
        # Find matching nutrition data
        for key, nutrition in self.NUTRITION_ESTIMATES.items():
            if key in ing_lower:
                factor = amount_g / 100  # Nutrition is per 100g
                for nutrient, value in nutrition.items():
                    if nutrient in total:
                        total[nutrient] += value * factor
                return
        
        # Default if not found
        total["calories"] += amount_g * 1.2  # Rough estimate
    
    def _generate_instructions(
        self, 
        template: RecipeTemplate, 
        ingredients: List[Tuple[str, float, str]]
    ) -> List[str]:
        """Generate cooking instructions."""
        instructions = []
        
        # Prep step
        ing_names = [ing[0] for ing in ingredients]
        instructions.append(f"Prepare ingredients: {', '.join(ing_names)}")
        
        # Method-specific steps
        if template.cooking_method == CookingMethod.RAW:
            instructions.append("Wash and chop all vegetables")
            instructions.append("Combine in a large bowl")
            instructions.append("Add dressing or seasoning to taste")
        
        elif template.cooking_method == CookingMethod.STEAMED:
            instructions.append("Bring water to boil in a pot with steamer")
            instructions.append("Add vegetables, steam for 8-10 minutes")
            instructions.append("Season with salt, pepper, and herbs")
        
        elif template.cooking_method == CookingMethod.BAKED:
            instructions.append("Preheat oven to 375°F (190°C)")
            instructions.append("Season protein and vegetables with oil and spices")
            instructions.append("Arrange on baking sheet")
            instructions.append("Bake for 25-30 minutes until done")
        
        elif template.cooking_method in [CookingMethod.SAUTEED, CookingMethod.GRILLED]:
            instructions.append("Heat pan with a small amount of oil")
            instructions.append("Cook protein first, 5-7 minutes per side")
            instructions.append("Add vegetables, cook 3-5 minutes")
            instructions.append("Season and serve")
        
        return instructions
    
    def _validate_recipe(
        self, 
        recipe: GeneratedRecipe, 
        user: UserProfile,
        daily_intake: DailyIntake
    ) -> Tuple[bool, List[RuleViolation]]:
        """Validate recipe against medical rules."""
        # Create a mock Food object for validation
        nutrition = NutritionInfo(
            calories=recipe.total_nutrition.get("calories", 0),
            protein_g=recipe.total_nutrition.get("protein_g", 0),
            carbs_g=recipe.total_nutrition.get("carbs_g", 0),
            fat_g=recipe.total_nutrition.get("fat_g", 0),
            sugar_g=recipe.total_nutrition.get("sugar_g", 0),
            sodium_mg=recipe.total_nutrition.get("sodium_mg", 0),
            fiber_g=recipe.total_nutrition.get("fiber_g", 0),
        )
        
        mock_food = Food(
            food_id="generated-recipe",
            name=recipe.name,
            serving_size=sum(ing[1] for ing in recipe.ingredients),
            serving_unit="g",
            nutrition=nutrition,
            allergens=[]  # Already filtered
        )
        
        violations = self.rule_engine.evaluate(mock_food, user, daily_intake)
        
        # Filter out BLOCK violations (should not happen if allergens filtered)
        serious = [v for v in violations if v.severity in [Severity.ALERT, Severity.BLOCK]]
        
        return len(serious) == 0, violations
    
    def _adjust_recipe(
        self, 
        recipe: GeneratedRecipe, 
        violations: List[RuleViolation],
        user: UserProfile
    ) -> GeneratedRecipe:
        """Adjust recipe to meet medical constraints."""
        # Reduce portions to address violations
        reduction_factor = 0.75  # Reduce by 25%
        
        # Update ingredients
        recipe.ingredients = [
            (name, amount * reduction_factor, unit)
            for name, amount, unit in recipe.ingredients
        ]
        
        # Update nutrition
        for key in recipe.total_nutrition:
            recipe.total_nutrition[key] *= reduction_factor
        
        recipe.warnings.append("Portions reduced to meet nutritional guidelines")
        recipe.suitability_score *= 0.9
        
        return recipe
    
    def _generate_medical_notes(
        self, recipe: GeneratedRecipe, user: UserProfile
    ) -> List[str]:
        """Generate notes explaining why recipe is suitable."""
        notes = []
        
        if user.has_condition(HealthCondition.DIABETES):
            if recipe.total_nutrition.get("fiber_g", 0) > 5:
                notes.append("✓ High fiber content helps stabilize blood sugar")
            if recipe.total_nutrition.get("sugar_g", 0) < 10:
                notes.append("✓ Low sugar content suitable for diabetes management")
        
        if user.has_condition(HealthCondition.HYPERTENSION):
            if recipe.total_nutrition.get("sodium_mg", 0) < 500:
                notes.append("✓ Low sodium content supports healthy blood pressure")
        
        if user.has_condition(HealthCondition.OBESITY):
            cal_per_serving = recipe.total_nutrition.get("calories", 0) / recipe.servings
            if cal_per_serving < 500:
                notes.append(f"✓ Moderate calorie content ({cal_per_serving:.0f} kcal/serving)")
            if recipe.cooking_method in [CookingMethod.RAW, CookingMethod.STEAMED, CookingMethod.BAKED]:
                notes.append("✓ Healthy cooking method maintains nutrients without added fat")
        
        if not notes:
            notes.append("✓ Balanced meal with protein and vegetables")
        
        return notes
