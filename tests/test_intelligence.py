"""
Tests for Phase 2: Food Intelligence Layer

Tests cover:
- OCR parsing and extraction
- MealFixer suggestions
- RecipeGenerator output
- Router integration
"""

import pytest

from models.food import Food, NutritionInfo, FoodCategory
from models.user import UserProfile, HealthCondition, DailyIntake, DailyTargets
from rules.engine import RuleEngine, Severity
from ocr.parser import NutritionParser, ParsedNutrition
from ocr.error_handler import OCRErrorHandler
from ocr.service import NutritionOCRService
from intelligence.meal_fixer import MealFixer, FixType, ProblemCategory
from intelligence.recipe_generator import RecipeGenerator, CookingMethod


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def rule_engine():
    return RuleEngine()

@pytest.fixture 
def parser():
    return NutritionParser()

@pytest.fixture
def ocr_service():
    return NutritionOCRService()

@pytest.fixture
def meal_fixer(rule_engine):
    return MealFixer(rule_engine)

@pytest.fixture
def recipe_generator(rule_engine):
    return RecipeGenerator(rule_engine)

@pytest.fixture
def diabetic_user():
    return UserProfile(
        user_id="test-diabetic",
        name="Test User",
        conditions=[HealthCondition.DIABETES],
        allergens=[],
        daily_targets=DailyTargets.for_diabetes()
    )

@pytest.fixture
def user_with_allergies():
    return UserProfile(
        user_id="test-allergic",
        name="Allergic User",
        conditions=[],
        allergens=["peanuts", "dairy"]
    )

@pytest.fixture
def high_sugar_food():
    return Food(
        food_id="candy-001",
        name="Chocolate Candy Bar",
        serving_size=50,
        serving_unit="g",
        nutrition=NutritionInfo(
            calories=250,
            protein_g=3,
            carbs_g=35,
            sugar_g=28,
            fat_g=12,
            saturated_fat_g=7,
            sodium_mg=50
        ),
        category=FoodCategory.SNACK
    )


# =============================================================================
# OCR PARSER TESTS
# =============================================================================

class TestNutritionParser:
    """Tests for regex-based nutrition parsing."""
    
    def test_detect_nutrition_panel(self, parser):
        """Should detect nutrition facts panel."""
        text = """
        Nutrition Facts
        Serving Size 1 bar (50g)
        Calories 250
        Total Fat 12g
        Protein 3g
        """
        assert parser.has_nutrition_panel(text) is True
    
    def test_no_panel_detection(self, parser):
        """Should not detect panel when absent."""
        text = "Chocolate Bar Made with real cocoa"
        assert parser.has_nutrition_panel(text) is False
    
    def test_parse_calories(self, parser):
        """Should extract calorie value."""
        text = "Nutrition Facts\nCalories 250\nTotal Fat 12g"
        result = parser.parse(text)
        assert result.calories == 250
    
    def test_parse_macros(self, parser):
        """Should extract all macronutrients."""
        text = """
        Nutrition Facts
        Calories 200
        Total Fat 10g
        Total Carbohydrate 25g
        Protein 5g
        Sugars 15g
        """
        result = parser.parse(text)
        assert result.calories == 200
        assert result.fat_g == 10
        assert result.carbs_g == 25
        assert result.protein_g == 5
        assert result.sugar_g == 15
    
    def test_parse_sodium(self, parser):
        """Should extract sodium value."""
        text = "Sodium 480mg"
        result = parser.parse(text)
        assert result.sodium_mg == 480
    
    def test_parse_salt_to_sodium(self, parser):
        """Should convert salt to sodium."""
        text = "Salt 1.2g"  # 1.2g salt = 480mg sodium
        result = parser.parse(text)
        assert result.sodium_mg == pytest.approx(480, rel=0.1)
    
    def test_extract_food_name(self, parser):
        """Should extract product name."""
        text = """Snickers Bar
        Nutrition Facts
        Calories 250"""
        name = parser.extract_food_name(text)
        assert name == "Snickers Bar"


# =============================================================================
# OCR ERROR HANDLER TESTS
# =============================================================================

class TestOCRErrorHandler:
    """Tests for OCR error correction."""
    
    def test_clean_numeric_text(self):
        """Should fix common OCR number errors."""
        # O → 0
        assert "0" in OCRErrorHandler.clean_numeric_text("2O0")
        # l → 1
        assert "1" in OCRErrorHandler.clean_numeric_text("l00")
    
    def test_validate_reasonable_values(self):
        """Should accept reasonable nutrition values."""
        nutrition = ParsedNutrition(
            calories=250,
            carbs_g=30,
            protein_g=5,
            fat_g=12
        )
        is_valid, errors = OCRErrorHandler.validate_nutrition(nutrition)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_detect_negative_values(self):
        """Should flag negative values as errors."""
        nutrition = ParsedNutrition(calories=-100)
        is_valid, errors = OCRErrorHandler.validate_nutrition(nutrition)
        assert is_valid is False
        assert any("negative" in e.lower() for e in errors)
    
    def test_detect_sugar_exceeds_carbs(self):
        """Should detect when sugar exceeds carbs."""
        nutrition = ParsedNutrition(
            carbs_g=20,
            sugar_g=30  # Sugar > carbs is impossible
        )
        is_valid, errors = OCRErrorHandler.validate_nutrition(nutrition)
        assert is_valid is False


# =============================================================================
# MEAL FIXER TESTS
# =============================================================================

class TestMealFixer:
    """Tests for meal fix suggestions."""
    
    def test_detect_high_sugar(self, meal_fixer, high_sugar_food, diabetic_user):
        """Should detect and fix high sugar for diabetic user."""
        fixes = meal_fixer.analyze_and_fix([high_sugar_food], diabetic_user)
        
        assert len(fixes) > 0
        sugar_fixes = [f for f in fixes if f.problem == ProblemCategory.EXCESS_SUGAR]
        assert len(sugar_fixes) >= 1
    
    def test_fix_has_explanation(self, meal_fixer, high_sugar_food, diabetic_user):
        """Each fix should have an explanation."""
        fixes = meal_fixer.analyze_and_fix([high_sugar_food], diabetic_user)
        
        for fix in fixes:
            assert fix.explanation != ""
            assert len(fix.explanation) > 10
    
    def test_fix_has_nutrition_impact(self, meal_fixer, high_sugar_food, diabetic_user):
        """Fixes should show nutritional impact."""
        fixes = meal_fixer.analyze_and_fix([high_sugar_food], diabetic_user)
        
        for fix in fixes:
            assert len(fix.nutrition_impact) > 0
    
    def test_no_fixes_for_healthy_meal(self, meal_fixer, diabetic_user):
        """Should return no fixes for healthy meal."""
        healthy_food = Food(
            food_id="apple-001",
            name="Apple",
            serving_size=150,
            serving_unit="g",
            nutrition=NutritionInfo(
                calories=80,
                carbs_g=20,
                sugar_g=10,
                fiber_g=4,
                fat_g=0.3,
                protein_g=0.5
            )
        )
        
        fixes = meal_fixer.analyze_and_fix([healthy_food], diabetic_user)
        # May have minor warnings but no major fixes required
        assert len([f for f in fixes if f.problem == ProblemCategory.EXCESS_SUGAR]) == 0
    
    def test_allergen_safety(self, meal_fixer, user_with_allergies, high_sugar_food):
        """Fixes should not suggest allergens."""
        fixes = meal_fixer.analyze_and_fix([high_sugar_food], user_with_allergies)
        
        for fix in fixes:
            if fix.replacement_food:
                assert "peanut" not in fix.replacement_food.lower()
                assert "dairy" not in fix.replacement_food.lower()
                assert "milk" not in fix.replacement_food.lower()


# =============================================================================
# RECIPE GENERATOR TESTS
# =============================================================================

class TestRecipeGenerator:
    """Tests for recipe generation."""
    
    def test_generate_with_protein_and_vegetable(self, recipe_generator, diabetic_user):
        """Should generate recipe with protein and vegetable."""
        ingredients = ["chicken breast", "broccoli", "rice"]
        recipe = recipe_generator.generate(ingredients, diabetic_user)
        
        assert recipe is not None
        assert len(recipe.ingredients) >= 2
        assert recipe.name != ""
    
    def test_recipe_has_instructions(self, recipe_generator, diabetic_user):
        """Recipe should have cooking instructions."""
        ingredients = ["salmon", "spinach"]
        recipe = recipe_generator.generate(ingredients, diabetic_user)
        
        assert recipe is not None
        assert len(recipe.instructions) > 0
    
    def test_recipe_has_nutrition(self, recipe_generator, diabetic_user):
        """Recipe should calculate total nutrition."""
        ingredients = ["chicken", "broccoli"]
        recipe = recipe_generator.generate(ingredients, diabetic_user)
        
        assert recipe is not None
        assert recipe.total_nutrition["calories"] > 0
        assert recipe.total_nutrition["protein_g"] > 0
    
    def test_filters_allergens(self, recipe_generator, user_with_allergies):
        """Should filter out allergen ingredients."""
        ingredients = ["peanut butter", "chicken", "broccoli"]
        recipe = recipe_generator.generate(ingredients, user_with_allergies)
        
        if recipe:
            ing_names = [ing[0].lower() for ing in recipe.ingredients]
            assert not any("peanut" in name for name in ing_names)
    
    def test_includes_medical_notes(self, recipe_generator, diabetic_user):
        """Recipe should include medical notes."""
        ingredients = ["chicken", "broccoli", "quinoa"]
        recipe = recipe_generator.generate(ingredients, diabetic_user)
        
        assert recipe is not None
        assert len(recipe.medical_notes) > 0
    
    def test_no_recipe_if_all_allergens(self, recipe_generator, user_with_allergies):
        """Should return None if all ingredients are allergens."""
        ingredients = ["peanuts", "milk", "cheese"]
        recipe = recipe_generator.generate(ingredients, user_with_allergies)
        
        # Either no recipe or minimal ingredients
        assert recipe is None or len(recipe.ingredients) == 0


# =============================================================================
# OCR SERVICE TESTS
# =============================================================================

class TestOCRService:
    """Tests for OCR service text extraction."""
    
    def test_extract_from_text_with_panel(self, ocr_service):
        """Should extract nutrition from text with panel."""
        text = """
        Nutrition Facts
        Serving Size 1 bar (50g)
        Calories 250
        Total Fat 12g
        Protein 5g
        Total Carbohydrate 30g
        Sugars 20g
        """
        result = ocr_service.extract_from_text(text)
        
        assert result.nutrition.is_complete()
        assert result.source.value == "ocr"
        assert result.nutrition.calories == 250
    
    def test_extract_from_text_without_panel(self, ocr_service):
        """Should fall back to RAG when no panel."""
        text = "Chocolate Chip Cookie from bakery"
        result = ocr_service.extract_from_text(text)
        
        assert result.requires_rag_lookup is True
        assert result.source.value == "rag"
        assert result.food_name is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_text_to_fix_pipeline(
        self, ocr_service, meal_fixer, rule_engine, diabetic_user
    ):
        """Test complete pipeline from text to fixes."""
        # Step 1: Extract from text
        text = """
        Nutrition Facts
        Calories 350
        Total Fat 15g
        Total Carbohydrate 45g
        Sugars 30g
        Protein 4g
        """
        extraction = ocr_service.extract_from_text(text)
        
        assert extraction.nutrition.is_complete()
        
        # Step 2: Create Food object
        food = Food(
            food_id="test-food",
            name="Test Sweet",
            serving_size=100,
            serving_unit="g",
            nutrition=NutritionInfo(
                calories=extraction.nutrition.calories or 0,
                carbs_g=extraction.nutrition.carbs_g or 0,
                sugar_g=extraction.nutrition.sugar_g or 0,
                fat_g=extraction.nutrition.fat_g or 0,
                protein_g=extraction.nutrition.protein_g or 0
            )
        )
        
        # Step 3: Validate with rules
        violations = rule_engine.evaluate(food, diabetic_user)
        assert len(violations) > 0  # Should have sugar violations
        
        # Step 4: Get fixes
        fixes = meal_fixer.analyze_and_fix([food], diabetic_user)
        assert len(fixes) > 0
