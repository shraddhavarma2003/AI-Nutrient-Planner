"""
Unit Tests for Medical Rule Engine

These tests verify that the rule engine correctly enforces
safety rules for various health conditions and allergens.
"""

import pytest
from models.food import Food, NutritionInfo, FoodCategory, DataSource
from models.user import UserProfile, HealthCondition, DailyIntake, DailyTargets
from rules.engine import RuleEngine, Severity


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def rule_engine():
    """Create a rule engine instance."""
    return RuleEngine()


@pytest.fixture
def healthy_apple():
    """A healthy, low-risk food."""
    return Food(
        food_id="apple-001",
        name="Apple, raw",
        serving_size=182,
        serving_unit="g",
        nutrition=NutritionInfo(
            calories=95,
            protein_g=0.5,
            carbs_g=25,
            sugar_g=19,
            fiber_g=4.4,
            fat_g=0.3,
            saturated_fat_g=0.1,
            sodium_mg=2,
            potassium_mg=195,
        ),
        allergens=[],
        category=FoodCategory.FRUIT,
    )


@pytest.fixture
def high_sodium_soup():
    """A high-sodium food for hypertension tests."""
    return Food(
        food_id="soup-001",
        name="Canned Chicken Noodle Soup",
        serving_size=240,
        serving_unit="ml",
        nutrition=NutritionInfo(
            calories=150,
            protein_g=8,
            carbs_g=18,
            sugar_g=2,
            fiber_g=1,
            fat_g=5,
            saturated_fat_g=1.5,
            sodium_mg=890,  # High sodium!
            potassium_mg=200,
        ),
        allergens=["wheat", "soy"],
        category=FoodCategory.PREPARED_MEAL,
    )


@pytest.fixture
def sugary_dessert():
    """A high-sugar food for diabetes tests."""
    return Food(
        food_id="dessert-001",
        name="Chocolate Cake Slice",
        serving_size=80,
        serving_unit="g",
        nutrition=NutritionInfo(
            calories=350,
            protein_g=4,
            carbs_g=50,
            sugar_g=35,  # Very high sugar!
            fiber_g=2,
            fat_g=16,
            saturated_fat_g=8,
            sodium_mg=300,
            potassium_mg=150,
            glycemic_index=85,  # High GI
        ),
        allergens=["wheat", "eggs", "dairy"],
        category=FoodCategory.SNACK,
    )


@pytest.fixture
def peanut_food():
    """A food containing peanuts for allergy tests."""
    return Food(
        food_id="peanut-001",
        name="Peanut Butter Sandwich",
        serving_size=100,
        serving_unit="g",
        nutrition=NutritionInfo(
            calories=350,
            protein_g=12,
            carbs_g=35,
            sugar_g=8,
            fiber_g=3,
            fat_g=18,
            saturated_fat_g=4,
            sodium_mg=400,
            potassium_mg=250,
        ),
        allergens=["peanuts", "wheat"],
        category=FoodCategory.SNACK,
    )


@pytest.fixture
def calorie_dense_food():
    """A calorie-dense food for obesity tests."""
    return Food(
        food_id="fried-001",
        name="Fried Chicken Wings",
        serving_size=100,
        serving_unit="g",
        nutrition=NutritionInfo(
            calories=450,  # 4.5 kcal/g - high density!
            protein_g=25,
            carbs_g=10,
            sugar_g=0,
            fiber_g=0,  # Very low fiber
            fat_g=35,
            saturated_fat_g=10,  # High saturated fat
            sodium_mg=500,
            potassium_mg=200,
        ),
        allergens=[],
        category=FoodCategory.PROTEIN,
    )


@pytest.fixture
def healthy_user():
    """A user with no health conditions."""
    return UserProfile(
        user_id="user-001",
        name="Healthy User",
        conditions=[],
        allergens=[],
    )


@pytest.fixture
def diabetic_user():
    """A user with diabetes."""
    return UserProfile(
        user_id="user-002",
        name="Diabetic User",
        conditions=[HealthCondition.DIABETES],
        allergens=[],
        daily_targets=DailyTargets.for_diabetes(),
    )


@pytest.fixture
def hypertensive_user():
    """A user with hypertension."""
    return UserProfile(
        user_id="user-003",
        name="Hypertensive User",
        conditions=[HealthCondition.HYPERTENSION],
        allergens=[],
        daily_targets=DailyTargets.for_hypertension(),
    )


@pytest.fixture
def user_with_peanut_allergy():
    """A user with peanut allergy."""
    return UserProfile(
        user_id="user-004",
        name="Allergic User",
        conditions=[],
        allergens=["peanuts"],
    )


@pytest.fixture
def obese_user():
    """A user managing obesity."""
    return UserProfile(
        user_id="user-005",
        name="Weight Loss User",
        conditions=[HealthCondition.OBESITY],
        allergens=[],
        daily_targets=DailyTargets.for_weight_loss(1500),
    )


# =============================================================================
# ALLERGY TESTS
# =============================================================================

class TestAllergyRules:
    """Tests for allergen detection and blocking."""
    
    def test_block_on_allergen_match(
        self, rule_engine, peanut_food, user_with_peanut_allergy
    ):
        """Food with user's allergen should be blocked."""
        violations = rule_engine.evaluate(peanut_food, user_with_peanut_allergy)
        
        assert len(violations) >= 1
        allergen_violations = [v for v in violations if v.category == "allergy"]
        assert len(allergen_violations) == 1
        assert allergen_violations[0].severity == Severity.BLOCK
        assert allergen_violations[0].rule_id == "AL-001"
        assert "peanuts" in allergen_violations[0].message.lower()
    
    def test_no_allergen_warning_when_no_match(
        self, rule_engine, healthy_apple, user_with_peanut_allergy
    ):
        """Food without allergens should not trigger allergen violations."""
        violations = rule_engine.evaluate(healthy_apple, user_with_peanut_allergy)
        
        allergen_blocks = [v for v in violations 
                          if v.category == "allergy" and v.severity == Severity.BLOCK]
        assert len(allergen_blocks) == 0
    
    def test_final_verdict_is_block(
        self, rule_engine, peanut_food, user_with_peanut_allergy
    ):
        """Final verdict should be BLOCK when allergen is detected."""
        violations = rule_engine.evaluate(peanut_food, user_with_peanut_allergy)
        verdict = rule_engine.get_final_verdict(violations)
        
        assert verdict == Severity.BLOCK


# =============================================================================
# DIABETES TESTS
# =============================================================================

class TestDiabetesRules:
    """Tests for diabetes-related rules."""
    
    def test_warn_on_high_sugar(
        self, rule_engine, sugary_dessert, diabetic_user
    ):
        """High sugar food should trigger warning for diabetic user."""
        violations = rule_engine.evaluate(sugary_dessert, diabetic_user)
        
        sugar_warnings = [v for v in violations if v.rule_id == "DM-001"]
        assert len(sugar_warnings) == 1
        assert sugar_warnings[0].severity == Severity.WARN
    
    def test_warn_on_high_gi(
        self, rule_engine, sugary_dessert, diabetic_user
    ):
        """High GI food should trigger warning for diabetic user."""
        violations = rule_engine.evaluate(sugary_dessert, diabetic_user)
        
        gi_warnings = [v for v in violations if v.rule_id == "DM-002"]
        assert len(gi_warnings) == 1
        assert "glycemic" in gi_warnings[0].message.lower()
    
    def test_alert_on_daily_sugar_exceeded(
        self, rule_engine, sugary_dessert, diabetic_user
    ):
        """Should alert when daily sugar limit would be exceeded."""
        # Simulate already consumed 10g sugar today
        daily = DailyIntake(sugar_g=10)
        
        violations = rule_engine.evaluate(sugary_dessert, diabetic_user, daily)
        
        # 10 + 35 = 45g > 25g limit
        daily_sugar_alerts = [v for v in violations if v.rule_id == "DM-004"]
        assert len(daily_sugar_alerts) == 1
        assert daily_sugar_alerts[0].severity == Severity.ALERT
    
    def test_no_diabetes_rules_for_healthy_user(
        self, rule_engine, sugary_dessert, healthy_user
    ):
        """Diabetes rules should not apply to users without diabetes."""
        violations = rule_engine.evaluate(sugary_dessert, healthy_user)
        
        diabetes_violations = [v for v in violations if v.category == "diabetes"]
        assert len(diabetes_violations) == 0


# =============================================================================
# HYPERTENSION TESTS
# =============================================================================

class TestHypertensionRules:
    """Tests for hypertension-related rules."""
    
    def test_warn_on_high_sodium(
        self, rule_engine, high_sodium_soup, hypertensive_user
    ):
        """High sodium food should trigger warning."""
        violations = rule_engine.evaluate(high_sodium_soup, hypertensive_user)
        
        sodium_violations = [v for v in violations 
                            if v.rule_id in ["HT-001", "HT-002"]]
        assert len(sodium_violations) >= 1
    
    def test_alert_on_very_high_sodium(
        self, rule_engine, hypertensive_user
    ):
        """Very high sodium (>1000mg) should trigger alert."""
        very_salty = Food(
            food_id="salty-001",
            name="Super Salty Chips",
            serving_size=100,
            serving_unit="g",
            nutrition=NutritionInfo(
                calories=500,
                sodium_mg=1200,  # Very high!
            ),
            allergens=[],
        )
        
        violations = rule_engine.evaluate(very_salty, hypertensive_user)
        
        alert_violations = [v for v in violations 
                          if v.rule_id == "HT-002" and v.severity == Severity.ALERT]
        assert len(alert_violations) == 1
    
    def test_alert_on_daily_sodium_exceeded(
        self, rule_engine, high_sodium_soup, hypertensive_user
    ):
        """Should alert when daily sodium limit would be exceeded."""
        # Already consumed 1000mg sodium today
        daily = DailyIntake(sodium_mg=1000)
        
        violations = rule_engine.evaluate(high_sodium_soup, hypertensive_user, daily)
        
        # 1000 + 890 = 1890mg > 1500mg limit
        daily_alerts = [v for v in violations if v.rule_id == "HT-003"]
        assert len(daily_alerts) == 1
        assert daily_alerts[0].severity == Severity.ALERT


# =============================================================================
# OBESITY TESTS
# =============================================================================

class TestObesityRules:
    """Tests for obesity/weight management rules."""
    
    def test_warn_on_calorie_density(
        self, rule_engine, calorie_dense_food, obese_user
    ):
        """High calorie density should trigger warning."""
        violations = rule_engine.evaluate(calorie_dense_food, obese_user)
        
        density_warnings = [v for v in violations if v.rule_id == "OB-001"]
        assert len(density_warnings) == 1
        assert "density" in density_warnings[0].message.lower()
    
    def test_warn_on_high_saturated_fat(
        self, rule_engine, calorie_dense_food, obese_user
    ):
        """High saturated fat should trigger warning."""
        violations = rule_engine.evaluate(calorie_dense_food, obese_user)
        
        fat_warnings = [v for v in violations if v.rule_id == "OB-002"]
        assert len(fat_warnings) == 1
    
    def test_warn_on_low_fiber(
        self, rule_engine, calorie_dense_food, obese_user
    ):
        """Very low fiber should trigger warning."""
        violations = rule_engine.evaluate(calorie_dense_food, obese_user)
        
        fiber_warnings = [v for v in violations if v.rule_id == "OB-004"]
        assert len(fiber_warnings) == 1
    
    def test_alert_on_daily_calories_exceeded(
        self, rule_engine, calorie_dense_food, obese_user
    ):
        """Should alert when daily calorie target would be exceeded."""
        # Already consumed 1200 kcal today (target is 1500)
        daily = DailyIntake(calories=1200)
        
        violations = rule_engine.evaluate(calorie_dense_food, obese_user, daily)
        
        # 1200 + 450 = 1650 > 1500 target
        calorie_alerts = [v for v in violations if v.rule_id == "OB-003"]
        assert len(calorie_alerts) == 1
        assert calorie_alerts[0].severity == Severity.ALERT


# =============================================================================
# VERDICT TESTS
# =============================================================================

class TestVerdictLogic:
    """Tests for final verdict calculation."""
    
    def test_allow_when_no_violations(
        self, rule_engine, healthy_apple, healthy_user
    ):
        """Should return ALLOW when no violations."""
        violations = rule_engine.evaluate(healthy_apple, healthy_user)
        verdict = rule_engine.get_final_verdict(violations)
        
        assert verdict == Severity.ALLOW
    
    def test_block_overrides_all(
        self, rule_engine, peanut_food
    ):
        """BLOCK severity should override WARN and ALERT."""
        # User with both allergy and diabetes
        user = UserProfile(
            user_id="complex-user",
            name="Complex User",
            conditions=[HealthCondition.DIABETES],
            allergens=["peanuts"],
        )
        
        violations = rule_engine.evaluate(peanut_food, user)
        verdict = rule_engine.get_final_verdict(violations)
        
        # Even though diabetes rules may trigger, allergy BLOCK wins
        assert verdict == Severity.BLOCK
    
    def test_alert_overrides_warn(
        self, rule_engine, sugary_dessert, diabetic_user
    ):
        """ALERT severity should override WARN."""
        # Set up daily intake to trigger ALERT
        daily = DailyIntake(sugar_g=10)
        
        violations = rule_engine.evaluate(sugary_dessert, diabetic_user, daily)
        verdict = rule_engine.get_final_verdict(violations)
        
        # Should have both WARN and ALERT, but ALERT wins
        assert verdict == Severity.ALERT


# =============================================================================
# FORMAT TESTS
# =============================================================================

class TestViolationFormatting:
    """Tests for violation formatting."""
    
    def test_format_no_violations(self, rule_engine):
        """Empty violations should return success message."""
        result = rule_engine.format_violations([])
        assert "✅" in result
        assert "No concerns" in result
    
    def test_format_includes_icons(
        self, rule_engine, peanut_food, user_with_peanut_allergy
    ):
        """Formatted output should include severity icons."""
        violations = rule_engine.evaluate(peanut_food, user_with_peanut_allergy)
        result = rule_engine.format_violations(violations)
        
        assert "🚫" in result  # BLOCK icon
        assert "AL-001" in result  # Rule ID
