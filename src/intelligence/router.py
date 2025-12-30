"""
Food Intelligence Router

Central router that processes food images through the complete pipeline:
OCR → Rules → Intelligence Features

This is the main entry point for all food-related operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from models.food import Food, NutritionInfo, FoodCategory, DataSource
from models.user import UserProfile, HealthCondition, DailyIntake
from rules.engine import RuleEngine, RuleViolation, Severity
from ocr.service import NutritionOCRService, FoodExtractionResult
from ocr.parser import NutritionSource, ParsedNutrition


class FeatureType(Enum):
    """Available features in the intelligence layer."""
    LOG_MEAL = "log_meal"
    FIX_MEAL = "fix_meal"
    GENERATE_RECIPE = "generate_recipe"
    SCAN_FOOD = "scan_food"


@dataclass
class RoutingResult:
    """
    Result from routing a request through the intelligence layer.
    """
    feature: FeatureType
    success: bool
    can_proceed: bool           # Whether user can proceed with action
    blocking_reason: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "feature": self.feature.value,
            "success": self.success,
            "can_proceed": self.can_proceed,
            "blocking_reason": self.blocking_reason,
            "data": self.data,
            "warnings": self.warnings
        }


class FoodIntelligenceRouter:
    """
    Central router for all food intelligence features.
    
    Routes requests through:
    1. OCR extraction (if image provided)
    2. RAG lookup (if OCR doesn't find nutrition)
    3. Rule engine validation
    4. Feature-specific processing
    
    SAFETY PRINCIPLE:
    All routes pass through the rule engine BEFORE any output.
    Medical rules always override AI suggestions.
    
    Usage:
        router = FoodIntelligenceRouter(...)
        
        # Scan food image
        result = router.scan_food(image_path, user, daily)
        
        # Fix problematic meal
        result = router.fix_meal(foods, user, daily)
        
        # Generate recipe
        result = router.generate_recipe(ingredients, user, daily)
    """
    
    def __init__(
        self,
        rule_engine: RuleEngine,
        ocr_service: Optional[NutritionOCRService] = None,
        rag_service: Any = None,
        meal_fixer: Any = None,
        recipe_generator: Any = None
    ):
        """
        Initialize the router with all required services.
        
        Args:
            rule_engine: Medical rule engine (required)
            ocr_service: OCR service for label extraction
            rag_service: RAG service for database lookup
            meal_fixer: MealFixer service
            recipe_generator: RecipeGenerator service
        """
        self.rule_engine = rule_engine
        self.ocr_service = ocr_service or NutritionOCRService()
        self.rag_service = rag_service
        self.meal_fixer = meal_fixer
        self.recipe_generator = recipe_generator
    
    def scan_food(
        self,
        image_path: str,
        user: UserProfile,
        daily_intake: Optional[DailyIntake] = None
    ) -> RoutingResult:
        """
        Scan a food image and extract nutrition information.
        
        Pipeline:
        1. OCR the image
        2. If nutrition panel found → use OCR values
        3. If no panel → RAG lookup by food name
        4. Validate against medical rules
        5. Return structured result
        
        Args:
            image_path: Path to food image
            user: User's health profile
            daily_intake: Current daily totals
            
        Returns:
            RoutingResult with food data and any violations
        """
        if daily_intake is None:
            daily_intake = DailyIntake()
        
        # Step 1: OCR extraction
        extraction = self.ocr_service.extract(image_path)
        
        # Step 2: Fill nutrition via RAG if needed
        if extraction.requires_rag_lookup and self.rag_service:
            rag_nutrition = self.rag_service.lookup(extraction.food_name)
            if rag_nutrition:
                extraction.nutrition = rag_nutrition
                extraction.source = NutritionSource.RAG
            else:
                return RoutingResult(
                    feature=FeatureType.SCAN_FOOD,
                    success=False,
                    can_proceed=False,
                    blocking_reason="Could not identify food. Please enter nutrition manually.",
                    data={"food_name": extraction.food_name}
                )
        
        # Step 3: Create Food object
        food = self._create_food_from_extraction(extraction)
        
        if food is None:
            return RoutingResult(
                feature=FeatureType.SCAN_FOOD,
                success=False,
                can_proceed=False,
                blocking_reason="Insufficient nutrition data extracted",
                warnings=extraction.warnings
            )
        
        # Step 4: Validate against rules
        violations = self.rule_engine.evaluate(food, user, daily_intake)
        verdict = self.rule_engine.get_final_verdict(violations)
        
        # Step 5: Build result
        return RoutingResult(
            feature=FeatureType.SCAN_FOOD,
            success=True,
            can_proceed=verdict != Severity.BLOCK,
            blocking_reason=self._get_blocking_reason(violations) if verdict == Severity.BLOCK else None,
            data={
                "food": food.to_dict(),
                "source": extraction.source.value,
                "confidence": extraction.confidence,
                "verdict": verdict.value,
                "violations": [v.to_dict() for v in violations],
                "suggested_action": self._suggest_action(verdict)
            },
            warnings=extraction.warnings
        )
    
    def log_meal(
        self,
        foods: List[Food],
        user: UserProfile,
        daily_intake: Optional[DailyIntake] = None
    ) -> RoutingResult:
        """
        Log a meal (one or more foods).
        
        Validates all foods against rules before logging.
        """
        if daily_intake is None:
            daily_intake = DailyIntake()
        
        all_violations: List[RuleViolation] = []
        blocked_foods: List[str] = []
        
        for food in foods:
            violations = self.rule_engine.evaluate(food, user, daily_intake)
            all_violations.extend(violations)
            
            verdict = self.rule_engine.get_final_verdict(violations)
            if verdict == Severity.BLOCK:
                blocked_foods.append(food.name)
        
        # Block if any food has BLOCK severity
        if blocked_foods:
            return RoutingResult(
                feature=FeatureType.LOG_MEAL,
                success=False,
                can_proceed=False,
                blocking_reason=f"Cannot log meal: {', '.join(blocked_foods)} blocked due to safety concerns",
                data={"blocked_foods": blocked_foods},
                warnings=[v.message for v in all_violations if v.severity == Severity.BLOCK]
            )
        
        # Calculate totals
        total_nutrition = self._sum_nutrition(foods)
        
        return RoutingResult(
            feature=FeatureType.LOG_MEAL,
            success=True,
            can_proceed=True,
            data={
                "logged": True,
                "foods": [f.to_dict() for f in foods],
                "total_nutrition": total_nutrition,
                "violations": [v.to_dict() for v in all_violations]
            },
            warnings=[v.message for v in all_violations if v.severity in [Severity.WARN, Severity.ALERT]]
        )
    
    def fix_meal(
        self,
        foods: List[Food],
        user: UserProfile,
        daily_intake: Optional[DailyIntake] = None
    ) -> RoutingResult:
        """
        Analyze meal and suggest fixes.
        """
        if self.meal_fixer is None:
            return RoutingResult(
                feature=FeatureType.FIX_MEAL,
                success=False,
                can_proceed=False,
                blocking_reason="MealFixer service not configured"
            )
        
        if daily_intake is None:
            daily_intake = DailyIntake()
        
        fixes = self.meal_fixer.analyze_and_fix(foods, user, daily_intake)
        
        return RoutingResult(
            feature=FeatureType.FIX_MEAL,
            success=True,
            can_proceed=True,
            data={
                "fixes": [f.to_dict() for f in fixes],
                "total_fixes": len(fixes),
                "original_meal": [f.to_dict() for f in foods]
            }
        )
    
    def generate_recipe(
        self,
        ingredients: List[str],
        user: UserProfile,
        daily_intake: Optional[DailyIntake] = None
    ) -> RoutingResult:
        """
        Generate a recipe from available ingredients.
        """
        if self.recipe_generator is None:
            return RoutingResult(
                feature=FeatureType.GENERATE_RECIPE,
                success=False,
                can_proceed=False,
                blocking_reason="RecipeGenerator service not configured"
            )
        
        if daily_intake is None:
            daily_intake = DailyIntake()
        
        recipe = self.recipe_generator.generate(ingredients, user, daily_intake)
        
        if recipe is None:
            return RoutingResult(
                feature=FeatureType.GENERATE_RECIPE,
                success=False,
                can_proceed=False,
                blocking_reason="Could not generate safe recipe with provided ingredients",
                data={"ingredients_provided": ingredients}
            )
        
        return RoutingResult(
            feature=FeatureType.GENERATE_RECIPE,
            success=True,
            can_proceed=True,
            data={"recipe": recipe.to_dict()}
        )
    
    def process_text_input(
        self,
        text: str,
        user: UserProfile,
        daily_intake: Optional[DailyIntake] = None
    ) -> RoutingResult:
        """
        Process text input (e.g., from OCR'd receipt or typed food name).
        """
        extraction = self.ocr_service.extract_from_text(text)
        
        if extraction.requires_rag_lookup and self.rag_service:
            rag_nutrition = self.rag_service.lookup(extraction.food_name)
            if rag_nutrition:
                extraction.nutrition = rag_nutrition
        
        food = self._create_food_from_extraction(extraction)
        
        if food is None:
            return RoutingResult(
                feature=FeatureType.SCAN_FOOD,
                success=False,
                can_proceed=False,
                blocking_reason="Could not extract food information from text"
            )
        
        return self.log_meal([food], user, daily_intake)
    
    def _create_food_from_extraction(
        self, extraction: FoodExtractionResult
    ) -> Optional[Food]:
        """Create Food object from extraction result."""
        n = extraction.nutrition
        
        if not n.is_complete():
            return None
        
        try:
            nutrition = NutritionInfo(
                calories=n.calories or 0,
                protein_g=n.protein_g or 0,
                carbs_g=n.carbs_g or 0,
                fat_g=n.fat_g or 0,
                sugar_g=n.sugar_g or 0,
                fiber_g=n.fiber_g or 0,
                sodium_mg=n.sodium_mg or 0,
                saturated_fat_g=n.saturated_fat_g or 0,
                cholesterol_mg=n.cholesterol_mg or 0,
                potassium_mg=n.potassium_mg or 0,
            )
            
            # Determine serving size
            serving_size = 100.0  # Default
            serving_unit = "g"
            if n.serving_size:
                try:
                    serving_size = float(n.serving_size)
                except ValueError:
                    pass
            if n.serving_unit:
                serving_unit = n.serving_unit
            
            return Food(
                food_id=f"ocr-{hash(extraction.food_name or 'unknown')}",
                name=extraction.food_name or "Unknown Food",
                serving_size=serving_size,
                serving_unit=serving_unit,
                nutrition=nutrition,
                allergens=[],
                category=FoodCategory.OTHER,
                data_source=DataSource.USDA if extraction.source == NutritionSource.RAG else DataSource.USER_VERIFIED
            )
        except (ValueError, TypeError):
            return None
    
    def _sum_nutrition(self, foods: List[Food]) -> dict:
        """Sum nutrition across multiple foods."""
        total = {
            "calories": 0, "protein_g": 0, "carbs_g": 0,
            "fat_g": 0, "sugar_g": 0, "sodium_mg": 0
        }
        
        for food in foods:
            total["calories"] += food.nutrition.calories
            total["protein_g"] += food.nutrition.protein_g
            total["carbs_g"] += food.nutrition.carbs_g
            total["fat_g"] += food.nutrition.fat_g
            total["sugar_g"] += food.nutrition.sugar_g
            total["sodium_mg"] += food.nutrition.sodium_mg
        
        return total
    
    def _get_blocking_reason(self, violations: List[RuleViolation]) -> str:
        """Get blocking reason from violations."""
        blocks = [v for v in violations if v.severity == Severity.BLOCK]
        if blocks:
            return "; ".join(v.message for v in blocks)
        return "Safety violation detected"
    
    def _suggest_action(self, verdict: Severity) -> str:
        """Suggest next action based on verdict."""
        if verdict == Severity.BLOCK:
            return "find_alternative"
        elif verdict == Severity.ALERT:
            return "review_and_fix"
        elif verdict == Severity.WARN:
            return "log_with_caution"
        return "log_meal"
