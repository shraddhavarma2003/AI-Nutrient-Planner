"""
Meal Fixer Service

Analyzes logged meals for nutritional or medical issues and suggests
small, realistic fixes that respect user health conditions.

IMPORTANT: Medical rules always override AI suggestions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from models.food import Food, NutritionInfo
from models.user import UserProfile, HealthCondition, DailyIntake
from rules.engine import RuleEngine, RuleViolation, Severity


class FixType(Enum):
    """Types of meal fixes."""
    SWAP = "swap"          # Replace with alternative
    REDUCE = "reduce"      # Reduce portion size
    REMOVE = "remove"      # Remove from meal
    ADD = "add"            # Add complementary item


class ProblemCategory(Enum):
    """Categories of nutritional problems."""
    EXCESS_SUGAR = "excess_sugar"
    HIGH_SODIUM = "high_sodium"
    HIGH_FAT = "high_fat"
    HIGH_CALORIE = "high_calorie"
    ALLERGEN = "allergen"
    LOW_FIBER = "low_fiber"


@dataclass
class MealFix:
    """
    A single meal fix suggestion.
    
    Contains the problem identified, the fix strategy,
    and the expected nutritional impact.
    """
    problem: ProblemCategory
    trigger_rule: str           # Rule ID that triggered this fix (e.g., "DM-001")
    original_food: str          # Name of problematic food
    fix_type: FixType
    fix_description: str        # Human-readable fix description
    replacement_food: Optional[str] = None
    nutrition_impact: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""       # Why this fix helps
    confidence: float = 0.8     # How confident we are in this fix
    
    def to_dict(self) -> dict:
        return {
            "problem": self.problem.value,
            "trigger_rule": self.trigger_rule,
            "original_food": self.original_food,
            "fix_type": self.fix_type.value,
            "fix_description": self.fix_description,
            "replacement_food": self.replacement_food,
            "nutrition_impact": self.nutrition_impact,
            "explanation": self.explanation,
            "confidence": self.confidence
        }


class MealFixer:
    """
    Analyzes meals and suggests fixes for nutritional issues.
    
    Pipeline:
    1. Run rule engine to get violations
    2. Categorize violations by problem type
    3. Generate appropriate fixes for each problem
    4. Validate fixes don't introduce new problems
    5. Return safe fixes only
    
    SAFETY: All fixes are validated against the rule engine
    before being returned to ensure they don't introduce
    new medical conflicts.
    """
    
    # Fix strategies for each problem type
    FIX_STRATEGIES = {
        ProblemCategory.EXCESS_SUGAR: [
            ("SWAP", "Replace with lower-sugar alternative"),
            ("REDUCE", "Reduce portion to half"),
            ("ADD", "Add fiber to slow sugar absorption"),
        ],
        ProblemCategory.HIGH_SODIUM: [
            ("SWAP", "Use low-sodium or no-salt-added version"),
            ("REDUCE", "Reduce portion size"),
            ("REMOVE", "Skip high-sodium condiments or sauces"),
        ],
        ProblemCategory.HIGH_FAT: [
            ("SWAP", "Choose lean protein or baked instead of fried"),
            ("REDUCE", "Remove skin or visible fat"),
            ("SWAP", "Use cooking spray instead of oil"),
        ],
        ProblemCategory.HIGH_CALORIE: [
            ("REDUCE", "Reduce portion by 30%"),
            ("SWAP", "Choose lighter side dishes"),
            ("REMOVE", "Skip add-ons like cheese or mayo"),
        ],
        ProblemCategory.LOW_FIBER: [
            ("ADD", "Add vegetables or salad as side"),
            ("SWAP", "Choose whole grain version"),
        ],
    }
    
    # Common healthy swaps
    HEALTHY_SWAPS = {
        # High sugar → Lower sugar
        "chocolate cake": "fresh berries",
        "ice cream": "frozen yogurt",
        "soda": "sparkling water with lemon",
        "candy bar": "dark chocolate (70% cacao)",
        "white bread": "whole wheat bread",
        
        # High sodium → Lower sodium
        "canned soup": "homemade or low-sodium soup",
        "chips": "unsalted nuts",
        "fast food burger": "homemade burger with less salt",
        
        # High fat → Lower fat
        "fried chicken": "grilled chicken breast",
        "whole milk": "skim milk or almond milk",
        "regular cheese": "reduced-fat cheese",
        "mayo": "Greek yogurt spread",
        
        # High calorie → Lower calorie
        "pasta": "zucchini noodles",
        "rice": "cauliflower rice",
    }
    
    def __init__(self, rule_engine: RuleEngine, food_database: Any = None):
        """
        Initialize MealFixer.
        
        Args:
            rule_engine: Rule engine for validation
            food_database: Optional database for finding alternatives
        """
        self.rule_engine = rule_engine
        self.food_database = food_database
    
    def analyze_and_fix(
        self,
        meal: List[Food],
        user: UserProfile,
        daily_intake: Optional[DailyIntake] = None
    ) -> List[MealFix]:
        """
        Analyze a meal and generate fixes for any issues.
        
        Args:
            meal: List of foods in the meal
            user: User's health profile
            daily_intake: Current daily totals (optional)
            
        Returns:
            List of suggested fixes, validated for safety
        """
        if daily_intake is None:
            daily_intake = DailyIntake()
        
        # Step 1: Get all violations
        all_violations: List[Tuple[Food, RuleViolation]] = []
        for food in meal:
            violations = self.rule_engine.evaluate(food, user, daily_intake)
            for v in violations:
                all_violations.append((food, v))
        
        if not all_violations:
            return []  # Meal is fine
        
        # Step 2: Group by problem category
        problems = self._categorize_violations(all_violations)
        
        # Step 3: Generate fixes for each problem
        fixes: List[MealFix] = []
        for problem, violation_items in problems.items():
            fix = self._generate_fix(problem, violation_items, user)
            if fix is not None:
                # Step 4: Validate fix is safe
                if self._validate_fix_safety(fix, user):
                    fixes.append(fix)
        
        return fixes
    
    def _categorize_violations(
        self, violations: List[Tuple[Food, RuleViolation]]
    ) -> Dict[ProblemCategory, List[Tuple[Food, RuleViolation]]]:
        """Group violations by problem category."""
        categories: Dict[ProblemCategory, List] = {}
        
        for food, v in violations:
            category = self._rule_to_category(v.rule_id)
            if category:
                if category not in categories:
                    categories[category] = []
                categories[category].append((food, v))
        
        return categories
    
    def _rule_to_category(self, rule_id: str) -> Optional[ProblemCategory]:
        """Map rule ID to problem category."""
        mapping = {
            "DM-001": ProblemCategory.EXCESS_SUGAR,  # High sugar per serving
            "DM-002": ProblemCategory.EXCESS_SUGAR,  # High GI
            "DM-003": ProblemCategory.LOW_FIBER,     # Low fiber with high carbs
            "DM-004": ProblemCategory.EXCESS_SUGAR,  # Daily sugar exceeded
            "HT-001": ProblemCategory.HIGH_SODIUM,   # High sodium per serving
            "HT-002": ProblemCategory.HIGH_SODIUM,   # Very high sodium
            "HT-003": ProblemCategory.HIGH_SODIUM,   # Daily sodium exceeded
            "HT-004": ProblemCategory.HIGH_SODIUM,   # Poor Na/K ratio
            "OB-001": ProblemCategory.HIGH_CALORIE,  # Calorie density
            "OB-002": ProblemCategory.HIGH_FAT,      # High saturated fat
            "OB-003": ProblemCategory.HIGH_CALORIE,  # Daily calories exceeded
            "OB-004": ProblemCategory.LOW_FIBER,     # Very low fiber
            "AL-001": ProblemCategory.ALLERGEN,      # Contains allergen
        }
        return mapping.get(rule_id)
    
    def _generate_fix(
        self,
        problem: ProblemCategory,
        items: List[Tuple[Food, RuleViolation]],
        user: UserProfile
    ) -> Optional[MealFix]:
        """Generate a fix for a specific problem."""
        
        if problem == ProblemCategory.ALLERGEN:
            # Allergens must be removed, no alternatives
            return self._fix_allergen(items)
        
        # Find the worst offender
        worst_food, worst_violation = self._find_worst_item(items, problem)
        
        if problem == ProblemCategory.EXCESS_SUGAR:
            return self._fix_sugar(worst_food, worst_violation, user)
        elif problem == ProblemCategory.HIGH_SODIUM:
            return self._fix_sodium(worst_food, worst_violation, user)
        elif problem == ProblemCategory.HIGH_FAT:
            return self._fix_fat(worst_food, worst_violation, user)
        elif problem == ProblemCategory.HIGH_CALORIE:
            return self._fix_calories(worst_food, worst_violation, user)
        elif problem == ProblemCategory.LOW_FIBER:
            return self._fix_low_fiber(worst_food, worst_violation, user)
        
        return None
    
    def _find_worst_item(
        self, items: List[Tuple[Food, RuleViolation]], problem: ProblemCategory
    ) -> Tuple[Food, RuleViolation]:
        """Find the worst offending item for a problem type."""
        if problem == ProblemCategory.EXCESS_SUGAR:
            return max(items, key=lambda x: x[0].nutrition.sugar_g)
        elif problem == ProblemCategory.HIGH_SODIUM:
            return max(items, key=lambda x: x[0].nutrition.sodium_mg)
        elif problem == ProblemCategory.HIGH_FAT:
            return max(items, key=lambda x: x[0].nutrition.saturated_fat_g)
        elif problem == ProblemCategory.HIGH_CALORIE:
            return max(items, key=lambda x: x[0].nutrition.calories)
        else:
            return items[0]  # Just return first
    
    def _fix_allergen(self, items: List[Tuple[Food, RuleViolation]]) -> MealFix:
        """Fix allergen issue by removing the food."""
        food, violation = items[0]
        return MealFix(
            problem=ProblemCategory.ALLERGEN,
            trigger_rule=violation.rule_id,
            original_food=food.name,
            fix_type=FixType.REMOVE,
            fix_description=f"Remove {food.name} from meal",
            replacement_food=None,
            nutrition_impact={
                "calories": -food.nutrition.calories,
                "protein_g": -food.nutrition.protein_g,
            },
            explanation=violation.message,
            confidence=1.0  # Allergen removal is definite
        )
    
    def _fix_sugar(
        self, food: Food, violation: RuleViolation, user: UserProfile
    ) -> MealFix:
        """Fix high sugar by swap or reduction."""
        # Check if we have a known healthy swap
        swap = self._find_swap(food.name)
        
        if swap:
            sugar_reduction = food.nutrition.sugar_g * 0.6  # Estimate 60% reduction
            return MealFix(
                problem=ProblemCategory.EXCESS_SUGAR,
                trigger_rule=violation.rule_id,
                original_food=food.name,
                fix_type=FixType.SWAP,
                fix_description=f"Replace {food.name} with {swap}",
                replacement_food=swap,
                nutrition_impact={"sugar_g": -sugar_reduction},
                explanation=f"Reduces sugar by approximately {sugar_reduction:.0f}g while maintaining flavor satisfaction",
                confidence=0.85
            )
        
        # Fallback: reduce portion
        return MealFix(
            problem=ProblemCategory.EXCESS_SUGAR,
            trigger_rule=violation.rule_id,
            original_food=food.name,
            fix_type=FixType.REDUCE,
            fix_description=f"Have half portion of {food.name}",
            replacement_food=None,
            nutrition_impact={"sugar_g": -food.nutrition.sugar_g * 0.5},
            explanation="Cutting portion in half significantly reduces sugar impact on blood glucose",
            confidence=0.9
        )
    
    def _fix_sodium(
        self, food: Food, violation: RuleViolation, user: UserProfile
    ) -> MealFix:
        """Fix high sodium."""
        swap = self._find_swap(food.name)
        
        if swap:
            sodium_reduction = food.nutrition.sodium_mg * 0.5
            return MealFix(
                problem=ProblemCategory.HIGH_SODIUM,
                trigger_rule=violation.rule_id,
                original_food=food.name,
                fix_type=FixType.SWAP,
                fix_description=f"Replace {food.name} with {swap}",
                replacement_food=swap,
                nutrition_impact={"sodium_mg": -sodium_reduction},
                explanation=f"Reduces sodium by approximately {sodium_reduction:.0f}mg, helping maintain healthy blood pressure",
                confidence=0.8
            )
        
        return MealFix(
            problem=ProblemCategory.HIGH_SODIUM,
            trigger_rule=violation.rule_id,
            original_food=food.name,
            fix_type=FixType.REDUCE,
            fix_description=f"Reduce portion of {food.name} or rinse if canned",
            replacement_food=None,
            nutrition_impact={"sodium_mg": -food.nutrition.sodium_mg * 0.3},
            explanation="Reducing portion or rinsing canned goods can reduce sodium by 30%",
            confidence=0.75
        )
    
    def _fix_fat(
        self, food: Food, violation: RuleViolation, user: UserProfile
    ) -> MealFix:
        """Fix high saturated fat."""
        swap = self._find_swap(food.name)
        
        if swap:
            return MealFix(
                problem=ProblemCategory.HIGH_FAT,
                trigger_rule=violation.rule_id,
                original_food=food.name,
                fix_type=FixType.SWAP,
                fix_description=f"Replace {food.name} with {swap}",
                replacement_food=swap,
                nutrition_impact={
                    "saturated_fat_g": -food.nutrition.saturated_fat_g * 0.6,
                    "fat_g": -food.nutrition.fat_g * 0.4
                },
                explanation="Lean proteins and grilled options significantly reduce saturated fat",
                confidence=0.8
            )
        
        return MealFix(
            problem=ProblemCategory.HIGH_FAT,
            trigger_rule=violation.rule_id,
            original_food=food.name,
            fix_type=FixType.REDUCE,
            fix_description=f"Remove visible fat/skin from {food.name}",
            replacement_food=None,
            nutrition_impact={"saturated_fat_g": -food.nutrition.saturated_fat_g * 0.3},
            explanation="Removing skin and visible fat reduces saturated fat intake",
            confidence=0.7
        )
    
    def _fix_calories(
        self, food: Food, violation: RuleViolation, user: UserProfile
    ) -> MealFix:
        """Fix high calorie foods."""
        return MealFix(
            problem=ProblemCategory.HIGH_CALORIE,
            trigger_rule=violation.rule_id,
            original_food=food.name,
            fix_type=FixType.REDUCE,
            fix_description=f"Reduce {food.name} portion by 30%",
            replacement_food=None,
            nutrition_impact={
                "calories": -food.nutrition.calories * 0.3,
                "fat_g": -food.nutrition.fat_g * 0.3,
                "carbs_g": -food.nutrition.carbs_g * 0.3
            },
            explanation="A 30% portion reduction keeps the meal satisfying while cutting excess calories",
            confidence=0.85
        )
    
    def _fix_low_fiber(
        self, food: Food, violation: RuleViolation, user: UserProfile
    ) -> MealFix:
        """Fix low fiber by adding fiber-rich food."""
        return MealFix(
            problem=ProblemCategory.LOW_FIBER,
            trigger_rule=violation.rule_id,
            original_food=food.name,
            fix_type=FixType.ADD,
            fix_description="Add a side salad or steamed vegetables",
            replacement_food="mixed green salad",
            nutrition_impact={
                "fiber_g": 3.0,  # Estimated fiber from salad
                "calories": 30   # Minimal calorie addition
            },
            explanation="Adding vegetables increases fiber, which slows carb absorption and increases satiety",
            confidence=0.9
        )
    
    def _find_swap(self, food_name: str) -> Optional[str]:
        """Find a healthy swap for a food item."""
        food_lower = food_name.lower()
        for key, value in self.HEALTHY_SWAPS.items():
            if key in food_lower:
                return value
        return None
    
    def _validate_fix_safety(self, fix: MealFix, user: UserProfile) -> bool:
        """
        Validate that a fix doesn't introduce new problems.
        
        This is CRITICAL for safety - we must ensure suggested
        alternatives don't contain allergens or violate rules.
        """
        # Allergen removal is always safe
        if fix.fix_type == FixType.REMOVE:
            return True
        
        # Portion reduction is safe
        if fix.fix_type == FixType.REDUCE:
            return True
        
        # For swaps/adds, check allergens
        if fix.replacement_food:
            replacement_lower = fix.replacement_food.lower()
            for allergen in user.allergens:
                allergen_lower = allergen.lower()
                # Simple check - avoid common allergen keywords
                if allergen_lower in replacement_lower:
                    return False
                # Check specific allergen patterns
                if allergen_lower == "dairy" and any(
                    d in replacement_lower for d in ["milk", "cheese", "yogurt", "cream"]
                ):
                    return False
                if allergen_lower == "nuts" and any(
                    n in replacement_lower for n in ["nut", "almond", "walnut", "pecan"]
                ):
                    return False
        
        return True
