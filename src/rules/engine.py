"""
Medical Rule Engine - Core Safety Layer

This is the DETERMINISTIC safety layer that evaluates foods against user health profiles.
Rules ALWAYS override AI suggestions. This engine ensures:
- Allergens are blocked
- Condition-specific warnings are issued
- Daily limits are enforced

IMPORTANT: This is NOT medical advice. This is a support tool.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

from models.food import Food
from models.user import UserProfile, HealthCondition, DailyIntake


class Severity(Enum):
    """
    Rule violation severity levels.
    Evaluated in priority order: BLOCK > ALERT > WARN > ALLOW
    """
    ALLOW = "allow"    # No issues found
    WARN = "warn"      # Caution advised, but can proceed
    ALERT = "alert"    # Significant concern, user should reconsider
    BLOCK = "block"    # Do not consume - safety risk (allergens)


@dataclass
class RuleViolation:
    """
    Represents a single rule violation.
    Contains enough information to explain the issue to the user.
    """
    rule_id: str              # e.g., "DM-001", "AL-001"
    category: str             # "diabetes", "hypertension", "allergy", "obesity"
    severity: Severity
    message: str              # Human-readable explanation
    suggestion: Optional[str] = None  # Actionable advice
    
    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "severity": self.severity.value,
            "message": self.message,
            "suggestion": self.suggestion,
        }


class RuleEngine:
    """
    Deterministic rule engine for medical safety.
    
    DESIGN PRINCIPLES:
    1. Rules are checked in priority order (allergens first)
    2. All applicable rules are evaluated (not short-circuited)
    3. The engine is stateless - results depend only on inputs
    4. Rules are conservative - when in doubt, warn
    
    USAGE:
        engine = RuleEngine()
        violations = engine.evaluate(food, user_profile, daily_intake)
        verdict = engine.get_final_verdict(violations)
    """
    
    # =========================================================================
    # RULE THRESHOLDS (configurable)
    # =========================================================================
    
    # Diabetes thresholds
    DIABETES_SUGAR_WARN = 15.0       # g per serving
    DIABETES_SUGAR_DAILY = 25.0      # g per day
    DIABETES_GI_WARN = 70            # glycemic index
    DIABETES_LOW_FIBER_CARB_RATIO = 3.0  # fiber_g threshold when carbs > 30g
    
    # Hypertension thresholds
    HYPERTENSION_SODIUM_WARN = 600.0    # mg per serving
    HYPERTENSION_SODIUM_ALERT = 1000.0  # mg per serving
    HYPERTENSION_SODIUM_DAILY = 1500.0  # mg per day
    HYPERTENSION_SODIUM_POTASSIUM_RATIO = 2.0  # sodium/potassium ratio
    
    # Obesity thresholds
    OBESITY_CALORIE_DENSITY = 4.0    # kcal per gram
    OBESITY_SATURATED_FAT = 5.0      # g per serving
    OBESITY_LOW_FIBER = 1.0          # g per serving
    
    def evaluate(
        self,
        food: Food,
        user: UserProfile,
        daily_intake: Optional[DailyIntake] = None
    ) -> List[RuleViolation]:
        """
        Evaluate a food item against user's health profile.
        
        Args:
            food: The food item to evaluate
            user: User's health profile
            daily_intake: Current day's cumulative intake (optional)
        
        Returns:
            List of rule violations (empty if food is safe)
        """
        violations: List[RuleViolation] = []
        daily = daily_intake.to_dict() if daily_intake else {}
        
        # ─────────────────────────────────────────────
        # PRIORITY 1: Allergy rules (can BLOCK)
        # Always check allergens regardless of conditions
        # ─────────────────────────────────────────────
        if user.allergens:
            violations.extend(self._check_allergy_rules(food, user))
        
        # ─────────────────────────────────────────────
        # PRIORITY 2: Condition-specific rules
        # ─────────────────────────────────────────────
        if user.has_condition(HealthCondition.DIABETES):
            violations.extend(self._check_diabetes_rules(food, user, daily))
        
        if user.has_condition(HealthCondition.HYPERTENSION):
            violations.extend(self._check_hypertension_rules(food, user, daily))
        
        if user.has_condition(HealthCondition.OBESITY):
            violations.extend(self._check_obesity_rules(food, user, daily))
        
        return violations
    
    def get_final_verdict(self, violations: List[RuleViolation]) -> Severity:
        """
        Get the highest severity from all violations.
        This determines the overall action for the food.
        """
        if not violations:
            return Severity.ALLOW
        
        # Priority order
        severities = [v.severity for v in violations]
        if Severity.BLOCK in severities:
            return Severity.BLOCK
        if Severity.ALERT in severities:
            return Severity.ALERT
        if Severity.WARN in severities:
            return Severity.WARN
        return Severity.ALLOW
    
    def format_violations(self, violations: List[RuleViolation]) -> str:
        """Format violations as human-readable text."""
        if not violations:
            return "✅ No concerns found."
        
        lines = []
        for v in violations:
            icon = {
                Severity.BLOCK: "🚫",
                Severity.ALERT: "🛑",
                Severity.WARN: "⚠️",
                Severity.ALLOW: "✅",
            }[v.severity]
            
            lines.append(f"{icon} [{v.rule_id}] {v.message}")
            if v.suggestion:
                lines.append(f"   → {v.suggestion}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # ALLERGY RULES
    # =========================================================================
    
    def _check_allergy_rules(
        self, food: Food, user: UserProfile
    ) -> List[RuleViolation]:
        """
        Check for allergen matches.
        Allergen detection results in BLOCK - the highest severity.
        """
        violations = []
        
        for allergen in user.allergens:
            if food.contains_allergen(allergen):
                violations.append(RuleViolation(
                    rule_id="AL-001",
                    category="allergy",
                    severity=Severity.BLOCK,
                    message=f"Contains {allergen} - listed in your allergens",
                    suggestion=f"Search for {allergen}-free alternatives"
                ))
        
        # If allergen data is missing (None, not empty list), warn about uncertainty
        if food.allergens is None and user.allergens:
            violations.append(RuleViolation(
                rule_id="AL-003",
                category="allergy",
                severity=Severity.WARN,
                message="Allergen information not available for this food",
                suggestion="Check packaging or ask about ingredients before consuming"
            ))
        
        return violations
    
    # =========================================================================
    # DIABETES RULES
    # =========================================================================
    
    def _check_diabetes_rules(
        self, food: Food, user: UserProfile, daily: Dict[str, Any]
    ) -> List[RuleViolation]:
        """
        Rules for diabetes management:
        - Monitor sugar per serving
        - Check glycemic index
        - Track daily sugar intake
        - Consider fiber-to-carb ratio
        """
        violations = []
        n = food.nutrition  # Shorthand
        
        # DM-001: High sugar per serving
        if n.sugar_g > self.DIABETES_SUGAR_WARN:
            violations.append(RuleViolation(
                rule_id="DM-001",
                category="diabetes",
                severity=Severity.WARN,
                message=f"High sugar content: {n.sugar_g:.1f}g per serving",
                suggestion="Consider a smaller portion or lower-sugar alternative"
            ))
        
        # DM-002: High glycemic index
        if n.glycemic_index and n.glycemic_index > self.DIABETES_GI_WARN:
            violations.append(RuleViolation(
                rule_id="DM-002",
                category="diabetes",
                severity=Severity.WARN,
                message=f"High glycemic index: {n.glycemic_index}",
                suggestion="Pair with protein or fiber to slow glucose absorption"
            ))
        
        # DM-003: Low fiber with high carbs (blood sugar spike risk)
        if n.carbs_g > 30 and n.fiber_g < self.DIABETES_LOW_FIBER_CARB_RATIO:
            violations.append(RuleViolation(
                rule_id="DM-003",
                category="diabetes",
                severity=Severity.WARN,
                message=f"High carbs ({n.carbs_g:.1f}g) with low fiber ({n.fiber_g:.1f}g)",
                suggestion="Add a fiber-rich side or choose whole grain version"
            ))
        
        # DM-004: Daily sugar limit exceeded
        current_sugar = daily.get("sugar_g", 0)
        projected_sugar = current_sugar + n.sugar_g
        if projected_sugar > self.DIABETES_SUGAR_DAILY:
            violations.append(RuleViolation(
                rule_id="DM-004",
                category="diabetes",
                severity=Severity.ALERT,
                message=f"Would exceed daily sugar limit: {projected_sugar:.1f}g / {self.DIABETES_SUGAR_DAILY}g",
                suggestion="Consider skipping dessert or choosing sugar-free options"
            ))
        
        return violations
    
    # =========================================================================
    # HYPERTENSION RULES
    # =========================================================================
    
    def _check_hypertension_rules(
        self, food: Food, user: UserProfile, daily: Dict[str, Any]
    ) -> List[RuleViolation]:
        """
        Rules for hypertension management:
        - Monitor sodium per serving
        - Track daily sodium intake
        - Consider sodium-to-potassium ratio
        """
        violations = []
        n = food.nutrition
        
        # HT-001 / HT-002: Sodium per serving
        if n.sodium_mg > self.HYPERTENSION_SODIUM_ALERT:
            violations.append(RuleViolation(
                rule_id="HT-002",
                category="hypertension",
                severity=Severity.ALERT,
                message=f"Very high sodium: {n.sodium_mg:.0f}mg per serving",
                suggestion="This exceeds 2/3 of daily recommendation in one serving"
            ))
        elif n.sodium_mg > self.HYPERTENSION_SODIUM_WARN:
            violations.append(RuleViolation(
                rule_id="HT-001",
                category="hypertension",
                severity=Severity.WARN,
                message=f"High sodium: {n.sodium_mg:.0f}mg per serving",
                suggestion="Look for low-sodium or 'no salt added' versions"
            ))
        
        # HT-003: Daily sodium limit exceeded
        current_sodium = daily.get("sodium_mg", 0)
        projected_sodium = current_sodium + n.sodium_mg
        if projected_sodium > self.HYPERTENSION_SODIUM_DAILY:
            violations.append(RuleViolation(
                rule_id="HT-003",
                category="hypertension",
                severity=Severity.ALERT,
                message=f"Would exceed daily sodium limit: {projected_sodium:.0f}mg / {self.HYPERTENSION_SODIUM_DAILY:.0f}mg",
                suggestion="Choose low-sodium foods for your remaining meals today"
            ))
        
        # HT-004: Poor sodium-to-potassium ratio
        if n.potassium_mg > 0:
            ratio = n.sodium_mg / n.potassium_mg
            if ratio > self.HYPERTENSION_SODIUM_POTASSIUM_RATIO:
                violations.append(RuleViolation(
                    rule_id="HT-004",
                    category="hypertension",
                    severity=Severity.WARN,
                    message=f"High sodium-to-potassium ratio: {ratio:.1f}:1",
                    suggestion="Balance with potassium-rich foods like bananas or leafy greens"
                ))
        
        return violations
    
    # =========================================================================
    # OBESITY RULES
    # =========================================================================
    
    def _check_obesity_rules(
        self, food: Food, user: UserProfile, daily: Dict[str, Any]
    ) -> List[RuleViolation]:
        """
        Rules for weight management:
        - Monitor calorie density
        - Track saturated fat
        - Check fiber content
        - Enforce daily calorie target
        """
        violations = []
        n = food.nutrition
        
        # OB-001: High calorie density
        density = food.calorie_density
        if density > self.OBESITY_CALORIE_DENSITY:
            violations.append(RuleViolation(
                rule_id="OB-001",
                category="obesity",
                severity=Severity.WARN,
                message=f"High calorie density: {density:.1f} kcal/g",
                suggestion="Practice portion control or choose lighter alternatives"
            ))
        
        # OB-002: High saturated fat
        if n.saturated_fat_g > self.OBESITY_SATURATED_FAT:
            violations.append(RuleViolation(
                rule_id="OB-002",
                category="obesity",
                severity=Severity.WARN,
                message=f"High saturated fat: {n.saturated_fat_g:.1f}g per serving",
                suggestion="Consider lean protein or plant-based alternatives"
            ))
        
        # OB-003: Daily calorie target exceeded
        current_calories = daily.get("calories", 0)
        projected_calories = current_calories + n.calories
        target = user.daily_targets.calories
        if projected_calories > target:
            violations.append(RuleViolation(
                rule_id="OB-003",
                category="obesity",
                severity=Severity.ALERT,
                message=f"Would exceed daily calorie target: {projected_calories:.0f} / {target} kcal",
                suggestion="Consider a lighter option or increase physical activity"
            ))
        
        # OB-004: Very low fiber (less satiating)
        if n.fiber_g < self.OBESITY_LOW_FIBER:
            violations.append(RuleViolation(
                rule_id="OB-004",
                category="obesity",
                severity=Severity.WARN,
                message=f"Very low fiber: {n.fiber_g:.1f}g per serving",
                suggestion="Add vegetables or choose whole grain options for better satiety"
            ))
        
        return violations
