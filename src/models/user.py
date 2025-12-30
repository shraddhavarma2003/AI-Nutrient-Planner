"""
User Profile Models

Defines user health profiles including medical conditions, allergens,
and daily nutritional targets. This data drives the rule engine.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class HealthCondition(Enum):
    """
    Supported health conditions that affect nutritional rules.
    Each condition activates a specific set of rules in the engine.
    """
    DIABETES = "diabetes"
    HYPERTENSION = "hypertension"
    OBESITY = "obesity"
    # Future conditions can be added here
    # KIDNEY_DISEASE = "kidney_disease"
    # HEART_DISEASE = "heart_disease"


class ActivityLevel(Enum):
    """Activity level for calorie target calculation."""
    SEDENTARY = "sedentary"          # Little or no exercise
    LIGHTLY_ACTIVE = "lightly_active"  # Light exercise 1-3 days/week
    MODERATELY_ACTIVE = "moderately_active"  # Moderate exercise 3-5 days/week
    VERY_ACTIVE = "very_active"        # Hard exercise 6-7 days/week
    EXTRA_ACTIVE = "extra_active"      # Very hard exercise, physical job


@dataclass
class DailyTargets:
    """
    Daily nutritional targets for a user.
    These are used to check cumulative intake across meals.
    """
    calories: int = 2000
    protein_g: int = 50
    carbs_g: int = 250
    fat_g: int = 65
    fiber_g: int = 25
    sugar_g: int = 50  # WHO recommends <50g, ideally <25g
    sodium_mg: int = 2300  # General population
    potassium_mg: int = 3500
    
    @classmethod
    def for_diabetes(cls, base_calories: int = 2000) -> "DailyTargets":
        """Targets adjusted for diabetes management."""
        return cls(
            calories=base_calories,
            protein_g=int(base_calories * 0.2 / 4),  # 20% from protein
            carbs_g=int(base_calories * 0.45 / 4),   # 45% from carbs (lower)
            fat_g=int(base_calories * 0.35 / 9),     # 35% from fat
            fiber_g=30,  # Higher fiber for blood sugar control
            sugar_g=25,  # Stricter sugar limit
            sodium_mg=2300,
            potassium_mg=3500,
        )
    
    @classmethod
    def for_hypertension(cls, base_calories: int = 2000) -> "DailyTargets":
        """Targets adjusted for hypertension (DASH diet inspired)."""
        return cls(
            calories=base_calories,
            protein_g=int(base_calories * 0.18 / 4),
            carbs_g=int(base_calories * 0.55 / 4),
            fat_g=int(base_calories * 0.27 / 9),
            fiber_g=30,
            sugar_g=50,
            sodium_mg=1500,  # Strict sodium limit
            potassium_mg=4700,  # Higher potassium
        )
    
    @classmethod
    def for_weight_loss(cls, base_calories: int = 1500) -> "DailyTargets":
        """Targets for weight loss (calorie deficit)."""
        return cls(
            calories=base_calories,
            protein_g=int(base_calories * 0.3 / 4),  # Higher protein for satiety
            carbs_g=int(base_calories * 0.4 / 4),
            fat_g=int(base_calories * 0.3 / 9),
            fiber_g=30,  # High fiber for satiety
            sugar_g=25,
            sodium_mg=2300,
            potassium_mg=3500,
        )


@dataclass
class DailyIntake:
    """
    Tracks cumulative daily intake.
    Updated as user logs meals throughout the day.
    """
    calories: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    sugar_g: float = 0.0
    sodium_mg: float = 0.0
    potassium_mg: float = 0.0
    
    def add_food(self, nutrition: "NutritionInfo", servings: float = 1.0):
        """Add a food's nutrition to daily totals."""
        from .food import NutritionInfo  # Avoid circular import
        self.calories += nutrition.calories * servings
        self.protein_g += nutrition.protein_g * servings
        self.carbs_g += nutrition.carbs_g * servings
        self.fat_g += nutrition.fat_g * servings
        self.fiber_g += nutrition.fiber_g * servings
        self.sugar_g += nutrition.sugar_g * servings
        self.sodium_mg += nutrition.sodium_mg * servings
        self.potassium_mg += nutrition.potassium_mg * servings
    
    def to_dict(self) -> dict:
        """Convert to dictionary for rule engine."""
        return {
            "calories": self.calories,
            "protein_g": self.protein_g,
            "carbs_g": self.carbs_g,
            "fat_g": self.fat_g,
            "fiber_g": self.fiber_g,
            "sugar_g": self.sugar_g,
            "sodium_mg": self.sodium_mg,
            "potassium_mg": self.potassium_mg,
        }


@dataclass
class UserProfile:
    """
    Complete user profile for personalized nutrition guidance.
    
    This profile drives the rule engine:
    - conditions: which rule categories to apply
    - allergens: foods to block
    - targets: limits for cumulative daily checks
    """
    # User identification
    user_id: str
    name: str
    
    # Health conditions (determine which rules apply)
    conditions: List[HealthCondition] = field(default_factory=list)
    
    # Allergens (trigger BLOCK action)
    allergens: List[str] = field(default_factory=list)
    
    # Daily nutritional targets
    daily_targets: DailyTargets = field(default_factory=DailyTargets)
    
    # Physical attributes (for calorie calculation)
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    activity_level: ActivityLevel = ActivityLevel.MODERATELY_ACTIVE
    
    def has_condition(self, condition: HealthCondition) -> bool:
        """Check if user has a specific health condition."""
        return condition in self.conditions
    
    def has_allergen(self, allergen: str) -> bool:
        """Check if user has listed a specific allergen."""
        return allergen.lower() in [a.lower() for a in self.allergens]
    
    def calculate_bmr(self) -> Optional[float]:
        """
        Calculate Basal Metabolic Rate using Mifflin-St Jeor equation.
        Returns None if required attributes are missing.
        """
        if not all([self.age, self.weight_kg, self.height_cm]):
            return None
        
        # Mifflin-St Jeor (using average of male/female since gender not stored)
        # Male: 10*weight + 6.25*height - 5*age + 5
        # Female: 10*weight + 6.25*height - 5*age - 161
        # Average: 10*weight + 6.25*height - 5*age - 78
        bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age - 78
        return bmr
    
    def calculate_tdee(self) -> Optional[float]:
        """
        Calculate Total Daily Energy Expenditure.
        TDEE = BMR * Activity Multiplier
        """
        bmr = self.calculate_bmr()
        if bmr is None:
            return None
        
        multipliers = {
            ActivityLevel.SEDENTARY: 1.2,
            ActivityLevel.LIGHTLY_ACTIVE: 1.375,
            ActivityLevel.MODERATELY_ACTIVE: 1.55,
            ActivityLevel.VERY_ACTIVE: 1.725,
            ActivityLevel.EXTRA_ACTIVE: 1.9,
        }
        
        return bmr * multipliers[self.activity_level]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "conditions": [c.value for c in self.conditions],
            "allergens": self.allergens,
            "age": self.age,
            "weight_kg": self.weight_kg,
            "height_cm": self.height_cm,
            "activity_level": self.activity_level.value,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        """Create UserProfile from dictionary."""
        conditions = [HealthCondition(c) for c in data.get("conditions", [])]
        
        # Set appropriate daily targets based on conditions
        if HealthCondition.DIABETES in conditions:
            targets = DailyTargets.for_diabetes()
        elif HealthCondition.HYPERTENSION in conditions:
            targets = DailyTargets.for_hypertension()
        elif HealthCondition.OBESITY in conditions:
            targets = DailyTargets.for_weight_loss()
        else:
            targets = DailyTargets()
        
        return cls(
            user_id=data["user_id"],
            name=data["name"],
            conditions=conditions,
            allergens=data.get("allergens", []),
            daily_targets=targets,
            age=data.get("age"),
            weight_kg=data.get("weight_kg"),
            height_cm=data.get("height_cm"),
            activity_level=ActivityLevel(data.get("activity_level", "moderately_active")),
        )
