"""
Food and Nutrition Data Models

These models represent the core nutrition data structure.
Data is sourced from USDA FoodData Central and OpenFoodFacts.
Values are NEVER guessed - they must come from trusted sources.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum


class DataSource(Enum):
    """Source of nutrition data - must be a trusted database."""
    USDA = "usda"
    OPENFOODFACTS = "openfoodfacts"
    USER_VERIFIED = "user_verified"  # User-entered but verified against source


class FoodCategory(Enum):
    """Food categories for classification."""
    FRUIT = "fruit"
    VEGETABLE = "vegetable"
    GRAIN = "grain"
    PROTEIN = "protein"
    DAIRY = "dairy"
    FAT_OIL = "fat_oil"
    BEVERAGE = "beverage"
    SNACK = "snack"
    PREPARED_MEAL = "prepared_meal"
    CONDIMENT = "condiment"
    OTHER = "other"


@dataclass
class NutritionInfo:
    """
    Nutrition information per serving.
    All values are per serving_size in serving_unit.
    """
    # Macronutrients
    calories: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    
    # Carbohydrate details
    sugar_g: float = 0.0
    fiber_g: float = 0.0
    
    # Fat details
    saturated_fat_g: float = 0.0
    trans_fat_g: float = 0.0
    
    # Minerals
    sodium_mg: float = 0.0
    potassium_mg: float = 0.0
    cholesterol_mg: float = 0.0
    
    # Optional - may not be available for all foods
    glycemic_index: Optional[int] = None  # 0-100 scale
    
    def validate(self) -> bool:
        """Validate that nutrition values are reasonable."""
        # All values must be non-negative
        if any(v < 0 for v in [
            self.calories, self.protein_g, self.carbs_g, self.fat_g,
            self.sugar_g, self.fiber_g, self.saturated_fat_g,
            self.sodium_mg, self.potassium_mg, self.cholesterol_mg
        ]):
            return False
        
        # Sugar and fiber cannot exceed total carbs
        if self.sugar_g + self.fiber_g > self.carbs_g + 0.1:  # 0.1 tolerance for rounding
            return False
        
        # Saturated + trans fat cannot exceed total fat
        if self.saturated_fat_g + self.trans_fat_g > self.fat_g + 0.1:
            return False
        
        # GI must be 0-100 if provided
        if self.glycemic_index is not None:
            if not 0 <= self.glycemic_index <= 100:
                return False
        
        return True
    
    @property
    def calorie_density(self) -> float:
        """Calculate calorie density (kcal per gram) - requires serving size."""
        # This is calculated at the Food level since it needs serving_size
        raise NotImplementedError("Use Food.calorie_density instead")


@dataclass
class Food:
    """
    A food item with complete nutrition information.
    
    All nutrition data must come from trusted sources (USDA, OpenFoodFacts).
    The data_source field tracks where the data originated.
    """
    # Identification
    food_id: str
    name: str
    
    # Serving information
    serving_size: float  # Amount in serving_unit
    serving_unit: str    # g, ml, cup, piece, etc.
    
    # Nutrition data
    nutrition: NutritionInfo
    
    # Allergen information (critical for safety)
    allergens: List[str] = field(default_factory=list)
    
    # Classification
    category: FoodCategory = FoodCategory.OTHER
    
    # Optional fields
    brand: Optional[str] = None  # For packaged foods
    
    # Data provenance
    data_source: DataSource = DataSource.USDA
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate the food item after initialization."""
        if self.serving_size <= 0:
            raise ValueError("serving_size must be positive")
        if not self.nutrition.validate():
            raise ValueError("Invalid nutrition values")
    
    @property
    def calorie_density(self) -> float:
        """
        Calculate calorie density in kcal/g.
        Used for obesity rules - high density foods (>4 kcal/g) need warnings.
        """
        if self.serving_unit in ["g", "ml"]:
            return self.nutrition.calories / self.serving_size if self.serving_size > 0 else 0
        # For non-gram units, we can't calculate density accurately
        return 0.0
    
    def contains_allergen(self, allergen: str) -> bool:
        """Check if food contains a specific allergen (case-insensitive)."""
        return allergen.lower() in [a.lower() for a in self.allergens]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "food_id": self.food_id,
            "name": self.name,
            "brand": self.brand,
            "serving_size": self.serving_size,
            "serving_unit": self.serving_unit,
            "calories": self.nutrition.calories,
            "protein_g": self.nutrition.protein_g,
            "carbs_g": self.nutrition.carbs_g,
            "sugar_g": self.nutrition.sugar_g,
            "fiber_g": self.nutrition.fiber_g,
            "fat_g": self.nutrition.fat_g,
            "saturated_fat_g": self.nutrition.saturated_fat_g,
            "sodium_mg": self.nutrition.sodium_mg,
            "potassium_mg": self.nutrition.potassium_mg,
            "cholesterol_mg": self.nutrition.cholesterol_mg,
            "glycemic_index": self.nutrition.glycemic_index,
            "allergens": self.allergens,
            "category": self.category.value,
            "data_source": self.data_source.value,
            "last_updated": self.last_updated.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Food":
        """Create Food instance from dictionary."""
        nutrition = NutritionInfo(
            calories=data.get("calories", 0),
            protein_g=data.get("protein_g", 0),
            carbs_g=data.get("carbs_g", 0),
            sugar_g=data.get("sugar_g", 0),
            fiber_g=data.get("fiber_g", 0),
            fat_g=data.get("fat_g", 0),
            saturated_fat_g=data.get("saturated_fat_g", 0),
            sodium_mg=data.get("sodium_mg", 0),
            potassium_mg=data.get("potassium_mg", 0),
            cholesterol_mg=data.get("cholesterol_mg", 0),
            glycemic_index=data.get("glycemic_index"),
        )
        
        return cls(
            food_id=data["food_id"],
            name=data["name"],
            brand=data.get("brand"),
            serving_size=data["serving_size"],
            serving_unit=data["serving_unit"],
            nutrition=nutrition,
            allergens=data.get("allergens", []),
            category=FoodCategory(data.get("category", "other")),
            data_source=DataSource(data.get("data_source", "usda")),
        )
