"""
OCR Error Handler Module

Handles common OCR errors, noise, and validation of extracted values.
"""

import re
from typing import Tuple, List, Optional
from .parser import ParsedNutrition


class OCRErrorHandler:
    """
    Handles common OCR errors and validates extracted nutrition values.
    
    Common OCR mistakes:
    - 0 ↔ O, o
    - 1 ↔ l, I, |
    - 5 ↔ S
    - 8 ↔ B
    - g ↔ 9
    """
    
    # Character substitution map for common OCR errors
    CHAR_CORRECTIONS = {
        "O": "0",
        "o": "0", 
        "Q": "0",
        "l": "1",
        "I": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
    }
    
    # Reasonable ranges for nutrition values per serving
    VALID_RANGES = {
        "calories": (0, 2000),
        "carbs_g": (0, 200),
        "protein_g": (0, 100),
        "fat_g": (0, 150),
        "sugar_g": (0, 100),
        "fiber_g": (0, 50),
        "sodium_mg": (0, 5000),
        "saturated_fat_g": (0, 50),
        "cholesterol_mg": (0, 500),
        "potassium_mg": (0, 3000),
    }
    
    @classmethod
    def clean_numeric_text(cls, text: str) -> str:
        """
        Clean OCR text by fixing common character recognition errors.
        Only applies to text that should be numeric.
        
        Args:
            text: Raw text that should contain numbers
            
        Returns:
            Cleaned text with character substitutions
        """
        result = []
        for char in text:
            if char in cls.CHAR_CORRECTIONS:
                result.append(cls.CHAR_CORRECTIONS[char])
            else:
                result.append(char)
        return "".join(result)
    
    @classmethod
    def preprocess_ocr_text(cls, raw_text: str) -> str:
        """
        Preprocess OCR text before parsing.
        
        Steps:
        1. Normalize whitespace
        2. Fix common OCR artifacts
        3. Standardize units
        """
        text = raw_text
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR artifacts
        text = text.replace('|', 'l')
        text = re.sub(r'(\d)[,.](\d{3})', r'\1\2', text)  # Remove thousand separators
        
        # Standardize units
        text = re.sub(r'grams?', 'g', text, flags=re.IGNORECASE)
        text = re.sub(r'milligrams?', 'mg', text, flags=re.IGNORECASE)
        text = re.sub(r'kilocalories?', 'kcal', text, flags=re.IGNORECASE)
        
        return text
    
    @classmethod
    def validate_nutrition(
        cls, nutrition: ParsedNutrition
    ) -> Tuple[bool, List[str]]:
        """
        Validate extracted nutrition values are reasonable.
        
        Args:
            nutrition: Parsed nutrition values
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check each field against valid ranges
        for field, (min_val, max_val) in cls.VALID_RANGES.items():
            value = getattr(nutrition, field, None)
            if value is not None:
                if value < min_val:
                    errors.append(f"{field} is negative ({value}), likely OCR error")
                elif value > max_val:
                    errors.append(f"{field} exceeds maximum ({value} > {max_val})")
        
        # Sugar cannot exceed carbs
        if (nutrition.sugar_g is not None and 
            nutrition.carbs_g is not None and 
            nutrition.sugar_g > nutrition.carbs_g + 0.5):  # 0.5 tolerance
            errors.append("Sugar exceeds total carbs - likely OCR error")
        
        # Saturated fat cannot exceed total fat
        if (nutrition.saturated_fat_g is not None and 
            nutrition.fat_g is not None and 
            nutrition.saturated_fat_g > nutrition.fat_g + 0.5):
            errors.append("Saturated fat exceeds total fat - likely OCR error")
        
        # Calorie cross-check (macros should roughly match calories)
        if cls._can_validate_calories(nutrition):
            calc_cal = cls._calculate_calories_from_macros(nutrition)
            if nutrition.calories is not None:
                diff = abs(nutrition.calories - calc_cal)
                if diff > nutrition.calories * 0.25:  # 25% tolerance
                    errors.append(
                        f"Calorie mismatch: stated {nutrition.calories}, "
                        f"calculated {calc_cal:.0f} from macros"
                    )
        
        return len(errors) == 0, errors
    
    @classmethod
    def _can_validate_calories(cls, nutrition: ParsedNutrition) -> bool:
        """Check if we have enough macros to validate calories."""
        return (nutrition.carbs_g is not None and 
                nutrition.protein_g is not None and 
                nutrition.fat_g is not None)
    
    @classmethod
    def _calculate_calories_from_macros(cls, nutrition: ParsedNutrition) -> float:
        """Calculate expected calories from macronutrients."""
        cal = 0
        if nutrition.carbs_g:
            cal += nutrition.carbs_g * 4
        if nutrition.protein_g:
            cal += nutrition.protein_g * 4
        if nutrition.fat_g:
            cal += nutrition.fat_g * 9
        return cal
    
    @classmethod
    def attempt_fix(cls, nutrition: ParsedNutrition) -> ParsedNutrition:
        """
        Attempt to fix common OCR errors in nutrition values.
        
        This is a best-effort correction - may not fix all issues.
        """
        # If calories is None but we have macros, calculate it
        if nutrition.calories is None and cls._can_validate_calories(nutrition):
            nutrition.calories = cls._calculate_calories_from_macros(nutrition)
        
        # If sugar > carbs, swap them (common OCR line-order error)
        if (nutrition.sugar_g is not None and 
            nutrition.carbs_g is not None and 
            nutrition.sugar_g > nutrition.carbs_g):
            # Check if swapping makes sense
            if nutrition.sugar_g <= 100:  # Reasonable for carbs
                nutrition.sugar_g, nutrition.carbs_g = (
                    nutrition.carbs_g, nutrition.sugar_g
                )
        
        return nutrition
    
    @classmethod
    def get_confidence_adjustment(cls, nutrition: ParsedNutrition) -> float:
        """
        Calculate confidence adjustment based on data quality.
        
        Returns multiplier between 0.5 and 1.0.
        """
        is_valid, errors = cls.validate_nutrition(nutrition)
        
        if is_valid:
            # Adjust based on completeness
            return 0.8 + (nutrition.completeness_score() * 0.2)
        else:
            # Reduce confidence based on error count
            return max(0.5, 1.0 - (len(errors) * 0.15))
