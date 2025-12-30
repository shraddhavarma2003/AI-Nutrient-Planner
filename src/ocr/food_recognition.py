"""
Food Recognition Service

Analyzes food images to:
1. Extract nutrition labels via OCR (Tesseract)
2. Classify food type from image
3. Look up nutrition from database
"""

import os
import re
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

# Try to import pytesseract (may not be installed)
try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("Warning: pytesseract not installed. Install with: pip install pytesseract pillow")
    print("Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")

from ocr.parser import NutritionParser, ParsedNutrition


@dataclass
class FoodRecognitionResult:
    """Result of food image analysis."""
    food_name: str
    nutrition: Dict[str, float]
    source: str  # "ocr", "database", "estimated"
    confidence: float
    raw_text: Optional[str] = None


class FoodRecognitionService:
    """
    Analyzes food images to extract nutrition information.
    
    Two main paths:
    1. Nutrition label detected → OCR extraction
    2. No label → Food classification + database lookup
    """
    
    # Common food patterns for classification
    FOOD_PATTERNS = {
        "pizza": ["pizza", "pepperoni", "cheese", "crust", "slice"],
        "burger": ["burger", "hamburger", "patty", "bun", "beef"],
        "salad": ["salad", "lettuce", "greens", "vegetable", "healthy"],
        "pasta": ["pasta", "spaghetti", "noodle", "macaroni", "penne"],
        "rice": ["rice", "grain", "fried rice", "biryani"],
        "chicken": ["chicken", "poultry", "grilled", "fried chicken"],
        "sandwich": ["sandwich", "bread", "sub", "wrap"],
        "apple": ["apple", "fruit", "red", "green apple"],
        "banana": ["banana", "yellow fruit"],
        "egg": ["egg", "eggs", "omelette", "scrambled"],
    }
    
    # Nutrition database (USDA-based values per 100g)
    NUTRITION_DB = {
        "pizza": {"calories": 266, "protein_g": 11, "carbs_g": 33, "fat_g": 10, "sugar_g": 4, "sodium_mg": 640},
        "burger": {"calories": 295, "protein_g": 17, "carbs_g": 24, "fat_g": 14, "sugar_g": 5, "sodium_mg": 500},
        "salad": {"calories": 65, "protein_g": 3, "carbs_g": 11, "fat_g": 1, "sugar_g": 4, "sodium_mg": 120},
        "pasta": {"calories": 131, "protein_g": 5, "carbs_g": 25, "fat_g": 1, "sugar_g": 1, "sodium_mg": 1},
        "rice": {"calories": 130, "protein_g": 2.7, "carbs_g": 28, "fat_g": 0.3, "sugar_g": 0, "sodium_mg": 1},
        "chicken": {"calories": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6, "sugar_g": 0, "sodium_mg": 74},
        "sandwich": {"calories": 252, "protein_g": 10, "carbs_g": 30, "fat_g": 10, "sugar_g": 4, "sodium_mg": 480},
        "apple": {"calories": 52, "protein_g": 0.3, "carbs_g": 14, "fat_g": 0.2, "sugar_g": 10, "sodium_mg": 1},
        "banana": {"calories": 89, "protein_g": 1.1, "carbs_g": 23, "fat_g": 0.3, "sugar_g": 12, "sodium_mg": 1},
        "egg": {"calories": 155, "protein_g": 13, "carbs_g": 1.1, "fat_g": 11, "sugar_g": 1.1, "sodium_mg": 124},
        "mixed_meal": {"calories": 200, "protein_g": 10, "carbs_g": 25, "fat_g": 8, "sugar_g": 5, "sodium_mg": 350},
    }
    
    def __init__(self):
        self.parser = NutritionParser()
    
    def analyze_image(self, image_path: str) -> FoodRecognitionResult:
        """
        Analyze a food image and return nutrition information.
        
        Priority:
        1. YOLO model classification (best.pt)
        2. OCR for nutrition labels (if Tesseract available)
        3. Pattern-based fallback
        
        Args:
            image_path: Path to the image file
            
        Returns:
            FoodRecognitionResult with nutrition data
        """
        raw_text = ""
        
        # Step 1: Try YOLO model classification (PRIMARY METHOD)
        try:
            from services.yolo_service import get_yolo_recognizer
            yolo = get_yolo_recognizer()
            
            if yolo.is_available:
                result = yolo.predict(image_path)
                
                if result.get("success") and result.get("food_name"):
                    food_name = result["food_name"]
                    confidence = result.get("confidence", 0.8)
                    print(f"[FoodRecognition] ✓ YOLO classified: {food_name} ({confidence:.0%})")
                    
                    # Look up nutrition from database
                    nutrition = self.NUTRITION_DB.get(
                        food_name.lower(), 
                        self.NUTRITION_DB["mixed_meal"]
                    )
                    
                    return FoodRecognitionResult(
                        food_name=food_name.title(),
                        nutrition=nutrition,
                        source="yolo_model",
                        confidence=confidence,
                        raw_text=None,
                    )
                else:
                    print(f"[FoodRecognition] YOLO: No food detected")
            else:
                print("[FoodRecognition] YOLO model not available")
        except ImportError as e:
            print(f"[FoodRecognition] ✗ YOLO not available: {e}")
        except Exception as e:
            print(f"[FoodRecognition] ✗ YOLO classification failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 2: Try OCR if Tesseract is available (for nutrition labels)
        if HAS_TESSERACT:
            raw_text = self._extract_text(image_path)
            
            # Check if this is a nutrition label
            if raw_text and self.parser.has_nutrition_panel(raw_text):
                parsed = self.parser.parse(raw_text)
                
                if parsed.is_complete():
                    # Get food name from label
                    food_name = self.parser.extract_food_name(raw_text) or "Packaged Food"
                    
                    return FoodRecognitionResult(
                        food_name=food_name,
                        nutrition=parsed.to_dict(),
                        source="ocr",
                        confidence=parsed.completeness_score(),
                        raw_text=raw_text,
                    )
        
        # Step 3: Fallback to pattern classification
        food_type = self._classify_food(raw_text, image_path)
        nutrition = self.NUTRITION_DB.get(food_type, self.NUTRITION_DB["mixed_meal"])
        
        return FoodRecognitionResult(
            food_name=food_type.title(),
            nutrition=nutrition,
            source="database" if food_type != "mixed_meal" else "estimated",
            confidence=0.7 if food_type != "mixed_meal" else 0.3,
            raw_text=raw_text if raw_text else None,
        )
    
    def _extract_text(self, image_path: str) -> str:
        """Extract text from image using Tesseract OCR."""
        try:
            image = Image.open(image_path)
            # Use Tesseract with custom config for better nutrition label parsing
            config = '--psm 6'  # Assume uniform block of text
            text = pytesseract.image_to_string(image, config=config)
            return text
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def _classify_food(self, text: str, image_path: str) -> str:
        """
        Classify food type from OCR text and image.
        
        Uses:
        1. Text patterns from OCR
        2. Filename hints
        """
        text_lower = text.lower() if text else ""
        filename_lower = os.path.basename(image_path).lower()
        combined = text_lower + " " + filename_lower
        
        # Score each food type
        best_match = "mixed_meal"
        best_score = 0
        
        for food, patterns in self.FOOD_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in combined)
            if score > best_score:
                best_score = score
                best_match = food
        
        return best_match
    
    def get_nutrition_by_name(self, food_name: str) -> Dict[str, float]:
        """Look up nutrition by food name."""
        name_lower = food_name.lower()
        
        for food, nutrition in self.NUTRITION_DB.items():
            if food in name_lower:
                return nutrition
        
        return self.NUTRITION_DB["mixed_meal"]


# Singleton instance
food_recognition_service = FoodRecognitionService()


def parse_nutrition_label(image_path: str) -> Dict[str, Any]:
    """
    Parse nutrition from image.
    
    This is the main entry point called by the upload endpoint.
    """
    result = food_recognition_service.analyze_image(image_path)
    
    return {
        "food_name": result.food_name,
        "nutrition": result.nutrition,
        "source": result.source,
        "confidence": result.confidence,
        "raw_text": result.raw_text,
    }
