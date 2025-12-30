"""
Nutrition OCR Service

Main service for extracting nutrition information from food images.
Supports packaged food labels, bills, and meal images.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
from enum import Enum

from .parser import NutritionParser, ParsedNutrition, NutritionSource
from .error_handler import OCRErrorHandler


@dataclass
class OCRResult:
    """Raw OCR output from the OCR engine."""
    raw_text: str
    confidence: float  # 0.0 to 1.0
    language: str = "en"
    bounding_boxes: List[dict] = field(default_factory=list)


@dataclass
class FoodExtractionResult:
    """
    Complete result of food extraction from image.
    
    Contains either:
    - Nutrition from OCR (source=OCR) - ground truth from label
    - Food name for RAG lookup (source=RAG) - needs database retrieval
    """
    food_name: Optional[str]
    nutrition: ParsedNutrition
    source: NutritionSource
    confidence: float
    warnings: List[str] = field(default_factory=list)
    requires_rag_lookup: bool = False
    
    def to_dict(self) -> dict:
        return {
            "food_name": self.food_name,
            "nutrition": self.nutrition.to_dict(),
            "source": self.source.value,
            "confidence": round(self.confidence, 2),
            "warnings": self.warnings,
            "requires_rag_lookup": self.requires_rag_lookup
        }


class OCREngineType(Enum):
    """Supported OCR backends."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    MOCK = "mock"  # For testing


class NutritionOCRService:
    """
    Main service for extracting nutrition from food images.
    
    Pipeline:
    1. Run OCR on image
    2. Detect if nutrition panel exists
    3. Parse values OR extract food name
    4. Validate and return structured result
    
    Usage:
        service = NutritionOCRService()
        result = service.extract("food_label.jpg")
        
        if result.source == NutritionSource.OCR:
            # Use extracted values directly
            calories = result.nutrition.calories
        else:
            # Need to lookup food_name in database
            nutrition = rag_service.lookup(result.food_name)
    """
    
    def __init__(self, ocr_engine: Any = None, engine_type: OCREngineType = OCREngineType.MOCK):
        """
        Initialize OCR service.
        
        Args:
            ocr_engine: OCR backend instance (Tesseract, EasyOCR, etc.)
            engine_type: Type of OCR engine for configuration
        """
        self.ocr_engine = ocr_engine
        self.engine_type = engine_type
        self.parser = NutritionParser()
    
    def extract(self, image_path: str) -> FoodExtractionResult:
        """
        Extract nutrition information from a food image.
        
        Args:
            image_path: Path to the food image
            
        Returns:
            FoodExtractionResult with nutrition data or food name
        """
        # Step 1: Run OCR
        ocr_result = self._run_ocr(image_path)
        
        # Step 2: Preprocess text
        cleaned_text = OCRErrorHandler.preprocess_ocr_text(ocr_result.raw_text)
        
        # Step 3: Check for nutrition panel
        has_panel = self.parser.has_nutrition_panel(cleaned_text)
        
        if has_panel:
            # Step 4a: Parse nutrition values
            nutrition = self.parser.parse(cleaned_text)
            
            # Step 5: Validate
            is_valid, errors = OCRErrorHandler.validate_nutrition(nutrition)
            
            if not is_valid:
                # Try to fix common errors
                nutrition = OCRErrorHandler.attempt_fix(nutrition)
                is_valid, errors = OCRErrorHandler.validate_nutrition(nutrition)
            
            # Step 6: Check if we have enough data
            if nutrition.is_complete():
                # Adjust confidence based on validation
                confidence = ocr_result.confidence * OCRErrorHandler.get_confidence_adjustment(nutrition)
                
                return FoodExtractionResult(
                    food_name=self.parser.extract_food_name(cleaned_text),
                    nutrition=nutrition,
                    source=NutritionSource.OCR,
                    confidence=confidence,
                    warnings=errors if not is_valid else [],
                    requires_rag_lookup=False
                )
        
        # Step 4b: No panel or incomplete - extract food name for RAG
        food_name = self.parser.extract_food_name(cleaned_text)
        
        if not food_name:
            # Try to get any meaningful text
            food_name = self._extract_best_food_name(cleaned_text)
        
        return FoodExtractionResult(
            food_name=food_name,
            nutrition=ParsedNutrition(),  # Empty - to be filled by RAG
            source=NutritionSource.RAG,
            confidence=ocr_result.confidence * 0.7,  # Lower confidence
            warnings=["Nutrition label not found, database lookup required"],
            requires_rag_lookup=True
        )
    
    def extract_from_text(self, text: str) -> FoodExtractionResult:
        """
        Extract nutrition from already-OCR'd text.
        Useful for testing or when OCR is done externally.
        """
        cleaned_text = OCRErrorHandler.preprocess_ocr_text(text)
        
        if self.parser.has_nutrition_panel(cleaned_text):
            nutrition = self.parser.parse(cleaned_text)
            is_valid, errors = OCRErrorHandler.validate_nutrition(nutrition)
            
            if nutrition.is_complete():
                return FoodExtractionResult(
                    food_name=self.parser.extract_food_name(cleaned_text),
                    nutrition=nutrition,
                    source=NutritionSource.OCR,
                    confidence=0.9 if is_valid else 0.7,
                    warnings=errors,
                    requires_rag_lookup=False
                )
        
        return FoodExtractionResult(
            food_name=self.parser.extract_food_name(cleaned_text),
            nutrition=ParsedNutrition(),
            source=NutritionSource.RAG,
            confidence=0.6,
            warnings=["Could not extract nutrition values"],
            requires_rag_lookup=True
        )
    
    def _run_ocr(self, image_path: str) -> OCRResult:
        """
        Run OCR engine on image.
        
        For production, integrate with actual OCR backend.
        """
        if self.engine_type == OCREngineType.MOCK:
            # Return empty result for mock engine
            return OCRResult(
                raw_text="",
                confidence=0.0,
                language="en"
            )
        
        if self.ocr_engine is None:
            raise ValueError("OCR engine not configured")
        
        # Call actual OCR engine
        result = self.ocr_engine.process(image_path)
        
        return OCRResult(
            raw_text=result.get("text", ""),
            confidence=result.get("confidence", 0.5),
            language=result.get("language", "en"),
            bounding_boxes=result.get("boxes", [])
        )
    
    def _extract_best_food_name(self, text: str) -> Optional[str]:
        """
        Extract the most likely food name from text when standard extraction fails.
        """
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        # Return first non-empty line that looks like a food name
        for line in lines[:3]:
            if len(line) >= 3 and len(line) <= 50:
                # Remove common non-food text
                if not any(skip in line.lower() for skip in 
                          ['ingredients', 'nutrition', 'serving', 'contains']):
                    return line
        
        return None
