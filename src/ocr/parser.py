"""
Nutrition Parser Module

Regex-based extraction of nutrition values from OCR text.
Handles various label formats (US, EU, Indian).
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class NutritionSource(Enum):
    """Source of nutrition data."""
    OCR = "ocr"           # Extracted directly from label
    RAG = "rag"           # Retrieved from database
    MANUAL = "manual"     # User entered


@dataclass
class ParsedNutrition:
    """
    Parsed nutrition values from OCR or database.
    All values are per serving unless specified otherwise.
    """
    calories: Optional[float] = None
    carbs_g: Optional[float] = None
    protein_g: Optional[float] = None
    fat_g: Optional[float] = None
    sugar_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    fiber_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    cholesterol_mg: Optional[float] = None
    potassium_mg: Optional[float] = None
    serving_size: Optional[str] = None
    serving_unit: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if we have minimum required fields for logging."""
        return self.calories is not None and (
            self.carbs_g is not None or 
            self.protein_g is not None or 
            self.fat_g is not None
        )
    
    def completeness_score(self) -> float:
        """Calculate how complete the nutrition data is (0-1)."""
        fields = [
            self.calories, self.carbs_g, self.protein_g, 
            self.fat_g, self.sugar_g, self.sodium_mg
        ]
        filled = sum(1 for f in fields if f is not None)
        return filled / len(fields)
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in {
            "calories": self.calories,
            "carbs_g": self.carbs_g,
            "protein_g": self.protein_g,
            "fat_g": self.fat_g,
            "sugar_g": self.sugar_g,
            "sodium_mg": self.sodium_mg,
            "fiber_g": self.fiber_g,
            "saturated_fat_g": self.saturated_fat_g,
            "cholesterol_mg": self.cholesterol_mg,
            "potassium_mg": self.potassium_mg,
            "serving_size": self.serving_size,
            "serving_unit": self.serving_unit,
        }.items() if v is not None}


class NutritionParser:
    """
    Parses nutrition values from OCR text using regex patterns.
    
    Supports multiple label formats:
    - US Nutrition Facts
    - EU Nutrition Information
    - Indian FSSAI labels
    """
    
    # Regex patterns for each nutrient
    # Multiple patterns per nutrient to handle variations
    PATTERNS = {
        "calories": [
            r"calories[:\s]*(\d+)",
            r"energy[:\s]*(\d+)\s*kcal",
            r"(\d+)\s*kcal",
            r"cal[:\s]*(\d+)",
            r"kilocalories[:\s]*(\d+)",
            r"kcal[:\s]*(\d+)",
        ],
        "carbs": [
            r"total\s+carbohydrate[s]?[:\s]*(\d+\.?\d*)\s*g",
            r"carbohydrate[s]?[:\s]*(\d+\.?\d*)\s*g",
            r"carbs[:\s]*(\d+\.?\d*)\s*g",
            r"carb[:\s]*(\d+\.?\d*)",
        ],
        "protein": [
            r"protein[s]?[:\s]*(\d+\.?\d*)\s*g",
            r"proteins[:\s]*(\d+\.?\d*)",
        ],
        "fat": [
            r"total\s+fat[:\s]*(\d+\.?\d*)\s*g",
            r"fat[:\s]*(\d+\.?\d*)\s*g",
            r"fats[:\s]*(\d+\.?\d*)",
        ],
        "saturated_fat": [
            r"saturated\s+fat[:\s]*(\d+\.?\d*)\s*g",
            r"saturated[:\s]*(\d+\.?\d*)\s*g",
            r"sat\.\s*fat[:\s]*(\d+\.?\d*)",
        ],
        "sugar": [
            r"total\s+sugar[s]?[:\s]*(\d+\.?\d*)\s*g",
            r"sugar[s]?[:\s]*(\d+\.?\d*)\s*g",
            r"of\s+which\s+sugars[:\s]*(\d+\.?\d*)",
        ],
        "sodium": [
            r"sodium[:\s]*(\d+\.?\d*)\s*mg",
            r"sodium[:\s]*(\d+\.?\d*)",
            r"na[:\s]*(\d+\.?\d*)\s*mg",
        ],
        "salt": [
            r"salt[:\s]*(\d+\.?\d*)\s*g",  # Convert: salt_g * 400 = sodium_mg
        ],
        "fiber": [
            r"dietary\s+fib[er|re]+[:\s]*(\d+\.?\d*)\s*g",
            r"fibre[:\s]*(\d+\.?\d*)\s*g",
            r"fiber[:\s]*(\d+\.?\d*)\s*g",
        ],
        "cholesterol": [
            r"cholesterol[:\s]*(\d+\.?\d*)\s*mg",
        ],
        "potassium": [
            r"potassium[:\s]*(\d+\.?\d*)\s*mg",
        ],
        "serving_size": [
            r"serving\s+size[:\s]*([^\n\r]+)",
            r"per\s+serving[:\s]*\(?([^\n\r\)]+)\)?",
            r"portion[:\s]*([^\n\r]+)",
        ],
    }
    
    # Keywords that indicate a nutrition facts panel
    PANEL_INDICATORS = [
        "nutrition facts",
        "nutritional information", 
        "nutrition information",
        "per serving",
        "per 100g",
        "per 100ml",
        "amount per serving",
        "calories",
        "protein",
        "carbohydrate",
    ]
    
    def has_nutrition_panel(self, text: str) -> bool:
        """
        Detect if OCR text contains a nutrition facts panel.
        Requires at least 2 indicators to avoid false positives.
        """
        text_lower = text.lower()
        matches = sum(1 for indicator in self.PANEL_INDICATORS 
                     if indicator in text_lower)
        return matches >= 2
    
    def parse(self, text: str) -> ParsedNutrition:
        """
        Parse nutrition values from OCR text.
        
        Args:
            text: Raw OCR text output
            
        Returns:
            ParsedNutrition with extracted values
        """
        text_lower = text.lower()
        nutrition = ParsedNutrition()
        
        # Extract each nutrient
        nutrition.calories = self._extract_value(text_lower, "calories")
        nutrition.carbs_g = self._extract_value(text_lower, "carbs")
        nutrition.protein_g = self._extract_value(text_lower, "protein")
        nutrition.fat_g = self._extract_value(text_lower, "fat")
        nutrition.saturated_fat_g = self._extract_value(text_lower, "saturated_fat")
        nutrition.sugar_g = self._extract_value(text_lower, "sugar")
        nutrition.fiber_g = self._extract_value(text_lower, "fiber")
        nutrition.cholesterol_mg = self._extract_value(text_lower, "cholesterol")
        nutrition.potassium_mg = self._extract_value(text_lower, "potassium")
        
        # Handle sodium (may come from salt)
        sodium = self._extract_value(text_lower, "sodium")
        if sodium is None:
            salt = self._extract_value(text_lower, "salt")
            if salt is not None:
                sodium = salt * 400  # Convert salt (g) to sodium (mg)
        nutrition.sodium_mg = sodium
        
        # Extract serving size
        serving = self._extract_serving(text_lower)
        if serving:
            nutrition.serving_size = serving[0]
            nutrition.serving_unit = serving[1]
        
        return nutrition
    
    def _extract_value(self, text: str, nutrient: str) -> Optional[float]:
        """Extract numeric value for a nutrient using regex patterns."""
        patterns = self.PATTERNS.get(nutrient, [])
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Basic validation
                    if value >= 0:
                        return value
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_serving(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract serving size and unit."""
        for pattern in self.PATTERNS["serving_size"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                serving_text = match.group(1).strip()
                # Try to parse quantity and unit
                qty_match = re.search(r"(\d+\.?\d*)\s*(g|ml|oz|cup|piece|bar)", 
                                     serving_text, re.IGNORECASE)
                if qty_match:
                    return (qty_match.group(1), qty_match.group(2).lower())
                return (serving_text, "serving")
        
        return None
    
    def extract_food_name(self, text: str) -> Optional[str]:
        """
        Extract food/product name from OCR text.
        Usually appears before nutrition panel or at the top.
        """
        lines = text.strip().split('\n')
        
        # Filter out nutrition-related lines
        nutrition_words = {'nutrition', 'calories', 'fat', 'protein', 'carb', 
                          'serving', 'amount', 'daily', 'value', 'ingredients'}
        
        for line in lines[:5]:  # Check first 5 lines
            line_lower = line.lower().strip()
            # Skip empty or short lines
            if len(line_lower) < 3:
                continue
            # Skip lines with nutrition keywords
            if any(word in line_lower for word in nutrition_words):
                continue
            # This is likely the product name
            return line.strip()
        
        return None


# =============================================================================
# MEDICAL REPORT OCR PARSER
# =============================================================================

# Known medical conditions to detect
KNOWN_CONDITIONS = [
    "diabetes", "diabetic", "type 1 diabetes", "type 2 diabetes",
    "hypertension", "high blood pressure", "hbp", "elevated bp",
    "heart disease", "cardiovascular", "coronary artery disease", "cad",
    "kidney disease", "renal disease", "ckd", "chronic kidney",
    "liver disease", "hepatic", "fatty liver",
    "thyroid", "hypothyroid", "hyperthyroid",
    "obesity", "overweight", "bmi >30",
    "cholesterol", "hyperlipidemia", "high cholesterol",
    "anemia", "low hemoglobin", "iron deficiency",
    "gout", "uric acid", "hyperuricemia",
    "celiac", "gluten intolerance",
    "lactose intolerance", "dairy intolerance",
    "ibs", "irritable bowel",
    "crohn", "ulcerative colitis",
    "gastritis", "gerd", "acid reflux",
]

# Known allergens to detect
KNOWN_ALLERGENS = [
    "peanut", "peanuts", "groundnut",
    "tree nut", "almond", "walnut", "cashew", "pistachio", "hazelnut",
    "milk", "dairy", "lactose",
    "egg", "eggs",
    "wheat", "gluten",
    "soy", "soybean", "soya",
    "fish", "shellfish", "shrimp", "crab", "lobster",
    "sesame",
    "sulfite", "sulphite",
    "mustard",
]


def parse_medical_report(file_path: str) -> dict:
    """
    Parse a medical report file and extract conditions and allergens.
    
    Uses OCR (pytesseract if available) to extract text,
    then pattern matching to identify medical conditions and allergens.
    
    Args:
        file_path: Path to the medical report (PDF or image)
        
    Returns:
        dict with:
            - raw_text: The extracted OCR text
            - conditions: List of detected medical conditions
            - allergens: List of detected allergens
    """
    print(f"[OCR] Parsing medical report: {file_path}")
    
    raw_text = ""
    conditions = []
    allergens = []
    
    try:
        # Try to use pytesseract for OCR
        import pytesseract
        from PIL import Image
        
        # Configure Tesseract path for Windows
        import os
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\hp\AppData\Local\Tesseract-OCR\tesseract.exe',
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"[OCR] Using Tesseract: {path}")
                break
        else:
            print("[OCR] Tesseract not found in common paths - using system PATH")
        
        # Handle PDF vs image
        if file_path.lower().endswith('.pdf'):
            # Try pdf2image for PDF
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(file_path)
                for page in pages:
                    raw_text += pytesseract.image_to_string(page) + "\n"
            except ImportError:
                print("[OCR] pdf2image not installed, cannot process PDF")
                raw_text = f"PDF file: {file_path} (pdf2image not installed)"
        else:
            # Process image
            img = Image.open(file_path)
            raw_text = pytesseract.image_to_string(img)
        
        print(f"[OCR] Extracted {len(raw_text)} characters")
        print(f"[OCR] Text sample (first 500 chars):\n{raw_text[:500]}")
        
    except ImportError:
        print("[OCR] pytesseract not installed, using filename heuristics")
        # Fallback: try to extract info from filename
        raw_text = f"File: {file_path}"
    except Exception as e:
        print(f"[OCR] Error: {e}")
        raw_text = f"Error processing file: {e}"
    
    # Extract conditions from text
    text_lower = raw_text.lower()
    
    for condition in KNOWN_CONDITIONS:
        if condition in text_lower:
            # Normalize condition name
            normalized = condition.replace("type 1 diabetes", "Type 1 Diabetes")
            normalized = normalized.replace("type 2 diabetes", "Type 2 Diabetes") 
            normalized = normalized.replace("diabetic", "Diabetes")
            normalized = normalized.replace("diabetes", "Diabetes")
            normalized = normalized.replace("hypertension", "Hypertension")
            normalized = normalized.replace("high blood pressure", "Hypertension")
            normalized = normalized.replace("kidney disease", "Kidney Disease")
            normalized = normalized.replace("heart disease", "Heart Disease")
            normalized = normalized.replace("thyroid", "Thyroid Disorder")
            normalized = normalized.replace("cholesterol", "High Cholesterol")
            normalized = normalized.replace("obesity", "Obesity")
            normalized = normalized.replace("anemia", "Anemia")
            
            # Only add if not already present
            if normalized.title() not in conditions:
                conditions.append(normalized.title())
    
    # Extract allergens from text
    for allergen in KNOWN_ALLERGENS:
        if allergen in text_lower:
            # Normalize allergen name
            normalized = allergen.title()
            if "peanut" in allergen:
                normalized = "Peanuts"
            elif "nut" in allergen:
                normalized = "Tree Nuts"
            elif "milk" in allergen or "dairy" in allergen:
                normalized = "Dairy"
            elif "egg" in allergen:
                normalized = "Eggs"
            elif "wheat" in allergen or "gluten" in allergen:
                normalized = "Gluten"
            elif "fish" in allergen or "shellfish" in allergen:
                normalized = "Seafood"
            elif "soy" in allergen:
                normalized = "Soy"
            
            if normalized not in allergens:
                allergens.append(normalized)
    
    print(f"[OCR] Found conditions: {conditions}")
    print(f"[OCR] Found allergens: {allergens}")
    
    # =================================================================
    # EXTRACT VITALS (Glucose Level, Cholesterol)
    # =================================================================
    vitals = {}
    
    # Glucose level patterns - more flexible to match various formats
    # Matches: "Glucose fasting (PHO) 83 mg/dl", "Glucose: 83", "Blood Sugar 126 mg/dL"
    glucose_patterns = [
        r'glucose\s*(?:fasting|level)?[^0-9]*(\d{2,3}(?:\.\d)?)\s*(?:mg/?dl)?',
        r'(?:blood\s*sugar|fbs|rbs|ppbs)[^0-9]*(\d{2,3}(?:\.\d)?)',
        r'(\d{2,3})\s*mg/?dl[^0-9]*glucose',
    ]
    
    for pattern in glucose_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            try:
                value = float(matches[0])
                if 40 <= value <= 400:  # Valid glucose range
                    vitals["glucose_level"] = value
                    print(f"[OCR] Found glucose level: {value}")
                    break
            except ValueError:
                pass
    
    # Cholesterol patterns - more flexible
    # Matches: "Cholesterol, total (PHO) 221 mg/dl", "Total Cholesterol: 200"
    cholesterol_patterns = [
        r'cholesterol[,\s]*(?:total)?[^0-9]*(\d{2,3}(?:\.\d)?)\s*(?:mg/?dl)?',
        r'(?:total\s*cholesterol|chol)[^0-9]*(\d{2,3}(?:\.\d)?)',
        r'(\d{2,3})\s*mg/?dl[^0-9]*cholesterol',
    ]
    
    for pattern in cholesterol_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            try:
                value = float(matches[0])
                if 50 <= value <= 400:  # Valid cholesterol range
                    vitals["cholesterol"] = value
                    print(f"[OCR] Found cholesterol: {value}")
                    break
            except ValueError:
                pass
    
    print(f"[OCR] Found vitals: {vitals}")
    
    return {
        "raw_text": raw_text,
        "conditions": conditions,
        "allergens": allergens,
        "vitals": vitals,
    }
