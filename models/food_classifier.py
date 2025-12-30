"""
Food Classifier Module

Integrates a trained .keras TensorFlow model for food image classification.
Maps predictions to nutrition data from CSV database and USDA API.

Designed for Food-101 dataset models.

Usage:
    from models.food_classifier import classify_food_image
    result = classify_food_image("path/to/food_image.jpg")
"""

import os
import json
import csv
from typing import Dict, Optional, Any
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "food_classifier.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
NUTRITION_CSV = os.path.join(os.path.dirname(MODEL_DIR), "data", "sample_foods.csv")

# Food-101 standard settings
IMAGE_SIZE = (224, 224)  # Most Food-101 models use 224x224

# Set to True if your model was trained with ImageNet preprocessing
# (e.g., MobileNet, ResNet, EfficientNet)
USE_IMAGENET_PREPROCESSING = True

# Set to True if your model was trained with simple /255 normalization
USE_SIMPLE_NORMALIZATION = False


@dataclass
class ClassificationResult:
    """Result from food image classification."""
    food_name: str
    confidence: float
    class_index: int
    nutrition: Dict[str, float]
    source: str


# =============================================================================
# GLOBAL STATE (Singleton pattern)
# =============================================================================

_model = None
_labels = None
_nutrition_db = None
_model_loaded = False


def _load_model():
    """Load the .keras model once at startup."""
    global _model, _model_loaded
    
    if _model_loaded:
        return _model
    
    _model_loaded = True
    
    if not os.path.exists(MODEL_PATH):
        print(f"[FoodClassifier] ERROR: Model not found at {MODEL_PATH}")
        print("[FoodClassifier] Please place your .keras model file in the models/ directory")
        return None
    
    try:
        import tensorflow as tf
        
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        
        print(f"[FoodClassifier] Loading model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        
        # Print model info
        print(f"[FoodClassifier] ✓ Model loaded successfully!")
        print(f"[FoodClassifier]   Input shape: {_model.input_shape}")
        print(f"[FoodClassifier]   Output shape: {_model.output_shape}")
        print(f"[FoodClassifier]   Number of classes: {_model.output_shape[-1]}")
        
        return _model
        
    except ImportError:
        print("[FoodClassifier] ERROR: TensorFlow not installed")
        print("[FoodClassifier] Run: pip install tensorflow")
        return None
    except Exception as e:
        print(f"[FoodClassifier] ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def _load_labels():
    """Load class labels from JSON."""
    global _labels
    
    if _labels is not None:
        return _labels
    
    if not os.path.exists(LABELS_PATH):
        print(f"[FoodClassifier] ERROR: Labels not found at {LABELS_PATH}")
        return {}
    
    try:
        with open(LABELS_PATH, 'r') as f:
            _labels = json.load(f)
        print(f"[FoodClassifier] ✓ Loaded {len(_labels)} class labels")
        
        # Validate labels
        expected_count = 101  # Food-101
        if len(_labels) != expected_count:
            print(f"[FoodClassifier] WARNING: Expected {expected_count} labels, got {len(_labels)}")
        
        return _labels
    except Exception as e:
        print(f"[FoodClassifier] ERROR: Failed to load labels: {e}")
        return {}


def _load_nutrition_db():
    """Load nutrition data from CSV."""
    global _nutrition_db
    
    if _nutrition_db is not None:
        return _nutrition_db
    
    _nutrition_db = {}
    
    if not os.path.exists(NUTRITION_CSV):
        print(f"[FoodClassifier] WARNING: Nutrition CSV not found at {NUTRITION_CSV}")
        return _nutrition_db
    
    try:
        with open(NUTRITION_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                food_id = row.get('food_id', '').lower().strip()
                name = row.get('name', '').lower().strip()
                
                nutrition = {
                    'calories': float(row.get('calories', 0) or 0),
                    'protein_g': float(row.get('protein_g', 0) or 0),
                    'carbs_g': float(row.get('carbs_g', 0) or 0),
                    'fat_g': float(row.get('fat_g', 0) or 0),
                    'sugar_g': float(row.get('sugar_g', 0) or 0),
                    'fiber_g': float(row.get('fiber_g', 0) or 0),
                    'sodium_mg': float(row.get('sodium_mg', 0) or 0),
                }
                
                # Store by food_id (exact match with labels)
                if food_id:
                    _nutrition_db[food_id] = nutrition
                
                # Also store by name variations
                if name:
                    _nutrition_db[name] = nutrition
                    # Simplified name without parentheses
                    simple = name.split('(')[0].strip()
                    if simple:
                        _nutrition_db[simple] = nutrition
        
        print(f"[FoodClassifier] ✓ Loaded nutrition data ({len(_nutrition_db)} entries)")
        return _nutrition_db
    except Exception as e:
        print(f"[FoodClassifier] ERROR: Failed to load nutrition CSV: {e}")
        return _nutrition_db


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def _preprocess_image(image_path: str):
    """
    Preprocess image for Food-101 model.
    
    Food-101 models typically use:
    - 224x224 or 299x299 input size
    - RGB color format
    - Normalization to [0, 1] or ImageNet preprocessing
    """
    try:
        from PIL import Image
        import numpy as np
        
        # Load image and convert to RGB
        img = Image.open(image_path).convert('RGB')
        print(f"[FoodClassifier] Original image size: {img.size}")
        
        # Resize to model input size
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Apply preprocessing
        if USE_IMAGENET_PREPROCESSING:
            # ImageNet preprocessing (for MobileNet, ResNet, etc.)
            try:
                import tensorflow as tf
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            except:
                # Manual ImageNet preprocessing: scale to [-1, 1]
                img_array = (img_array / 127.5) - 1.0
        elif USE_SIMPLE_NORMALIZATION:
            # Simple normalization to [0, 1]
            img_array = img_array / 255.0
        
        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"[FoodClassifier] Preprocessed shape: {img_array.shape}, dtype: {img_array.dtype}")
        print(f"[FoodClassifier] Pixel range: [{img_array.min():.2f}, {img_array.max():.2f}]")
        
        return img_array
        
    except ImportError:
        print("[FoodClassifier] ERROR: PIL not installed. Run: pip install pillow")
        return None
    except Exception as e:
        print(f"[FoodClassifier] ERROR: Image preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# NUTRITION LOOKUP
# =============================================================================

def _normalize_food_name(food_name: str) -> str:
    """Convert Food-101 label to readable format for USDA lookup."""
    # Convert underscores to spaces
    normalized = food_name.replace('_', ' ')
    # Title case
    normalized = normalized.title()
    return normalized


def _lookup_nutrition(food_name: str) -> Dict[str, float]:
    """
    Look up nutrition data for a food name.
    
    Priority:
    1. Exact match in CSV (by food_id)
    2. Name match in CSV
    3. USDA API (if CSV fails)
    4. Default fallback
    """
    if _nutrition_db is None:
        _load_nutrition_db()
    
    # Normalize the food name
    food_id = food_name.lower().replace(' ', '_')
    name_search = food_name.lower().replace('_', ' ')
    
    print(f"[FoodClassifier] Looking up nutrition for: '{food_name}'")
    print(f"[FoodClassifier]   food_id: '{food_id}'")
    print(f"[FoodClassifier]   name: '{name_search}'")
    
    # Try exact food_id match
    if food_id in _nutrition_db:
        print(f"[FoodClassifier] ✓ Found by food_id: {food_id}")
        return _nutrition_db[food_id]
    
    # Try name match
    if name_search in _nutrition_db:
        print(f"[FoodClassifier] ✓ Found by name: {name_search}")
        return _nutrition_db[name_search]
    
    # Try partial match
    for key in _nutrition_db:
        if food_id in key or key in food_id:
            print(f"[FoodClassifier] ✓ Partial match: {key}")
            return _nutrition_db[key]
    
    # Try word match
    words = name_search.split()
    for word in words:
        if len(word) > 3:  # Skip short words
            for key in _nutrition_db:
                if word in key:
                    print(f"[FoodClassifier] ✓ Word match: '{word}' in '{key}'")
                    return _nutrition_db[key]
    
    # Try USDA API
    print(f"[FoodClassifier] Not in CSV, trying USDA API...")
    try:
        import sys
        src_dir = os.path.join(os.path.dirname(MODEL_DIR), "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        from services.usda_service import fetch_usda_nutrition
        usda_result = fetch_usda_nutrition(name_search)
        
        if usda_result:
            print(f"[FoodClassifier] ✓ Found in USDA: {usda_result.get('food_name')}")
            return {
                'calories': usda_result.get('calories', 0),
                'protein_g': usda_result.get('protein_g', 0),
                'carbs_g': usda_result.get('carbs_g', 0),
                'fat_g': usda_result.get('fat_g', 0),
                'sugar_g': usda_result.get('sugar_g', 0),
                'fiber_g': usda_result.get('fiber_g', 0),
                'sodium_mg': usda_result.get('sodium_mg', 0),
            }
    except Exception as e:
        print(f"[FoodClassifier] USDA API error: {e}")
    
    # Default fallback
    print(f"[FoodClassifier] ✗ Not found, using defaults for: {food_name}")
    return {
        'calories': 200,
        'protein_g': 8,
        'carbs_g': 25,
        'fat_g': 8,
        'sugar_g': 5,
        'fiber_g': 2,
        'sodium_mg': 300,
    }


# =============================================================================
# MAIN CLASSIFICATION FUNCTION
# =============================================================================

def classify_food_image(image_path: str) -> Dict[str, Any]:
    """
    Classify a food image and return nutrition information.
    
    This is the MAIN ENTRY POINT for the food classification pipeline.
    
    Args:
        image_path: Path to the food image file
        
    Returns:
        Dictionary with food_name, confidence, nutrition, source
    """
    global _model, _labels
    
    print(f"\n{'='*60}")
    print(f"[FoodClassifier] Processing: {image_path}")
    print(f"{'='*60}")
    
    # Load model if needed
    if _model is None:
        _model = _load_model()
    
    # Load labels if needed
    if _labels is None:
        _labels = _load_labels()
    
    # Load nutrition DB if needed
    if _nutrition_db is None:
        _load_nutrition_db()
    
    # If model not available, use fallback
    if _model is None:
        print("[FoodClassifier] Model not available, using fallback")
        return _classify_fallback(image_path)
    
    # Preprocess image
    img_array = _preprocess_image(image_path)
    if img_array is None:
        print("[FoodClassifier] Preprocessing failed, using fallback")
        return _classify_fallback(image_path)
    
    try:
        import numpy as np
        
        # Run inference
        print("[FoodClassifier] Running model prediction...")
        predictions = _model.predict(img_array, verbose=0)
        
        print(f"[FoodClassifier] Prediction shape: {predictions.shape}")
        
        # Get top prediction
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx])
        
        print(f"[FoodClassifier] Top prediction: class {class_idx} with {confidence:.2%} confidence")
        
        # Get top 5 predictions for debugging
        top5_indices = np.argsort(predictions[0])[-5:][::-1]
        print("[FoodClassifier] Top 5 predictions:")
        for i, idx in enumerate(top5_indices):
            label = _labels.get(str(idx), f"class_{idx}")
            prob = predictions[0][idx]
            print(f"[FoodClassifier]   {i+1}. {label}: {prob:.2%}")
        
        # Map class index to food name
        if _labels and str(class_idx) in _labels:
            food_name_raw = _labels[str(class_idx)]
        else:
            print(f"[FoodClassifier] WARNING: Class {class_idx} not found in labels!")
            food_name_raw = f"class_{class_idx}"
        
        # Normalize food name for display
        food_name = _normalize_food_name(food_name_raw)
        
        print(f"[FoodClassifier] Food identified: {food_name}")
        
        # Look up nutrition
        nutrition = _lookup_nutrition(food_name_raw)
        
        print(f"[FoodClassifier] Nutrition: {nutrition.get('calories', 0)} cal")
        print(f"{'='*60}\n")
        
        return {
            "food_name": food_name,
            "confidence": confidence,
            "class_index": class_idx,
            "nutrition": nutrition,
            "source": "model",
        }
        
    except Exception as e:
        print(f"[FoodClassifier] ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return _classify_fallback(image_path)


def _classify_fallback(image_path: str) -> Dict[str, Any]:
    """Fallback when model is not available."""
    filename = os.path.basename(image_path).lower()
    
    # Try to guess from filename
    common_foods = ['pizza', 'burger', 'hamburger', 'salad', 'chicken', 'rice', 
                    'pasta', 'sandwich', 'egg', 'soup', 'apple', 'banana']
    
    for food in common_foods:
        if food in filename:
            nutrition = _lookup_nutrition(food)
            return {
                "food_name": food.title(),
                "confidence": 0.5,
                "class_index": -1,
                "nutrition": nutrition,
                "source": "fallback",
            }
    
    return {
        "food_name": "Mixed Meal",
        "confidence": 0.3,
        "class_index": -1,
        "nutrition": _lookup_nutrition("default"),
        "source": "fallback",
    }


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def get_model_status() -> Dict[str, Any]:
    """Get diagnostic information about the model."""
    global _model, _labels, _nutrition_db
    
    if _model is None:
        _load_model()
    if _labels is None:
        _load_labels()
    if _nutrition_db is None:
        _load_nutrition_db()
    
    status = {
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_loaded": _model is not None,
        "labels_path": LABELS_PATH,
        "labels_count": len(_labels) if _labels else 0,
        "nutrition_csv": NUTRITION_CSV,
        "nutrition_count": len(_nutrition_db) if _nutrition_db else 0,
        "image_size": IMAGE_SIZE,
        "preprocessing": "imagenet" if USE_IMAGENET_PREPROCESSING else "simple",
    }
    
    if _model is not None:
        try:
            status["input_shape"] = str(_model.input_shape)
            status["output_shape"] = str(_model.output_shape)
            status["num_classes"] = int(_model.output_shape[-1])
        except:
            pass
    
    return status


def test_classification(test_class_index: int = 0) -> Dict[str, str]:
    """Test the label mapping for a specific class index."""
    if _labels is None:
        _load_labels()
    
    label = _labels.get(str(test_class_index), "NOT FOUND")
    return {
        "class_index": test_class_index,
        "label": label,
        "normalized": _normalize_food_name(label) if label != "NOT FOUND" else "N/A"
    }


# Initialize on import
print("[FoodClassifier] Initializing...")
_load_labels()
_load_nutrition_db()
