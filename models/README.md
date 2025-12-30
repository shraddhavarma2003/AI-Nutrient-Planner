# Models Directory

This directory contains the trained food classification model.

## Required Files

| File | Description |
|------|-------------|
| `food_classifier.keras` | Trained TensorFlow model (you provide) |
| `labels.json` | Class index to food name mapping |
| `food_classifier.py` | Python module for model inference |

## Setup

1. Place your trained `.keras` model file here as `food_classifier.keras`
2. Update `labels.json` to match your model's class indices
3. Restart the server

## Usage

```python
from models.food_classifier import classify_food_image

result = classify_food_image("path/to/food_image.jpg")
# Returns:
# {
#     "food_name": "Pizza",
#     "confidence": 0.92,
#     "nutrition": {"calories": 285, "protein_g": 12, ...},
#     "source": "model"
# }
```

## Model Requirements

- Input shape: `(224, 224, 3)` (configurable in `food_classifier.py`)
- Output: Softmax probabilities for each food class
- Format: TensorFlow SavedModel in `.keras` format
