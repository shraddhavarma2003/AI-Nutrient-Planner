# Context-Aware Personalized Nutrition Planner

A healthcare-support system (NOT a medical diagnosis tool) that helps users log meals,
track nutrition, and receive personalized guidance based on their health conditions.

## Features

- Multi-modal meal logging (text, image, voice)
- RAG-powered nutrition lookup from USDA/OpenFoodFacts
- Medical Safety Engine (allergies, diabetes, hypertension, obesity)
- Smart meal corrections ("Fix My Meal")
- Reverse Recipe Generation
- Context-aware Virtual Coach

## Project Structure

```
AI Nutrition/
├── data/                   # Nutrition datasets
├── src/
│   ├── models/            # Data models
│   ├── rules/             # Medical rule engine
│   └── database/          # Database schemas
└── tests/                 # Unit tests
```

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Important

- This is a healthcare SUPPORT tool, not a medical diagnosis system
- Medical rules always override AI suggestions
- Nutrition data comes from trusted databases only (never guessed)
