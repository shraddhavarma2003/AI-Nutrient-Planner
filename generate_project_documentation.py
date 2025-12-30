"""
AI Nutrition Project - File Documentation Generator

Generates a comprehensive PDF explaining each file in the project.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import os
from datetime import datetime


# ============================================================================
# PROJECT FILE DOCUMENTATION
# ============================================================================

PROJECT_FILES = {
    "Root Files": {
        "main.py (src/)": {
            "purpose": "Main FastAPI application entry point",
            "description": """The core application file that defines all REST API endpoints for the AI Nutrition system.
            
Key Features:
- Authentication endpoints (login, register)
- Virtual Coach chat API with RAG integration
- Analytics endpoints (health score, trends, insights)
- Food scanning and logging APIs
- Medical report upload with OCR
- Exercise Guidance AI
- Fix My Meal (Clinical Nutrition AI)

Technologies: FastAPI, Pydantic, async/await""",
            "key_endpoints": [
                "POST /api/coach/chat - AI-powered nutrition chat",
                "GET /api/analytics/score - Health score calculation",
                "POST /api/food/scan - YOLO-based food recognition",
                "POST /api/upload/medical-report - OCR medical report parsing",
                "GET /api/analytics/feature-importance - Feature selection analysis"
            ]
        },
        "requirements.txt": {
            "purpose": "Python dependencies",
            "description": "Lists all required packages: FastAPI, uvicorn, python-dotenv, PyJWT, ultralytics (YOLO), etc."
        },
        "Dockerfile": {
            "purpose": "Container configuration",
            "description": "Docker configuration for containerized deployment of the application."
        },
        "docker-compose.yml": {
            "purpose": "Multi-container orchestration",
            "description": "Defines services, networks, and volumes for Docker deployment."
        }
    },
    
    "Analytics Module (src/analytics/)": {
        "analytics_service.py": {
            "purpose": "Health analytics and scoring",
            "description": """Computes health metrics and insights from user meal data.
            
Features:
- Health score calculation (0-100)
- Daily/weekly nutrient trends
- Pattern detection (late-night eating, sugar spikes)
- Actionable insights generation
- Meal log storage and retrieval"""
        },
        "feature_selection.py": {
            "purpose": "ML Feature Selection Analysis",
            "description": """Analyzes nutritional features to determine importance for health predictions.
            
Techniques Used:
- Pearson correlation analysis between nutrients
- Variance threshold filtering
- Feature importance scoring for health conditions
- Median-based missing value imputation
- Duplicate detection and removal

Health Conditions Analyzed:
- diabetes_risk (based on sugar/carbs)
- hypertension_risk (based on sodium)
- obesity_risk (based on calories/fats)
- heart_health_risk (based on fats)"""
        }
    },
    
    "Services Module (src/services/)": {
        "llm_service.py": {
            "purpose": "LLM Integration (Ollama/Gemma)",
            "description": """Provides AI-powered responses using local LLM models.
            
Features:
- Ollama API integration
- System prompts for nutrition coaching
- RAG context injection
- Fallback handling when LLM unavailable"""
        },
        "rag_service.py": {
            "purpose": "Retrieval-Augmented Generation",
            "description": """Retrieves relevant context for AI responses.
            
Data Sources:
- User medical profiles (conditions, allergens, medications)
- Meal history (recent logs)
- Food nutrition database (Indian Food CSV)
- Real-time food context from scans"""
        },
        "yolo_service.py": {
            "purpose": "YOLO Food Recognition",
            "description": """Computer vision for food detection.
            
Features:
- YOLOv8/v11 model integration
- Food item classification
- Confidence scoring
- Integration with nutrition lookup"""
        },
        "usda_service.py": {
            "purpose": "USDA Nutrition Database",
            "description": "External API integration for USDA FoodData Central nutrition lookups."
        }
    },
    
    "Rules Module (src/rules/)": {
        "engine.py": {
            "purpose": "Medical Rule Engine (Safety Layer)",
            "description": """Deterministic safety rules that ALWAYS override AI suggestions.
            
Rule Categories:
- Allergen detection (BLOCK severity)
- Diabetes rules (sugar, glycemic index, fiber-to-carb ratio)
- Hypertension rules (sodium monitoring)
- Obesity rules (calorie density, saturated fat)

Severity Levels: ALLOW → WARN → ALERT → BLOCK"""
        }
    },
    
    "Models Module (src/models/)": {
        "food.py": {
            "purpose": "Food and Nutrition Data Models",
            "description": """Pydantic/dataclass models for food items.
            
Classes:
- NutritionInfo (calories, protein, carbs, fat, sugar, fiber, sodium)
- Food (food_id, name, serving_size, nutrition, allergens)
- FoodCategory (enum)"""
        },
        "user.py": {
            "purpose": "User Profile Models",
            "description": """User health profile definitions.
            
Classes:
- UserProfile (user_id, conditions, allergens, daily_targets)
- HealthCondition (enum: DIABETES, HYPERTENSION, OBESITY)
- DailyTargets (calorie/nutrient limits)
- DailyIntake (current day's consumption)"""
        },
        "conversation.py": {
            "purpose": "Chat Conversation Models",
            "description": "Models for chat history, messages, and conversation context."
        },
        "feedback.py": {
            "purpose": "User Feedback Models",
            "description": "Models for collecting user feedback on AI responses."
        },
        "analytics_models.py": {
            "purpose": "Analytics Data Models",
            "description": "Models for health scores, trends, patterns, and insights."
        }
    },
    
    "OCR Module (src/ocr/)": {
        "parser.py": {
            "purpose": "Medical Report Parser",
            "description": """Extracts medical data from uploaded reports.
            
Extracts:
- Glucose levels (blood sugar)
- Cholesterol values (HDL, LDL, total)
- Blood pressure readings
- Health conditions detection"""
        },
        "service.py": {
            "purpose": "OCR Service Orchestrator",
            "description": "Coordinates OCR processing, PDF handling, and result storage."
        },
        "food_recognition.py": {
            "purpose": "Food Label OCR",
            "description": "Extracts nutrition facts from food packaging labels."
        },
        "error_handler.py": {
            "purpose": "OCR Error Handling",
            "description": "Robust error handling for OCR processing failures."
        }
    },
    
    "Coach Module (src/coach/)": {
        "virtual_coach.py": {
            "purpose": "AI Virtual Nutrition Coach",
            "description": """Context-aware nutrition coaching.
            
Features:
- Personalized dietary advice
- Food safety evaluation
- Meal suggestions
- Rule engine integration for medical safety"""
        }
    },
    
    "Intelligence Module (src/intelligence/)": {
        "recipe_generator.py": {
            "purpose": "AI Recipe Generation",
            "description": "Generates healthy recipes based on user preferences and restrictions."
        },
        "meal_fixer.py": {
            "purpose": "Clinical Meal Analysis",
            "description": """Analyzes meals and suggests improvements.
            
Features:
- Problem detection based on health conditions
- REMOVE/REDUCE/REPLACE suggestions
- Healthier alternatives"""
        },
        "router.py": {
            "purpose": "Intelligence Request Router",
            "description": "Routes requests to appropriate AI services."
        }
    },
    
    "Auth Module (src/auth/)": {
        "auth_service.py": {
            "purpose": "Authentication Service",
            "description": "JWT-based authentication, token generation and verification."
        },
        "database.py": {
            "purpose": "Database Operations",
            "description": "SQLite database for users, medical profiles, and uploads."
        }
    },
    
    "Feedback Module (src/feedback/)": {
        "feedback_service.py": {
            "purpose": "User Feedback Collection",
            "description": "Collects and stores user feedback on AI responses for improvement."
        }
    },
    
    "Frontend (static/)": {
        "login.html": {
            "purpose": "Login Page",
            "description": "User authentication interface."
        },
        "register.html": {
            "purpose": "Registration Page",
            "description": "New user registration form."
        },
        "dashboard.html": {
            "purpose": "Main Dashboard",
            "description": "Health score, trends, insights, and AI coach interface."
        },
        "food-scan.html": {
            "purpose": "Food Scanner",
            "description": "Camera-based food scanning with YOLO detection."
        },
        "upload.html": {
            "purpose": "Document Upload",
            "description": "Medical report upload for OCR processing."
        }
    },
    
    "Tests (tests/)": {
        "test_rule_engine.py": {
            "purpose": "Rule Engine Tests",
            "description": "Unit tests for medical safety rules."
        },
        "test_feature_selection.py": {
            "purpose": "Feature Selection Tests",
            "description": "Tests for correlation, variance, and feature importance."
        },
        "test_intelligence.py": {
            "purpose": "Intelligence Module Tests",
            "description": "Tests for meal fixer and recipe generator."
        },
        "test_phase3.py": {
            "purpose": "Integration Tests",
            "description": "End-to-end tests for complete workflows."
        }
    },
    
    "Data (data/)": {
        "Indian_Food_Nutrition_Processed.csv": {
            "purpose": "Nutrition Database",
            "description": """Indian food nutrition dataset with 1016 food items.
            
Columns:
- Dish Name
- Calories (kcal)
- Carbohydrates (g)
- Protein (g)
- Fats (g)
- Free Sugar (g)
- Fibre (g)
- Sodium (mg)
- Calcium (mg)
- Iron (mg)
- Vitamin C (mg)
- Folate (µg)"""
        }
    }
}


def create_pdf(output_path: str):
    """Generate the project documentation PDF."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a5f7a')
    )
    
    section_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#2c3e50')
    )
    
    file_style = ParagraphStyle(
        'FileTitle',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=12,
        textColor=colors.HexColor('#27ae60'),
        fontName='Helvetica-Bold'
    )
    
    purpose_style = ParagraphStyle(
        'Purpose',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#e74c3c'),
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        leading=14
    )
    
    # Build document content
    content = []
    
    # Title
    content.append(Paragraph("AI Nutrition Project", title_style))
    content.append(Paragraph("File Documentation", title_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                             ParagraphStyle('Date', parent=styles['Normal'], alignment=TA_CENTER)))
    content.append(Spacer(1, 40))
    
    # Table of Contents
    content.append(Paragraph("Table of Contents", section_style))
    for section_name in PROJECT_FILES.keys():
        content.append(Paragraph(f"• {section_name}", body_style))
    content.append(PageBreak())
    
    # Sections
    for section_name, files in PROJECT_FILES.items():
        content.append(Paragraph(section_name, section_style))
        content.append(Spacer(1, 10))
        
        for file_name, info in files.items():
            content.append(Paragraph(f"📄 {file_name}", file_style))
            content.append(Paragraph(f"Purpose: {info['purpose']}", purpose_style))
            content.append(Spacer(1, 4))
            
            # Description
            desc = info.get('description', '')
            for line in desc.split('\n'):
                if line.strip():
                    content.append(Paragraph(line.strip(), body_style))
            
            # Key endpoints if present
            if 'key_endpoints' in info:
                content.append(Spacer(1, 6))
                content.append(Paragraph("Key Endpoints:", purpose_style))
                for endpoint in info['key_endpoints']:
                    content.append(Paragraph(f"  • {endpoint}", body_style))
            
            content.append(Spacer(1, 15))
        
        content.append(PageBreak())
    
    # Build PDF
    doc.build(content)
    print(f"✅ PDF generated: {output_path}")


if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), "AI_Nutrition_File_Documentation.pdf")
    create_pdf(output_file)
