# AI Nutrition Planner - Project Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statements](#problem-statements)
3. [System Architecture](#system-architecture)
4. [Technologies Used](#technologies-used)
5. [Key Features](#key-features)
6. [Data Flow](#data-flow)
7. [API Endpoints](#api-endpoints)

---

## 🎯 Project Overview

**AI Nutrition Planner** is an intelligent, context-aware nutrition guidance system that leverages AI to provide personalized dietary recommendations based on user health profiles, medical conditions, and food analysis.

The application uses **Retrieval-Augmented Generation (RAG)** to combine user-specific health data with AI-powered insights, ensuring personalized and medically-aware nutrition advice.

---

## ❓ Problem Statements

### 1. Lack of Personalized Nutrition Guidance
**Problem:** Generic nutrition apps don't consider individual health conditions (diabetes, hypertension, allergies) when providing dietary advice.

**Solution:** Our system retrieves user medical profiles and integrates them into every AI response, ensuring personalized recommendations.

### 2. Difficulty Understanding Food Safety
**Problem:** People with medical conditions struggle to know if a food is safe for them without consulting healthcare professionals.

**Solution:** Automated rule engine evaluates foods against user conditions and provides instant safety verdicts (Safe/Warning/Danger).

### 3. Medical Report Complexity
**Problem:** Medical reports contain important health data but are difficult for average users to interpret and apply to daily food choices.

**Solution:** OCR-based extraction of medical data (conditions, glucose levels, cholesterol) that automatically informs AI recommendations.

### 4. Meal Planning Challenges
**Problem:** Creating healthy meals that respect medical constraints is time-consuming and requires nutritional expertise.

**Solution:** "Fix My Meal" feature analyzes meals and suggests specific improvements based on user health profile.

### 5. Need for Real-time Food Analysis
**Problem:** Users can't quickly determine the nutritional content and health suitability of foods they encounter.

**Solution:** Image-based food recognition with instant nutrition analysis and safety checks.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Static HTML/JS)                │
│                    dashboard.html, app.html, login.html         │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND (main.py)                  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Auth Layer  │  │  RAG Service │  │  LLM Service │          │
│  │   (JWT)      │  │  (Retrieval) │  │ (Ollama/Gemma)│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Rule Engine  │  │   OCR Engine │  │ Food Database │          │
│  │ (Safety)     │  │  (EasyOCR)   │  │   (SQLite)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│                                                                  │
│    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│    │ nutrition.db│    │sample_foods │    │ User Meal   │       │
│    │  (SQLite)   │    │   (.csv)    │    │    Logs     │       │
│    └─────────────┘    └─────────────┘    └─────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies Used

### Backend Framework
| Technology | Version | Usage |
|------------|---------|-------|
| **FastAPI** | ≥0.100.0 | REST API framework, async support, automatic OpenAPI docs |
| **Uvicorn** | ≥0.23.0 | ASGI server for running FastAPI |
| **Pydantic** | ≥2.0.0 | Data validation and serialization |

### AI/ML Components
| Technology | Version | Usage |
|------------|---------|-------|
| **Ollama** | Latest | Local LLM inference server |
| **Gemma 3 (1B)** | gemma3:1b | Language model for nutrition advice |
| **EasyOCR** | ≥1.7.0 | Text extraction from medical reports |
| **Pillow** | ≥10.0.0 | Image processing for food/report images |

### Database & Storage
| Technology | Version | Usage |
|------------|---------|-------|
| **SQLite** | Built-in | User profiles, medical data, uploads |
| **SQLAlchemy** | ≥2.0.0 | ORM for database operations |
| **CSV** | - | Food nutrition database (126 foods) |

### Authentication & Security
| Technology | Usage |
|------------|-------|
| **JWT (JSON Web Tokens)** | User authentication |
| **bcrypt** | Password hashing |
| **python-dotenv** | Environment variable management |

### Frontend
| Technology | Usage |
|------------|-------|
| **HTML5** | Page structure |
| **Vanilla CSS** | Styling with modern design |
| **JavaScript** | API calls, UI interactions |

### DevOps & Deployment
| Technology | Usage |
|------------|-------|
| **Docker** | Containerization |
| **docker-compose** | Multi-container orchestration |
| **GitHub Actions** | CI/CD workflows |

---

## ✨ Key Features

### 1. Virtual Nutrition Coach
- AI-powered chat interface
- RAG integration for personalized responses
- Shows user health profile before each response
- Considers medical conditions in advice

### 2. Exercise Guidance AI
- General fitness recommendations
- Calorie burn estimates
- Considers user health conditions
- Safety warnings for medical concerns

### 3. Fix My Meal
- Analyzes meals against health profile
- Identifies problematic ingredients
- Suggests removals, reductions, replacements
- AI-generated improvement suggestions

### 4. Food Image Recognition
- Upload food images for analysis
- Nutrition extraction
- Safety evaluation
- Meal logging

### 5. Medical Report Processing
- OCR text extraction
- Condition detection (diabetes, hypertension, etc.)
- Allergen identification
- Vital signs extraction (glucose, cholesterol)

### 6. Recipe Generation
- Create recipes from ingredients
- Nutrition estimation
- Medical constraint checking

---

## 🔄 Data Flow (RAG Pipeline)

```
User Query → RAG Service → [Retrieve User Data]
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         Medical          Meal           Food
         Profile         History        Database
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                       Build Context
                              │
                              ▼
                    Ollama/Gemma LLM
                              │
                              ▼
                  Personalized Response
                  (with health profile shown)
```

---

## 📡 API Endpoints

### Authentication
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | User registration |
| `/api/auth/login` | POST | User login, returns JWT |

### AI Features
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/coach/chat` | POST | Virtual nutrition coach with RAG |
| `/api/exercise/chat` | POST | Exercise guidance AI |
| `/api/meal/fix` | POST | Fix My Meal analysis |

### Food Analysis
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/food/upload` | POST | Upload food image for analysis |
| `/api/food/analyze` | POST | Analyze food nutrition |
| `/api/recipe/generate` | POST | Generate recipe from ingredients |

### Medical Profile
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/medical-report/upload` | POST | Upload medical report for OCR |
| `/api/medical-profile` | GET | Get user's medical profile |

---

## 📁 Project Structure

```
AI Nutrition/
├── src/
│   ├── main.py              # FastAPI application
│   ├── services/
│   │   ├── llm_service.py   # Ollama/Gemma LLM integration
│   │   └── rag_service.py   # RAG retrieval service
│   ├── auth/
│   │   ├── auth_service.py  # JWT authentication
│   │   └── database.py      # User & medical profile DB
│   ├── rules/
│   │   └── engine.py        # Rule-based safety checks
│   ├── models/
│   │   └── food.py          # Food & nutrition models
│   └── ocr/
│       └── ...              # OCR processing
├── static/
│   ├── login.html           # Login page
│   ├── dashboard.html       # Main dashboard
│   └── app.html             # Food analysis app
├── data/
│   ├── sample_foods.csv     # Food nutrition database
│   └── nutrition.db         # SQLite database
├── models/                   # ML model files
├── docs/                     # Documentation
├── tests/                    # Unit tests
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 🚀 Getting Started

### Prerequisites
1. Python 3.10+
2. Ollama installed and running
3. Gemma model pulled: `ollama pull gemma3:1b`

### Installation
```bash
# Clone repository
git clone <repo-url>
cd "AI Nutrition"

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start the application
cd src
python -m uvicorn main:app --reload
```

### Access
- Dashboard: http://localhost:8000/static/dashboard.html
- API Docs: http://localhost:8000/docs

---

## 👥 Contributors
- Pratish01/Mcq-App

## 📄 License
This project is for educational purposes.
