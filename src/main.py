"""
FastAPI Application for AI Nutrition

Provides REST API endpoints for:
- Authentication (login, register)
- Virtual Coach (chat)
# RELOAD TRIGGER 1
- Analytics (health score, trends, insights)
- Food logging
- Medical report upload with OCR
"""

# Load environment variables FIRST (before other imports that may need them)
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import os
import sys
import uuid
import tempfile
import re
import json
from dataclasses import asdict

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

# =============================================================================
# MANDATORY DEPENDENCY CHECK - HARD FAIL IF MISSING
# =============================================================================
# The stable food recognition pipeline REQUIRES SigLIP via HuggingFace Transformers.
# If SigLIP is not available, the app MUST NOT start.
# This prevents silent fallback to broken classification.

try:
    import transformers
    print(f"[STARTUP] HuggingFace Transformers v{transformers.__version__} available - stable food recognition enabled")
except ImportError as e:
    print("[STARTUP] CRITICAL ERROR: HuggingFace Transformers is NOT installed!")
    print("[STARTUP] The stable food recognition pipeline CANNOT run without SigLIP.")
    print("[STARTUP] To fix: pip install transformers torch")
    raise RuntimeError(
        "HuggingFace Transformers is not installed. Stable food recognition pipeline cannot run. "
        "Install with: pip install transformers torch"
    ) from e

from models.food import Food, NutritionInfo, FoodCategory
from models.user import UserProfile, HealthCondition, DailyTargets, DailyIntake, ActivityLevel
from rules.engine import RuleEngine
from coach.virtual_coach import VirtualCoach
from analytics.analytics_service import AnalyticsService, MealLogStore
from feedback.feedback_service import FeedbackService, FeedbackStore
from auth.auth_service import auth_service
from auth.database import (
    init_database, UserRepository, MedicalProfileRepository, 
    UploadRepository, MealRepository, DailyLogRepository
)
from services.llm_service import get_mistral_service, get_llm_service
from services.rag_service import get_rag_service
from analytics.weight_forecaster import get_weight_forecaster
# Legacy pipeline removed
from services.continental_retrieval import get_continental_retrieval_system
from services.nutrition_registry import get_nutrition_registry
from services.meal_planner import WeeklyPlanGenerator, GroceryGenerator
from services.analytics_engine import AnalyticsEngine

# =============================================================================
# APP SETUP
# =============================================================================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Foundational startup: database init and eager model loading."""
    print("[STARTUP] Initializing database...")
    from auth.database import init_database
    init_database()
    
    print("[STARTUP] Warming up Continental Food Retrieval System (CLIP)...")
    # Eager load the new CLIP retrieval system
    continental_system = get_continental_retrieval_system()
    print("[STARTUP] System READY!")
    
    yield
    print("[SHUTDOWN] Cleaning up resources...")

app = FastAPI(
    title="AI Nutrition API",
    description="Context-aware nutrition guidance with medical safety",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (Robust Pathing)
from pathlib import Path
static_path = Path(__file__).parent.parent / "static"
static_path_vals = str(static_path.resolve())
print(f"[STARTUP] Mounting static files from: {static_path_vals}")

if not os.path.exists(static_path_vals):
    print(f"[STARTUP] WARNING: Static directory does not exist at {static_path_vals}")
    alt_path = Path(os.getcwd()) / "static"
    if alt_path.exists():
        print(f"[STARTUP] Falling back to: {alt_path}")
        static_path_vals = str(alt_path.resolve())

app.mount("/static", StaticFiles(directory=static_path_vals), name="static")


# =============================================================================
# GLOBAL STATE & SERVICES
# =============================================================================

# Food Recognition & Nutrition
# Food Recognition & Nutrition
continental_system = get_continental_retrieval_system()
nutrition_registry = get_nutrition_registry()
nutrition_registry = get_nutrition_registry()

# Default demo user
demo_user = UserProfile(
    user_id="demo-user",
    name="Demo User",
    conditions=[HealthCondition.DIABETES],
    allergens=["peanuts"],
    daily_targets=DailyTargets.for_diabetes(),
)

# In-memory meal log storage (per user)
# Format: { user_id: [{ food_name, nutrition, timestamp, source }, ...] }
user_meal_logs: Dict[str, List[Dict[str, Any]]] = {}

# Current food context (most recently scanned food per user)
# AI coach uses this for immediate context after food scan
current_food_context: Dict[str, Dict[str, Any]] = {}

# Services
rule_engine = RuleEngine()
meal_log_store = MealLogStore()
analytics_service = AnalyticsService(meal_log_store)
feedback_service = FeedbackService()
virtual_coach = VirtualCoach(rule_engine, demo_user)
weight_forecaster = get_weight_forecaster()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    food: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    message: str
    safety_level: str
    confidence: float
    violations: List[Dict[str, Any]] = []
    suggestions: List[Dict[str, Any]] = []


class FoodInput(BaseModel):
    name: str
    serving_size: float = 100
    serving_unit: str = "g"
    calories: float
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0
    sugar_g: float = 0
    fiber_g: float = 0
    sodium_mg: float = 0


class LogMealRequest(BaseModel):
    foods: List[FoodInput]


class FeedbackRequest(BaseModel):
    context_type: str
    context_id: str
    rating: str  # helpful, not_helpful, incorrect
    comment: Optional[str] = None


# =============================================================================
# AUTHENTICATION DEPENDENCY
# =============================================================================

async def get_current_user(authorization: Optional[str] = Header(None)):
    """Dependency to get current authenticated user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = authorization.replace("Bearer ", "")
    payload = auth_service.verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload


async def get_optional_user(authorization: Optional[str] = Header(None)):
    """Dependency to get current user or None (for demo mode)."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    token = authorization.replace("Bearer ", "")
    payload = auth_service.verify_token(token)
    
    return payload  # Returns None if invalid


def get_user_profile_for_rules(user_id: str) -> UserProfile:
    """
    Load user's medical profile from database and construct a UserProfile
    object with proper HealthCondition enums for rule engine evaluation.
    
    This ensures the rule engine uses the user's ACTUAL conditions,
    not hardcoded demo values.
    """
    # Try to get profile from database
    db_profile = MedicalProfileRepository.get_by_user_id(user_id)
    
    if not db_profile:
        # Fall back to demo user if no profile exists
        return demo_user
    
    # Map condition strings to HealthCondition enums
    conditions = []
    for condition_str in db_profile.get("conditions", []):
        condition_lower = condition_str.lower()
        if "diabetes" in condition_lower:
            conditions.append(HealthCondition.DIABETES)
        elif "hypertension" in condition_lower or "blood pressure" in condition_lower:
            conditions.append(HealthCondition.HYPERTENSION)
        elif "obesity" in condition_lower or "overweight" in condition_lower:
            conditions.append(HealthCondition.OBESITY)
        # Log unknown conditions for debugging
        else:
            print(f"[RuleEngine] Unknown condition: {condition_str}")
    
    # Get allergens from database
    allergens = db_profile.get("allergens", [])
    
    # Determine daily targets based on conditions
    if HealthCondition.DIABETES in conditions:
        targets = DailyTargets.for_diabetes()
    elif HealthCondition.HYPERTENSION in conditions:
        targets = DailyTargets.for_hypertension()
    elif HealthCondition.OBESITY in conditions:
        targets = DailyTargets.for_weight_loss()
    else:
        targets = DailyTargets()
    
    # Create UserProfile with actual conditions
    user_profile = UserProfile(
        user_id=user_id,
        name=db_profile.get("name", "User"),
        conditions=conditions,
        allergens=allergens,
        daily_targets=targets,
        age=db_profile.get("age"),
        gender=db_profile.get("gender"),
        weight_kg=db_profile.get("weight_kg"),
        height_cm=db_profile.get("height_cm"),
        activity_level=ActivityLevel(db_profile.get("activity_level") or "moderately_active"),
        fitness_goal=db_profile.get("fitness_goal"),
    )
    
    print(f"[RuleEngine] Loaded profile for {user_id}: conditions={[c.value for c in conditions]}, allergens={allergens}")
    
    return user_profile


# =============================================================================
# ROUTES: INDEX
# =============================================================================

@app.get("/")
async def index():
    """Redirect to login page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/login.html")


@app.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration.
    
    Used by:
    - Docker HEALTHCHECK
    - Kubernetes liveness/readiness probes
    - Load balancer health checks
    """
    return {
        "status": "healthy",
        "version": "4.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "services": {
            "rule_engine": "ok",
            "coach": "ok",
            "analytics": "ok",
            "feedback": "ok",
        },
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# ROUTES: COACH
# =============================================================================

@app.post("/api/coach/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
    """Chat with the Virtual Coach using RAG + Mistral LLM."""
    with open("debug_chat.log", "a") as f:
        f.write(f"[{datetime.now()}] Chat called: {request.message}\n")
    try:
        user_id = user["sub"]
        
        # Retrieve recent meal logs for context
        # Convert MealLogEntry objects to dicts for RAG service compatibility
        all_logs = meal_log_store.get_by_user(user_id)
        user_meal_logs = {
            user_id: [log.to_dict() for log in all_logs]
        }
        
        # =====================================================
        # RAG STEP 1: RETRIEVAL - Auto-fetch food context
        # =====================================================
        food_data = None
        
        # Priority 1: Use food from request if provided
        if request.food and request.food.get("name"):
            food_data = request.food
            print(f"[RAG] Using food from request: {food_data.get('name')}")
        
        # Priority 2: Auto-retrieve from current_food_context (last scanned food)
        elif user_id in current_food_context:
            ctx = current_food_context[user_id]
            food_data = {
                "name": ctx.get("food_name"),
                "calories": ctx["nutrition"]["calories"],
                "protein_g": ctx["nutrition"]["protein_g"],
                "carbs_g": ctx["nutrition"]["carbs_g"],
                "fat_g": ctx["nutrition"]["fat_g"],
                "sugar_g": ctx["nutrition"]["sugar_g"],
                "fiber_g": ctx["nutrition"]["fiber_g"],
                "sodium_mg": ctx["nutrition"]["sodium_mg"],
            }
            print(f"[RAG] Auto-retrieved food context: {food_data.get('name')}")
        else:
            print(f"[RAG] No food context available")
        
        # =====================================================
        # RAG STEP 2: AUGMENTATION - Build Food object
        # =====================================================
        food = None
        if food_data and food_data.get("name"):
            # Get values with defaults
            calories = float(food_data.get("calories", 0) or 0)
            protein_g = float(food_data.get("protein_g", 0) or 0)
            carbs_g = float(food_data.get("carbs_g", 0) or 0)
            fat_g = float(food_data.get("fat_g", 0) or 0)
            sugar_g = float(food_data.get("sugar_g", 0) or 0)
            fiber_g = float(food_data.get("fiber_g", 0) or 0)
            sodium_mg = float(food_data.get("sodium_mg", 0) or 0)
            
            # Ensure carbs >= sugar + fiber (validation requirement)
            if carbs_g < sugar_g + fiber_g:
                carbs_g = sugar_g + fiber_g
            
            food = Food(
                food_id=f"input-{datetime.now().timestamp()}",
                name=food_data.get("name", "Unknown Food"),
                serving_size=float(food_data.get("serving_size", 100) or 100),
                serving_unit=food_data.get("serving_unit", "g") or "g",
                nutrition=NutritionInfo(
                    calories=calories,
                    protein_g=protein_g,
                    carbs_g=carbs_g,
                    fat_g=fat_g,
                    sugar_g=sugar_g,
                    fiber_g=fiber_g,
                    sodium_mg=sodium_mg,
                ),
                allergens=food_data.get("allergens") or [],
            )
            print(f"[RAG] Built Food object: {food.name} ({food.nutrition.calories} cal)")
        
        # =====================================================
        # RAG STEP 3: GENERATION - Use Ollama/Gemma with RAG context
        # =====================================================
        # =====================================================
        # RAG STEP 3: GENERATION - Use Ollama/Gemma with RAG context
        # =====================================================
        llm_service = get_llm_service()
        
        rag_context = ""
        rag_service = None
        user_profile_data = None
        
        try:
            rag_service = get_rag_service()
            # Build comprehensive RAG context from user data
            rag_context = rag_service.build_context(
                user_id=user_id,
                meal_logs=user_meal_logs,
                current_food=food_data,
                user_question=request.message
            )
            print(f"[RAG] Built context: {len(rag_context)} chars")
            
            # Cache profile data for later use to avoid re-fetching
            user_profile_data = rag_service.get_medical_profile(user_id)
            
        except Exception as e:
            print(f"[Coach] RAG Service warning (continuing without RAG): {e}")
            import traceback
            traceback.print_exc()
        
        # Try LLM-powered response if available
        llm_response = None
        if llm_service.is_available:
            print(f"[Coach] LLM is available, calling chat...")
            try:
                llm_response = llm_service.chat(
                    prompt=request.message,
                    system_prompt="nutrition_coach",
                    rag_context=rag_context
                )
            except Exception as e:
                print(f"[Coach] LLM chat failed: {e}")
                llm_response = None
            
            if llm_response and llm_response.success:
                print(f"[Coach] Using Ollama/Gemma LLM with RAG context")
                
                # Get user profile data to show in response
                profile_header = ""
                profile = user_profile_data
                
                if profile and (profile.get('conditions') or profile.get('allergens')):
                    profile_header = "📋 **Your Health Profile:**\n"
                    if profile.get('conditions'):
                        profile_header += f"• Conditions: {', '.join(profile['conditions'])}\n"
                    if profile.get('allergens'):
                        profile_header += f"• Allergens: {', '.join(profile['allergens'])}\n"
                    if profile.get('medications'):
                        profile_header += f"• Medications: {', '.join(profile['medications'])}\n"
                    if profile.get('_is_demo'):
                        profile_header += "_(Demo profile - upload your medical report for personalized data)_\n"
                    profile_header += "\n---\n\n"
                
                # Still run rule engine for safety checks
                violations = []
                safety_level = "safe"
                if food:
                    from rules.engine import RuleEngine
                    rule_engine_local = RuleEngine()
                    
                    try:
                         # get_user_profile_for_rules is defined in global scope of main.py
                         user_profile = get_user_profile_for_rules(user_id)
                         
                         rule_violations = rule_engine_local.evaluate(food, user_profile)
                         violations = [v.to_dict() for v in rule_violations]
                         verdict = rule_engine_local.get_final_verdict(rule_violations)
                         safety_level = verdict.value
                    except Exception as e:
                         print(f"[Coach] Rule engine warning: {e}")

                return ChatResponse(
                    message=profile_header + llm_response.content,
                    safety_level=safety_level,
                    confidence=0.9,
                    violations=violations,
                    suggestions=[{"type": "llm_powered", "source": "ollama_gemma", "rag_enabled": True}],
                )
            else:
                if llm_response:
                    print(f"[Coach] LLM response failed: {llm_response.error}")
        else:
            print(f"[Coach] LLM service not available, using fallback")
        
        # Fallback to original rule-based coach - STILL SHOW PROFILE DATA
        profile = user_profile_data
        # If accessing rag_service failed earlier, user_profile_data is None. Try one more time strictly for profile?
        # Or just use empty.
        
        profile_header = ""
        if profile and (profile.get('conditions') or profile.get('allergens')):
            profile_header = "📋 **Your Health Profile:**\n"
            if profile.get('conditions'):
                profile_header += f"• Conditions: {', '.join(profile['conditions'])}\n"
            if profile.get('allergens'):
                profile_header += f"• Allergens: {', '.join(profile['allergens'])}\n"
            if profile.get('medications'):
                profile_header += f"• Medications: {', '.join(profile['medications'])}\n"
            profile_header += "\n---\n\n"
        
        # Isolation fix: Update virtual_coach state for current user
        user_profile = get_user_profile_for_rules(user_id)
        virtual_coach.user = user_profile
        virtual_coach.context.user_id = user_id
        
        response = virtual_coach.respond(request.message, food=food)
        
        return ChatResponse(
            message=profile_header + response.message,
            safety_level=response.safety_level,
            confidence=response.confidence,
            violations=response.violations,
            suggestions=response.suggestions,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/coach/history")
async def get_history():
    """Get conversation history."""
    return {
        "session_id": virtual_coach.context.session_id,
        "messages": [m.to_dict() for m in virtual_coach.context.messages],
    }


@app.post("/api/coach/clear")
async def clear_history():
    """Clear conversation history."""
    virtual_coach.clear_context()
    return {"status": "cleared"}


# =============================================================================
# ROUTES: EXERCISE GUIDANCE AI
# =============================================================================

class ExerciseRequest(BaseModel):
    message: str

# Simple exercise knowledge base
EXERCISE_DATA = {
    "calorie_burn": {
        "walking": 4,      # cal per minute
        "running": 11,
        "cycling": 8,
        "swimming": 10,
        "yoga": 3,
        "weight_training": 5,
        "hiit": 12,
        "dancing": 6,
        "jumping_rope": 12,
    },
    "met_values": {
        "walking": 3.5,
        "running": 9.8,
        "cycling": 7.5,
        "swimming": 8.0,
        "yoga": 2.5,
    }
}

@app.post("/api/exercise/chat")
async def exercise_chat(request: ExerciseRequest, user: dict = Depends(get_current_user)):
    """
    Exercise Guidance AI - provides general fitness advice.
    Uses Ollama/Gemma LLM for intelligent responses.
    Shows user health profile for personalized context.
    """
    user_id = user["sub"]
    message = request.message.lower().strip()
    
    # Get user profile for personalized advice
    rag_service = get_rag_service()
    profile = rag_service.get_medical_profile(user_id)
    
    # Build profile header to show in response
    profile_header = ""
    if profile and (profile.get('conditions') or profile.get('allergens')):
        profile_header = "📋 **Your Health Profile:**\n"
        if profile.get('conditions'):
            profile_header += f"• Conditions: {', '.join(profile['conditions'])}\n"
        if profile.get('allergens'):
            profile_header += f"• Allergens: {', '.join(profile['allergens'])}\n"
        if profile.get('medications'):
            profile_header += f"• Medications: {', '.join(profile['medications'])}\n"
        if profile.get('_is_demo'):
            profile_header += "_(Demo profile - upload your medical report for personalized data)_\n"
        profile_header += "\n---\n\n"
    
    # Disclaimer to include in responses
    disclaimer = "\n\n_This is general guidance only, not medical advice. Consult a healthcare provider before starting any exercise program._"
    
    # Check for dangerous/medical questions
    dangerous_keywords = ["injury", "pain", "hurt", "chest pain", "dizzy", "faint", "pregnant", "surgery", "heart condition"]
    if any(kw in message for kw in dangerous_keywords):
        return {
            "message": profile_header + "That sounds like a medical concern. Please consult a healthcare professional or doctor before exercising. Your safety is the priority!" + disclaimer,
            "type": "safety_warning"
        }
    
    # Try LLM-powered response first
    llm_service = get_llm_service()
    if llm_service.is_available:
        # Build RAG context for personalized exercise advice
        rag_context = f"""=== USER HEALTH PROFILE ===
Health Conditions: {', '.join(profile.get('conditions', [])) if profile else 'None'}
Allergens: {', '.join(profile.get('allergens', [])) if profile else 'None'}
Medications: {', '.join(profile.get('medications', [])) if profile else 'None'}

Consider these health conditions when providing exercise advice."""
        
        llm_response = llm_service.chat(
            prompt=request.message,
            system_prompt="exercise_guide",
            rag_context=rag_context
        )
        if llm_response.success:
            return {
                "message": profile_header + llm_response.content + disclaimer,
                "type": "llm_response",
                "powered_by": "ollama_gemma"
            }
    
    # Fallback to rule-based responses
    # Calorie burn questions
    if "burn" in message or "calorie" in message:
        response = "**Estimated Calorie Burn (per minute):**\n"
        response += "• Walking: ~4 cal/min\n"
        response += "• Running: ~11 cal/min\n"
        response += "• Cycling: ~8 cal/min\n"
        response += "• Swimming: ~10 cal/min\n"
        response += "• HIIT: ~12 cal/min\n"
        response += "• Weight training: ~5 cal/min\n"
        response += "• Yoga: ~3 cal/min\n\n"
        response += "These are approximate values for a 70kg person. Actual burn varies based on intensity and body weight."
        return {"message": profile_header + response + disclaimer, "type": "calorie_info"}
    
    # Exercise recommendations
    if "recommend" in message or "suggest" in message or "should i" in message:
        response = "**General Exercise Recommendations:**\n\n"
        response += "• **Beginners:** Start with 20-30 min walking, 3x per week\n"
        response += "• **Cardio:** 150 min moderate OR 75 min vigorous per week\n"
        response += "• **Strength:** 2-3 sessions per week, all major muscle groups\n"
        response += "• **Flexibility:** Daily stretching or yoga\n\n"
        response += "Start slowly and gradually increase intensity!"
        return {"message": profile_header + response + disclaimer, "type": "recommendation"}
    
    # Weight loss questions
    if "lose weight" in message or "weight loss" in message or "fat" in message:
        response = "**Weight Loss Exercise Tips:**\n\n"
        response += "1. **Combine cardio + strength training**\n"
        response += "2. **HIIT** is time-efficient for burning calories\n"
        response += "3. **Aim for 300+ min of moderate exercise per week**\n"
        response += "4. **Add walking** throughout the day (10,000 steps goal)\n"
        response += "5. **Consistency** matters more than intensity\n\n"
        response += "Remember: Diet + exercise work together for weight loss."
        return {"message": profile_header + response + disclaimer, "type": "weight_loss"}
    
    # Muscle building
    if "muscle" in message or "strength" in message or "build" in message:
        response = "**Muscle Building Basics:**\n\n"
        response += "1. **Lift weights** 3-4 times per week\n"
        response += "2. **Progressive overload** - gradually increase weight\n"
        response += "3. **Compound exercises:** squats, deadlifts, bench press\n"
        response += "4. **Rest:** 48 hours between same muscle groups\n"
        response += "5. **Protein intake:** important for muscle recovery"
        return {"message": profile_header + response + disclaimer, "type": "strength"}
    
    # Default helpful response
    response = "I'm your Exercise Guidance AI! I can help with:\n\n"
    response += "• **Calorie burn estimates** - \"How many calories does running burn?\"\n"
    response += "• **Exercise recommendations** - \"What exercises should I do?\"\n"
    response += "• **Weight loss tips** - \"How to exercise for weight loss?\"\n"
    response += "• **Strength training** - \"How to build muscle?\"\n\n"
    response += "Ask me anything about fitness and exercise!"
    
    return {"message": profile_header + response + disclaimer, "type": "general"}


# =============================================================================
# ROUTES: FIX MY MEAL (Clinical Nutrition AI)
# =============================================================================

class FixMealRequest(BaseModel):
    food_name: Optional[str] = None
    nutrition: Optional[Dict[str, Any]] = None
    # If not provided, will auto-fetch from current_food_context

# Medical thresholds for conditions
CONDITION_THRESHOLDS = {
    "diabetes": {"sugar_g": 10, "carbs_g": 45},
    "hypertension": {"sodium_mg": 500},
    "high_cholesterol": {"fat_g": 15, "saturated_fat_g": 5},
    "kidney_disease": {"sodium_mg": 400, "protein_g": 20},
    "obesity": {"calories": 400, "fat_g": 15},
}

@app.post("/api/meal/fix")
async def fix_meal_endpoint(request: FixMealRequest, user: dict = Depends(get_current_user)):
    """
    Clinical Nutrition AI - analyzes meal and suggests improvements.
    Uses detected food data, portion estimation, and user medical profile.
    Provides actionable fix suggestions based on nutritional analysis.
    """
    user_id = user["sub"]
    
    # Get food data - from request or auto-fetch from context
    food_name = request.food_name
    nutrition = request.nutrition
    portion_grams = 100  # Default portion size
    
    if not food_name or not nutrition:
        # Auto-fetch from current_food_context
        if user_id in current_food_context:
            ctx = current_food_context[user_id]
            food_name = ctx.get("food_name", "Unknown Food")
            nutrition = ctx.get("nutrition", {})
            # Try to get portion estimate if available
            portion_grams = ctx.get("portion_grams", 100)
        else:
            return {
                "verdict": "No Data",
                "message": "No food data available. Please scan a food item first.",
                "problems": [],
                "suggestions": {}
            }
    
    # Get user medical profile and conditions
    profile = MedicalProfileRepository.get_by_user_id(user_id)
    conditions = []
    if profile:
        conditions = profile.get("conditions", [])
    
    # Build detected items list for the meal fix service
    detected_items = [{
        "name": food_name,
        "grams": portion_grams,
        # Pass nutrition directly if we have it
        "nutrition_override": nutrition
    }]
    
    # Use the new MealFixService for smart analysis
    try:
        from services.meal_fix_service import get_meal_fix_service
        meal_fix = get_meal_fix_service()
        
        # Run analysis
        result = meal_fix.analyze_meal(detected_items, conditions)
        
        if result.get("success"):
            analysis = result.get("analysis", {})
            suggestions = result.get("suggestions", [])
            totals = analysis.get("totals", {})
            
            # Build formatted response with emoji template
            if result.get("verdict") == "Healthy":
                formatted_response = f"✅ This meal is suitable for your dietary needs.\n\n"
                formatted_response += f"📊 Nutrition Summary:\n"
                formatted_response += f"  • Calories: {totals.get('calories', 0):.0f} kcal\n"
                formatted_response += f"  • Protein: {totals.get('protein_g', 0):.0f}g\n"
                formatted_response += f"  • Carbs: {totals.get('carbs_g', 0):.0f}g\n"
                formatted_response += f"  • Fat: {totals.get('fat_g', 0):.0f}g\n"
            else:
                formatted_response = f"⚠️ This meal needs some adjustments based on your health profile.\n\n"
                
                # List problems with icons
                formatted_response += "❌ Issues Found:\n"
                for s in suggestions:
                    if s.get("type") == "warning":
                        formatted_response += f"  {s.get('icon', '⚠️')} {s.get('title')}: {s.get('message')}\n"
                
                # List fixes
                formatted_response += "\n💡 How to Fix:\n"
                for s in suggestions:
                    if s.get("fix"):
                        formatted_response += f"  • {s.get('fix')}\n"
                
                # Nutrition summary
                formatted_response += f"\n📊 Current Nutrition:\n"
                formatted_response += f"  • Calories: {totals.get('calories', 0):.0f} kcal\n"
                formatted_response += f"  • Protein: {totals.get('protein_g', 0):.0f}g\n"
                formatted_response += f"  • Carbs: {totals.get('carbs_g', 0):.0f}g\n"
                formatted_response += f"  • Sugar: {totals.get('sugar_g', 0):.0f}g\n"
                formatted_response += f"  • Sodium: {totals.get('sodium_mg', 0):.0f}mg\n"
            
            # Try LLM enhancement if available
            llm_suggestions = None
            llm_service = get_llm_service()
            
            if llm_service.is_available and suggestions:
                try:
                    # Build simple context string instead of raw dict for model efficiency
                    context_lines = [
                        f"Current Meal: {food_name}",
                        f"Health Profile: {', '.join(conditions) if conditions else 'No specific conditions'}"
                    ]
                    
                    # Add current nutrition
                    nut_str = f"Calories: {totals.get('calories',0)}, Protein: {totals.get('protein_g',0)}g, Sugar: {totals.get('sugar_g',0)}g"
                    context_lines.append(f"Nutrition: {nut_str}")
                    
                    # Detailed issues list
                    issues_str = "\n".join([
                        f"- {s.get('title')}: {s.get('message')}"
                        for s in suggestions if s.get("type") == "warning"
                    ])
                    
                    fix_prompt = f"""Analyze the likely ingredients of '{food_name}' and provide 3 quick, practical fixes.
                    
MEAL INFO: {food_name}
NUTRITION: {nut_str}
ISSUES DETECTED:
{issues_str}

USER CONDITIONS: {', '.join(conditions) if conditions else 'None'}

INSTRUCTIONS:
1. Infer the typical ingredients in this dish (e.g., refined flour, oil, sugar).
2. Explain potential health risks of these specific ingredients for the user's conditions.
3. Suggest specific swaps (e.g., "Use almond flour instead of wheat")."""
                    
                    llm_response = llm_service.chat(
                        prompt=fix_prompt,
                        system_prompt="clinical_nutritionist",
                        rag_context="\n".join(context_lines)
                    )
                    if llm_response.success:
                        llm_suggestions = llm_response.content
                        formatted_response += f"\n\n🤖 AI Dietitian's Advice:\n{llm_suggestions}"
                except Exception as e:
                    print(f"[MEAL FIX] LLM enhancement error: {e}")
            
            return {
                "verdict": result.get("verdict"),
                "message": result.get("message"),
                "formatted_response": formatted_response,
                "food_name": food_name,
                "portion_grams": portion_grams,
                "nutrition": totals,
                "targets": result.get("targets", {}),
                "problems": [
                    {
                        "category": s.get("category"),
                        "title": s.get("title"),
                        "message": s.get("message")
                    }
                    for s in suggestions if s.get("type") == "warning"
                ],
                "suggestions": [
                    {
                        "category": s.get("category"),
                        "fix": s.get("fix")
                    }
                    for s in suggestions if s.get("fix")
                ],
                "conditions_checked": conditions
            }
            
    except ImportError as ie:
        print(f"[MEAL FIX] MealFixService import error: {ie}")
    except Exception as e:
        print(f"[MEAL FIX] Analysis error: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback to basic analysis if MealFixService fails
    print(f"[MEAL FIX] Running basic fallback analysis for {food_name}")
    calories = nutrition.get("calories", 0)
    carbs = nutrition.get("carbs_g", 0)
    sugar = nutrition.get("sugar_g", 0)
    sodium = nutrition.get("sodium_mg", 0)
    protein = nutrition.get("protein_g", 0)
    
    problems = []
    suggestions_list = []
    
    # Basic checks
    if sugar > 15:
        problems.append({"title": "High Sugar", "message": f"Sugar content is {sugar}g (limit: 15g)"})
        suggestions_list.append({"category": "sugar", "fix": "Choose unsweetened alternatives or non-starchy vegetables."})
    if sodium > 600:
        problems.append({"title": "High Sodium", "message": f"Sodium is {sodium}mg (limit: 600mg)"})
        suggestions_list.append({"category": "sodium", "fix": "Use herbs and spices instead of salt. Avoid processed toppings."})
    if protein < 15:
        problems.append({"title": "Low Protein", "message": f"Only {protein}g protein (need 15-25g)"})
        suggestions_list.append({"category": "protein", "fix": "Add a protein source like eggs, paneer, dal, or lean meat."})
    
    verdict = "Healthy" if not problems else "Needs Fix"
    
    # Build a more descriptive formatted response for the fallback
    if verdict == "Healthy":
        formatted_response = f"✅ Your meal '{food_name}' looks well-balanced."
    else:
        formatted_response = f"⚠️ Issues found in your meal '{food_name}':\n\n"
        for p in problems:
            formatted_response += f"• **{p['title']}**: {p['message']}\n"
        formatted_response += "\n💡 Quick Fix:\n"
        for s in suggestions_list:
            formatted_response += f"• {s['fix']}\n"
    
    return {
        "verdict": verdict,
        "message": f"Analyzed {food_name}" + (f" with {len(problems)} issues" if problems else ""),
        "formatted_response": formatted_response,
        "food_name": food_name,
        "nutrition": nutrition,
        "problems": problems,
        "suggestions": suggestions_list,
        "conditions_checked": conditions
    }


# =============================================================================
# ROUTES: ANALYTICS
# =============================================================================

@app.get("/api/analytics/score")
async def get_health_score(user: dict = Depends(get_current_user)):
    """Get current health score."""
    user_id = user["sub"]
    user_profile = get_user_profile_for_rules(user_id)
    score = analytics_service.compute_health_score(
        user_id,
        user_profile.daily_targets,
    )
    return score.to_dict()


@app.get("/api/analytics/trends")
async def get_trends(days: int = 7, user: dict = Depends(get_current_user)):
    """Get nutrient trends."""
    user_id = user["sub"]
    user_profile = get_user_profile_for_rules(user_id)
    trends = analytics_service.compute_daily_trends(
        user_id,
        user_profile.daily_targets,
        days=days,
    )
    return {k: v.to_dict() for k, v in trends.items()}


@app.get("/api/analytics/patterns")
async def get_patterns(days: int = 14, user: dict = Depends(get_current_user)):
    """Get detected behavior patterns."""
    user_id = user["sub"]
    user_profile = get_user_profile_for_rules(user_id)
    patterns = analytics_service.detect_patterns(
        user_id,
        user_profile.daily_targets,
        days=days,
    )
    return [p.to_dict() for p in patterns]


@app.get("/api/analytics/insight")
async def get_insight(user: dict = Depends(get_current_user)):
    """Get daily actionable insight."""
    user_id = user["sub"]
    user_profile = get_user_profile_for_rules(user_id)
    insight = analytics_service.generate_insight(
        user_id,
        user_profile.daily_targets,
    )
    return {"insight": insight}


@app.get("/api/analytics/snapshot")
async def get_snapshot(user: dict = Depends(get_current_user)):
    """Get complete analytics snapshot."""
    user_id = user["sub"]
    user_profile = get_user_profile_for_rules(user_id)
    snapshot = analytics_service.get_snapshot(
        user_id,
        user_profile.daily_targets,
    )
    return snapshot.to_dict()


@app.get("/api/analytics/feature-importance")
async def get_feature_importance():
    """
    Get feature importance analysis for health predictions.
    
    Returns rankings of nutritional features based on their importance
    for predicting various health conditions (diabetes, hypertension, etc.).
    """
    try:
        from analytics.feature_selection import analyze_features
        results = analyze_features()
        return {
            "status": "success",
            "data": results
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }


# =============================================================================
# ROUTES: FOOD
# =============================================================================


@app.post("/api/food/analyze")
async def analyze_food(food: FoodInput, user: dict = Depends(get_current_user)):
    """Analyze a food item for safety."""
    user_id = user["sub"]
    # Ensure carbs >= sugar + fiber (validation requirement)
    carbs_g = food.carbs_g
    if carbs_g < food.sugar_g + food.fiber_g:
        carbs_g = food.sugar_g + food.fiber_g
    
    # Load user's actual profile for rule evaluation
    user_profile = get_user_profile_for_rules(user_id)
    
    food_obj = Food(
        food_id=f"analyze-{datetime.now().timestamp()}",
        name=food.name,
        serving_size=food.serving_size,
        serving_unit=food.serving_unit,
        nutrition=NutritionInfo(
            calories=food.calories,
            protein_g=food.protein_g,
            carbs_g=carbs_g,
            fat_g=food.fat_g,
            sugar_g=food.sugar_g,
            fiber_g=food.fiber_g,
            sodium_mg=food.sodium_mg,
        ),
        allergens=[],
    )
    
    violations = rule_engine.evaluate(food_obj, user_profile)
    verdict = rule_engine.get_final_verdict(violations)
    
    return {
        "food": food.name,
        "verdict": verdict.value,
        "violations": [v.to_dict() for v in violations],
        "formatted": rule_engine.format_violations(violations),
    }


@app.post("/api/food/log")
async def log_meal(request: LogMealRequest, user: dict = Depends(get_current_user)):
    """Log a meal."""
    user_id = user["sub"]
    foods = []
    for f in request.foods:
        foods.append({
            "name": f.name,
            "calories": f.calories,
            "protein_g": f.protein_g,
            "carbs_g": f.carbs_g,
            "fat_g": f.fat_g,
            "sugar_g": f.sugar_g,
            "fiber_g": f.fiber_g,
            "sodium_mg": f.sodium_mg,
        })
    
    entry = analytics_service.log_meal(
        user_id=user_id,
        foods=foods,
    )
    
    return {
        "log_id": entry.log_id,
        "total_nutrition": entry.total_nutrition,
        "meal_type": entry.meal_type,
    }


# =============================================================================
# ROUTES: FEEDBACK
# =============================================================================

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest, user: dict = Depends(get_current_user)):
    """Submit feedback on a response."""
    user_id = user["sub"]
    feedback = feedback_service.submit_feedback(
        user_id=user_id,
        context_type=request.context_type,
        context_id=request.context_id,
        rating=request.rating,
        comment=request.comment,
    )
    return {"feedback_id": feedback.feedback_id, "status": "submitted"}


# =============================================================================
# ROUTES: USER
# =============================================================================

# Keep old endpoint for backward compatibility
@app.get("/api/user/profile")
async def get_user_profile_legacy(user: dict = Depends(get_current_user)):
    """Legacy endpoint - redirects to /api/profile"""
    return await get_medical_profile(user)


# =============================================================================
# ROUTES: MEDICAL REPORT UPLOAD
# =============================================================================

@app.post("/api/medical-report/upload")
async def upload_medical_report(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Upload a medical report and extract conditions, allergens, and vitals.
    Uses OCR to parse the document.
    """
    user_id = user["sub"]
    print(f"\n[MEDICAL REPORT] ====== Processing medical report ======")
    print(f"[MEDICAL REPORT] User: {user_id}")
    print(f"[MEDICAL REPORT] File: {file.filename}")
    
    try:
        # Save the uploaded file temporarily
        import tempfile
        import os
        
        content = await file.read()
        
        # Get file extension
        ext = os.path.splitext(file.filename)[1].lower() if file.filename else '.pdf'
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"[MEDICAL REPORT] Saved to: {tmp_path}")
        
        # Parse the medical report using OCR
        from ocr.parser import parse_medical_report
        result = parse_medical_report(tmp_path)
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        print(f"[MEDICAL REPORT] Conditions (Regex): {result.get('conditions', [])}")
        print(f"[MEDICAL REPORT] Vitals (Regex): {result.get('vitals', {})}")
        
        # Enhanced LLM Parsing
        from services.llm_service import get_llm_service
        llm = get_llm_service()
        
        # Consolidate bio-metrics from history (all previous profiles)
        existing_biometrics = MedicalProfileRepository.get_consolidated_biometrics(user_id)
        
        # Initialize lists with regex results, merging with existing biometrics
        final_conditions = result.get('conditions', [])
        final_allergens = result.get('allergens', [])
        final_vitals = result.get('vitals', {})
        final_biometrics = existing_biometrics.copy()
        # Regex result overrides existing if present
        final_biometrics.update(result.get('biometrics', {}))
        final_medications = result.get('medications', [])
        
        if llm.is_available and result.get('raw_text'):
            print("[MEDICAL REPORT] 🧠 Analyzing text with AI...")
            llm_response = llm.parse_medical_report(result['raw_text'])
            
            if llm_response.success:
                try:
                    import json
                    import re
                    
                    # Clean JSON parsing
                    content = llm_response.content.strip()
                    
                    # 1. Try to extract from Markdown code blocks first
                    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if code_block_match:
                        content = code_block_match.group(1)
                    else:
                        # 2. Fallback to finding first outer brace pair
                        # Non-greedy match to avoid capturing extra text at end
                        json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                        if json_match:
                            content = json_match.group(1)
                    
                    # 3. Cleanup common LLM JSON errors
                    # Fix trailing commas: ,} -> } and ,] -> ]
                    content = re.sub(r',\s*\}', '}', content)
                    content = re.sub(r',\s*\]', ']', content)
                    
                    ai_data = json.loads(content)
                    print(f"[MEDICAL REPORT] AI Data: {ai_data}")
                    
                    # Merge structured data
                    if ai_data.get('conditions'):
                        # Union of regex and AI conditions
                        final_conditions = list(set(final_conditions + ai_data['conditions']))
                        
                    if ai_data.get('allergens'):
                        final_allergens = list(set(final_allergens + ai_data['allergens']))
                    
                    if ai_data.get('medications'):
                        final_medications = list(set(final_medications + ai_data['medications']))
                        
                    # Update vitals (AI overrides regex if present and non-zero)
                    if ai_data.get('vitals'):
                        for k, v in ai_data['vitals'].items():
                            if v is not None and v != 0:
                                final_vitals[k] = v
                    
                    # Update biometrics
                    if ai_data.get('biometrics'):
                        for k, v in ai_data['biometrics'].items():
                            if v is not None:
                                final_biometrics[k] = v
                                
                except Exception as e:
                    print(f"[MEDICAL REPORT] AI parsing failed: {e}")
        
        # Build daily_targets with vitals
        daily_targets = final_vitals
        
        # Save to database
        import uuid
        profile_id = str(uuid.uuid4())
        
        MedicalProfileRepository.create(
            profile_id=profile_id,
            user_id=user_id,
            conditions=final_conditions,
            allergens=final_allergens,
            medications=final_medications,
            daily_targets=daily_targets,  # Contains vitals
            raw_ocr_text=result.get('raw_text', '')[:5000],  # Limit size
            source_file=file.filename,
            **final_biometrics
        )
        
        print(f"[MEDICAL REPORT] ✓ Saved profile: {profile_id}")
        
        return {
            "success": True,
            "profile_id": profile_id,
            "profile": {  # Frontend expects 'profile' key for immediate display
                "conditions": final_conditions,
                "allergens": final_allergens,
                "vitals": final_vitals,
                "biometrics": final_biometrics,
                "medications": final_medications
            },
            "conditions": final_conditions,
            "allergens": final_allergens,
            "vitals": final_vitals,
            "biometrics": final_biometrics,
            "medications": final_medications,
            "debug_text": result.get('raw_text', '')[:2000],  # Return first 2000 chars for debug
            "message": f"Analyzed report. Found {len(final_conditions)} conditions and {len(final_vitals)} vitals."
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[MEDICAL REPORT] Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to process medical report"
        }


# =============================================================================
# ROUTES: AUTHENTICATION
# =============================================================================

class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


@app.post("/auth/register")
async def register(request: RegisterRequest):
    """Register a new user."""
    result = auth_service.register(request.email, request.password, request.name)
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {
        "user_id": result.user_id,
        "token": result.token,
        "name": request.name,
    }


@app.post("/auth/login")
async def login(request: LoginRequest):
    """Login and get JWT token."""
    result = auth_service.login(request.email, request.password)
    
    if not result.success:
        raise HTTPException(status_code=401, detail=result.error)
    
    user = auth_service.get_user(result.user_id)
    
    return {
        "user_id": result.user_id,
        "token": result.token,
        "name": user.get("name", "User") if user else "User",
    }


@app.get("/auth/me")
async def get_current_user_info(user: dict = Depends(get_current_user)):
    """Get current user info."""
    user_data = auth_service.get_user(user["sub"])
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    return user_data


@app.post("/auth/logout")
async def logout(user: dict = Depends(get_current_user)):
    """
    Logout the current user.
    
    For JWT-based auth, the actual token invalidation happens client-side
    by removing the token from storage. This endpoint:
    1. Confirms the user was authenticated
    2. Can be extended to add token to a blacklist if needed
    3. Provides a clean API contract for frontend logout flows
    """
    return {
        "success": True,
        "message": "Logged out successfully",
        "user_id": user["sub"]
    }


# =============================================================================
# ROUTES: PROFILE (Authenticated)
# =============================================================================

@app.get("/profile")
@app.get("/api/profile")
async def get_medical_profile(user: dict = Depends(get_current_user)):
    """Get authenticated user's medical profile with dynamic calorie targets."""
    raw_profile = MedicalProfileRepository.get_by_user_id(user["sub"])
    
    if not raw_profile:
        return {"message": "No profile found", "profile": None}
    
    try:
        # Wire up dynamic calculations
        profile_obj = UserProfile.from_dict(raw_profile)
        
        # Consolidate biometrics across history for dashboard visibility and TDEE calc
        consolidated = MedicalProfileRepository.get_consolidated_biometrics(user["sub"])
        for field, val in consolidated.items():
            if val is not None and getattr(profile_obj, field, None) in [None, "--", 0, ""]:
                setattr(profile_obj, field, val)
                
        tdee = profile_obj.calculate_tdee()
        
        if tdee and tdee > 1000: # Sanity check
            # Update targets based on Activity Level & Goal
            # 1. Calc base target
            target_calories = int(tdee)
            
            # Adjust for goal (UserProfile.fitness_goal)
            goal = (profile_obj.fitness_goal or "").lower()
            if "loss" in goal or "lose" in goal:
                target_calories -= 500
            elif "gain" in goal or "build" in goal:
                target_calories += 300
            
            # Safety checks
            if target_calories < 1200: target_calories = 1200
            
            # 2. Update the profile's daily targets
            # Use specific factory methods if conditions exist, otherwise generic
            if profile_obj.has_condition(HealthCondition.DIABETES):
                profile_obj.daily_targets = DailyTargets.for_diabetes(base_calories=target_calories)
            elif profile_obj.has_condition(HealthCondition.HYPERTENSION):
                profile_obj.daily_targets = DailyTargets.for_hypertension(base_calories=target_calories)
            elif "loss" in goal or profile_obj.has_condition(HealthCondition.OBESITY):
                profile_obj.daily_targets = DailyTargets.for_weight_loss(base_calories=target_calories)
            else:
                # Manual macro distribution for maintenance/general
                # 50% Carb, 20% Protein, 30% Fat
                profile_obj.daily_targets = DailyTargets(
                    calories=target_calories,
                    protein_g=int(target_calories * 0.20 / 4),
                    carbs_g=int(target_calories * 0.50 / 4),
                    fat_g=int(target_calories * 0.30 / 9),
                    # Keep default recommendations for others
                )

        # Merge back into response
        response = raw_profile.copy()
        response.update(profile_obj.to_dict())
        response['daily_targets'] = asdict(profile_obj.daily_targets) if hasattr(profile_obj.daily_targets, '__dataclass_fields__') else profile_obj.daily_targets.__dict__
        
        # Calculate BMI
        bmi = "--"
        if profile_obj.weight_kg and profile_obj.height_cm and profile_obj.height_cm > 0:
            h = profile_obj.height_cm / 100
            bmi = round(profile_obj.weight_kg / (h * h), 1)

        # Ensure bio_metrics block exists for frontend (standardized response)
        response['bio_metrics'] = {
            "age": profile_obj.age if profile_obj.age not in [None, "", 0] else "--",
            "weight_kg": profile_obj.weight_kg if profile_obj.weight_kg not in [None, "", 0] else "--",
            "height_cm": profile_obj.height_cm if profile_obj.height_cm not in [None, "", 0] else "--",
            "bmi": bmi if bmi not in [None, "", 0] else "--",
            "activity_level": profile_obj.activity_level.value if hasattr(profile_obj.activity_level, 'value') else str(profile_obj.activity_level),
            "fitness_goal": profile_obj.fitness_goal or "--",
            "gender": profile_obj.gender or "--"
        }
        
        # KEY FIX: Dashboard expects 'allergies' but model has 'allergens'
        response['allergies'] = response.get('allergens', [])
        response['conditions'] = response.get('conditions', [])

        # Reconstruct vitals from daily_targets
        raw_targets = raw_profile.get("daily_targets", {})
        response['vitals'] = {
            "glucose_level": raw_targets.get("glucose_level") or raw_targets.get("sugar_level") or "--",
            "cholesterol": raw_targets.get("cholesterol") or "--",
        }

        return response

    except Exception as e:
        print(f"[PROFILE] Error calculating dynamic targets: {e}")
        # Build minimal bio_metrics even on error if possible
        if raw_profile:
            raw_profile['bio_metrics'] = {
                "age": raw_profile.get('age') or "--",
                "weight_kg": raw_profile.get('weight_kg') or "--",
                "height_cm": raw_profile.get('height_cm') or "--",
                "bmi": "--",
                "gender": raw_profile.get('gender', '--')
            }
        return raw_profile


# =============================================================================
# ROUTES: UPLOAD & OCR
# =============================================================================

@app.post("/upload/medical-report")
async def upload_medical_report_legacy(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Upload medical report, run OCR, and create profile.
    
    Accepts PDF or image files.
    """
    # Validate file type
    allowed_types = ["application/pdf", "image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: PDF, JPG, PNG"
        )
    
    # Create upload record
    upload_id = str(uuid.uuid4())
    UploadRepository.create(upload_id, user["sub"], file.filename, file.content_type)
    
    try:
        # Save file temporarily
        content = await file.read()
        
        # Check file size (max 10MB)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Run OCR (use existing OCR parser if available, or simulate)
            ocr_text = ""
            conditions = []
            allergens = []
            
            # Try to use existing OCR
            try:
                from ocr.parser import parse_medical_report
                ocr_result = parse_medical_report(tmp_path)
                ocr_text = ocr_result.get("raw_text", "")
                conditions = ocr_result.get("conditions", [])
                allergens = ocr_result.get("allergens", [])
                vitals = ocr_result.get("vitals", {})  # Extract vitals!
            except ImportError:
                # Fallback: Extract basic info using simple text patterns
                # In real app, integrate actual OCR library like Tesseract
                
                # For demo: simulate extraction based on common patterns
                # This would be replaced with real OCR in production
                ocr_text = f"Processed file: {file.filename}"
                
                # Simulate some extraction
                conditions = ["general_checkup"]
                allergens = []
                vitals = {}
            
            # Update upload status
            UploadRepository.update_status(upload_id, "completed")
            
            # Create medical profile with vitals in daily_targets
            profile_id = str(uuid.uuid4())
            MedicalProfileRepository.create(
                profile_id=profile_id,
                user_id=user["sub"],
                conditions=conditions,
                allergens=allergens,
                daily_targets=vitals,  # Save vitals here!
                raw_ocr_text=ocr_text,
                source_file=file.filename,
            )
            
            return {
                "status": "success",
                "upload_id": upload_id,
                "profile": {
                    "id": profile_id,
                    "conditions": conditions,
                    "allergens": allergens,
                    "vitals": vitals,  # Return vitals in response
                },
                "message": "Medical profile created successfully",
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        UploadRepository.update_status(upload_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# =============================================================================
# ROUTES: PROFILE COMPLETION
# =============================================================================

class CompleteProfileRequest(BaseModel):
    age: int
    gender: str
    weight_kg: float
    height_cm: float
    activity_level: str
    fitness_goal: str


@app.post("/api/user/complete-profile")
async def complete_profile(
    request: CompleteProfileRequest,
    user: dict = Depends(get_current_user)
):
    """
    Complete user profile with bio-metrics.
    """
    user_id = user["sub"]
    
    # Check if profile exists, if not create one
    existing_profile = MedicalProfileRepository.get_by_user_id(user_id)
    
    if existing_profile:
        # Update existing
        success = MedicalProfileRepository.update(
            user_id=user_id,
            age=request.age,
            gender=request.gender,
            weight_kg=request.weight_kg,
            height_cm=request.height_cm,
            activity_level=request.activity_level,
            fitness_goal=request.fitness_goal
        )
    else:
        # Create new
        success = MedicalProfileRepository.create(
            profile_id=str(uuid.uuid4()),
            user_id=user_id,
            conditions=[],
            allergens=[],
            age=request.age,
            gender=request.gender,
            weight_kg=request.weight_kg,
            height_cm=request.height_cm,
            activity_level=request.activity_level,
            fitness_goal=request.fitness_goal
        )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save profile")
        
    return {"status": "success", "message": "Profile completed successfully"}


# =============================================================================
# ROUTES: FOOD IMAGE UPLOAD
# =============================================================================

# Load food database from Indian_Continental_Nutrition_With_Dal_Variants.csv
def _load_food_database():
    """Load food database from CSV for nutrition lookups."""
    import csv
    food_db = {}
    
    # Try Indian food nutrition dataset first
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "food_nutrition_with_serving_category (1).csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "FINAL_ACCURATE_FOOD_DATASET_WITH_CUISINE (1).csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "Indian_Continental_Nutrition_With_Dal_Variants.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "healthy_eating_dataset.csv")
    
    # Fallback defaults
    food_db["default"] = {"calories": 200, "protein_g": 8, "carbs_g": 25, "fat_g": 8, "sugar_g": 5}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support multiple formats
                dish_name = row.get('Dish Name') or row.get('food name') or row.get('meal_name') or row.get('name', '')
                if dish_name:
                    dish_name = dish_name.strip()
                    # Extract food type keywords for fallback matches
                    keywords = ['pasta', 'rice', 'salad', 'soup', 'curry', 'dal', 'roti', 'paratha', 'dosa', 'idli', 'biryani', 'pulao', 'kheer', 'halwa', 'sandwich', 'chapati', 'paneer', 'samosa', 'pakora', 'chicken', 'kebab', 'tikka', 'poha', 'vada', 'burger', 'pizza']
                    
                    # Standardized extraction
                    nutrition_data = {
                        "calories": float(row.get('calories (kcal)') or row.get('Calories (kcal)') or row.get('calories', 0) or 0),
                        "protein_g": float(row.get('protein (g)') or row.get('Protein (g)') or row.get('protein_g', 0) or 0),
                        "carbs_g": float(row.get('carbohydrates (g)') or row.get('Carbohydrates (g)') or row.get('carbs_g', 0) or 0),
                        "fat_g": float(row.get('fats (g)') or row.get('Fats (g)') or row.get('fat_g', 0) or 0),
                        "sugar_g": float(row.get('free sugar (g)') or row.get('Free Sugar (g)') or row.get('sugar_g', 0) or 0),
                        "sodium_mg": float(row.get('sodium (mg)') or row.get('Sodium (mg)') or row.get('sodium_mg', 0) or 0),
                        "fiber_g": float(row.get('fibre (g)') or row.get('Fibre (g)') or row.get('fiber_g', 0) or 0),
                        "density": float(row.get('Density (g/cm3)') or row.get('density', 1.0) or 1.0),
                    }
                    
                    for keyword in keywords:
                        if keyword in dish_name.lower():
                            if keyword not in food_db:
                                food_db[keyword] = nutrition_data
                    
                    # Also add full dish name for exact matches
                    food_db[dish_name.lower()] = nutrition_data
        print(f"[FOOD DB] Loaded {len(food_db)} food entries from {os.path.basename(csv_path)}")
    except Exception as e:
        print(f"[FOOD DB] Warning: Could not load CSV, using defaults: {e}")
        # Add fallback entries
        food_db.update({
            "apple": {"calories": 95, "protein_g": 0.5, "carbs_g": 25, "fat_g": 0.3, "sugar_g": 19},
            "banana": {"calories": 105, "protein_g": 1.3, "carbs_g": 27, "fat_g": 0.4, "sugar_g": 14},
            "pizza": {"calories": 285, "protein_g": 12, "carbs_g": 36, "fat_g": 10, "sugar_g": 4},
        })
    
    return food_db

FOOD_DATABASE = _load_food_database()


@app.post("/api/food/scan")
async def upload_food_image(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """
    Upload food image, run OCR, and return nutrition data.
    """
    user_id = user["sub"]
    
    print(f"\n[FOOD UPLOAD] ====== Starting food image analysis ======")
    print(f"[FOOD UPLOAD] User: {user.get('sub')}")
    print(f"[FOOD UPLOAD] File: {file.filename} ({file.content_type})")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        content = await file.read()
        
        # Check file size
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        print(f"[FOOD UPLOAD] File size: {len(content) / 1024:.1f} KB")
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"[FOOD UPLOAD] Saved to temp: {tmp_path}")
        
        try:
            # Use food recognition service for image analysis
            food_name = "Unknown Food"
            nutrition = None
            source = "estimated"
            confidence = 0.5
            
            
            
            # =========================================================================
            # =========================================================================
            # CONTINENTAL RETRIEVAL SYSTEM (CLIP)
            # =========================================================================
            try:
                print(f"[FOOD UPLOAD] Calling CONTINENTAL retrieval pipeline...")
                
                # Run retrieval (Top-5)
                # Note: main_inference handles PIL conversion and logic internally
                retrieval_result = continental_system.main_inference(tmp_path, k=5)
                
                # 1. RESOLUTION HIERARCHY: Specific Dish -> Food Group -> Unknown
                # Start with best match
                top_match = retrieval_result["top_k_predictions"][0]
                final_food_name = top_match["dish"]
                final_confidence = top_match["score"]
                
                print(f"[FOOD UPLOAD] ✓ Top Match: {final_food_name} ({final_confidence:.3f})")
                
                # Get nutrition if available
                # 1. Try exact/fuzzy match in global registry (Central source)
                from services.nutrition_registry import get_nutrition_registry
                registry = get_nutrition_registry()
                lookup = registry.get_by_name(final_food_name)
                
                # 2. Try match in local CSV database (Backward compatibility)
                if not lookup:
                    lookup = FOOD_DATABASE.get(final_food_name.lower())
                
                # 3. Keyword/Recursive fallback
                if not lookup:
                    # Generic keyword fallback for common dish types
                    common_keywords = ['dal', 'curry', 'roti', 'paratha', 'rice', 'sandwich', 'pasta', 'salad', 'samosa', 'paneer', 'chicken', 'kebab', 'tikka', 'poha', 'vada', 'burger', 'pizza', 'dosa', 'idli']
                    name_lower = final_food_name.lower()
                    for kw in common_keywords:
                        if kw in name_lower:
                            # Try registry again with just the keyword
                            lookup = registry.get_by_name(kw) or FOOD_DATABASE.get(kw)
                            if lookup:
                                print(f"[FOOD UPLOAD] ⚠ Falling back to keyword match: {kw}")
                                break

                nutrition = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "sugar_g": 0, "fiber_g": 0, "sodium_mg": 0}
                if lookup:
                    nutrition = lookup
                    print(f"[FOOD UPLOAD] ✓ Nutrition found for {final_food_name}: {nutrition.get('calories')} kcal")
                else:
                    print(f"[FOOD UPLOAD] ❌ No nutrition found for {final_food_name}")

                # Build final response
                
                # UPDATE CONTEXT FOR AI COACH & LOGGING
                # This ensures the Chat and Fix Meal features know about this food
                meal_data = {
                    "food_name": final_food_name,
                    "food_group": "Continental",
                    "resolution": "retrieval",
                    "nutrition": nutrition.copy(),
                    "timestamp": datetime.now().isoformat(),
                    "source": "continental_retrieval",
                    "confidence": final_confidence,
                }
                
                # Save to user session (Persistent DB)
                # Note: MealRepository.create() already updates daily_logs internally
                MealRepository.create(
                    user_id=user_id,
                    food_name=meal_data["food_name"],
                    nutrition=meal_data["nutrition"],
                    source=meal_data["source"],
                    confidence=meal_data["confidence"]
                )
                
                # Also keep in-memory for immediate context if needed (optional)
                if user_id not in user_meal_logs:
                    user_meal_logs[user_id] = []
                user_meal_logs[user_id].append(meal_data)
                
                # Update GLOBAL context for AI Coach
                current_food_context[user_id] = meal_data
                print(f"[FOOD UPLOAD] ✓ Context updated for user {user_id}")

                # Build final response
                
                # INTEGRATE RULE ENGINE FOR REAL-TIME SAFETY VERDICT
                user_profile = get_user_profile_for_rules(user_id)
                from rules.engine import Severity
                
                # Create Food object for evaluation
                eval_food = Food(
                    food_id=f"scan-{datetime.now().timestamp()}",
                    name=final_food_name,
                    serving_size=100.0,
                    serving_unit="g",
                    nutrition=NutritionInfo(
                        calories=float(nutrition.get("calories", 0)),
                        protein_g=float(nutrition.get("protein_g", 0)),
                        carbs_g=float(nutrition.get("carbs_g", 0)),
                        fat_g=float(nutrition.get("fat_g", 0)),
                        sugar_g=float(nutrition.get("sugar_g", 0)),
                        fiber_g=float(nutrition.get("fiber_g", 0)),
                        sodium_mg=float(nutrition.get("sodium_mg", 0)),
                    ),
                    allergens=[] # registry handles fuzzy allergen checks via conditions
                )
                
                rule_violations = rule_engine.evaluate(eval_food, user_profile)
                verdict = rule_engine.get_final_verdict(rule_violations)
                
                # Map verdict to frontend labels
                safety_label = "safe"
                if verdict == Severity.BLOCK:
                    safety_label = "danger" # Frontend uses 'danger' for red badge
                elif verdict in [Severity.WARN, Severity.ALERT]:
                    safety_label = "warning" # Frontend uses 'warning' for yellow/caution badge
                
                # Build formatted response for frontend (food-scan.html)
                print(f"[FOOD UPLOAD] SUCCESS: {final_food_name} ({final_confidence:.2f}) - Verdict: {safety_label}")
                
                return {
                    "status": "success",
                    "food_name": final_food_name,
                    "confidence": final_confidence,
                    "resolution_type": "retrieval",
                    "safety": {
                        "verdict": safety_label,
                        "violations": [
                            {"rule_id": v.rule_id, "message": v.message, "severity": v.severity.value} 
                            for v in rule_violations
                        ]
                    },
                    "nutrition": {
                        "calories": float(nutrition.get("calories", 0)),
                        "protein_g": float(nutrition.get("protein_g", 0)),
                        "carbs_g": float(nutrition.get("carbs_g", 0)),
                        "fat_g": float(nutrition.get("fat_g", 0)),
                        "sugar_g": float(nutrition.get("sugar_g", 0)),
                        "fiber_g": float(nutrition.get("fiber_g", 0)),
                        "sodium_mg": float(nutrition.get("sodium_mg", 0)),
                    },
                    "meta": {
                        "top_k": retrieval_result["top_k_predictions"]
                    }
                }
                    
            except Exception as e:
                print(f"[FOOD UPLOAD] ERROR: Retrieval failed: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# =============================================================================
# ROUTES: MEALS API
# =============================================================================

@app.get("/api/meals/today")
async def get_today_meals(date: Optional[str] = None, user: dict = Depends(get_current_user)):
    """
    Get all meals logged on a specific date (defaults to today).
    """
    user_id = user["sub"]
    target_date = date or datetime.now().strftime("%Y-%m-%d")
    
    # Get persistent meals from DB
    meals = MealRepository.get_meals_by_date(user_id, target_date)
    
    return {
        "meals": meals,
        "count": len(meals),
        "date": target_date,
    }


@app.delete("/api/meals/clear")
async def clear_meals(user: dict = Depends(get_current_user)):
    """Clear all meals for the current user (for testing)."""
    user_id = user["sub"]
    if user_id in user_meal_logs:
        user_meal_logs[user_id] = []
    return {"message": "Meals cleared"}


@app.get("/api/daily-stats")
async def get_daily_stats(date: Optional[str] = None, user: dict = Depends(get_current_user)):
    """Fetch aggregated daily stats for a specific date (defaults to today)."""
    user_id = user["sub"]
    date_str = date or datetime.now().strftime("%Y-%m-%d")
    
    stats = DailyLogRepository.get_or_create(user_id, date_str)
    
    # Matching dashboard values exactly
    return {
        "calories_consumed": stats.get("calories_consumed", 0),
        "calories_burned": stats.get("calories_burned", 0),
        "calories_target": stats.get("calories_target", 2000),
        "protein_g": stats.get("protein_g", 0),
        "carbs_g": stats.get("carbs_g", 0),
        "fat_g": stats.get("fat_g", 0),
        "water_cups": stats.get("water_cups", 0),
        "water_target": stats.get("water_target", 8),
        "date": date_str
    }


@app.post("/api/water/update")
async def update_water(delta: int = Body(..., embed=True), user: dict = Depends(get_current_user)):
    """Increment or decrement water intake."""
    user_id = user["sub"]
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    DailyLogRepository.update_water(user_id, date_str, delta)
    stats = DailyLogRepository.get_or_create(user_id, date_str)
    
    return {"status": "success", "water_cups": stats["water_cups"]}


@app.post("/api/exercise/log")
async def log_exercise(calories: float = Body(..., embed=True), user: dict = Depends(get_current_user)):
    """Record burned calories."""
    user_id = user["sub"]
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    DailyLogRepository.log_exercise(user_id, date_str, calories)
    stats = DailyLogRepository.get_or_create(user_id, date_str)
    
    return {"status": "success", "calories_burned": stats["calories_burned"]}


@app.get("/api/debug/dump")
async def debug_dump_meals(user: dict = Depends(get_current_user)):
    """Dump ALL meals for debugging."""
    user_id = user["sub"]
    
    # Import here to avoid circulars if any, though nicely imported at top
    from auth.database import DB_PATH
    
    debug_info = {
        "user_id": user_id,
        "db_path_absolute": os.path.abspath(DB_PATH),
        "db_exists": os.path.exists(DB_PATH),
        "server_time": datetime.now().isoformat()
    }

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM meal_logs WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
        rows = [dict(r) for r in cursor.fetchall()]
        for r in rows:
            try:
                r['nutrition'] = json.loads(r['nutrition']) if r['nutrition'] else {}
            except: pass
            
    return {"count": len(rows), "rows": rows, "debug_info": debug_info}

@app.get("/api/food/current")
async def get_current_food(user: dict = Depends(get_current_user)):
    """
    Get the most recently scanned food for immediate AI coach context.
    
    This is THE PRIMARY endpoint the AI coach should use.
    Returns the food that was just analyzed, with full nutrition data.
    """
    user_id = user["sub"]
    
    if user_id not in current_food_context:
        return {
            "has_food": False,
            "message": "No food scanned yet. Upload a food image first."
        }
    
    food_data = current_food_context[user_id]
    
    return {
        "has_food": True,
        "food": {
            "name": food_data["food_name"],
            "calories": food_data["nutrition"]["calories"],
            "protein_g": food_data["nutrition"]["protein_g"],
            "carbs_g": food_data["nutrition"]["carbs_g"],
            "fat_g": food_data["nutrition"]["fat_g"],
            "sugar_g": food_data["nutrition"]["sugar_g"],
            "fiber_g": food_data["nutrition"]["fiber_g"],
            "sodium_mg": food_data["nutrition"]["sodium_mg"],
        },
        "source": food_data["source"],
        "confidence": food_data["confidence"],
        "scanned_at": food_data["timestamp"]
    }

# =============================================================================
# ROUTES: RECIPE GENERATION
# =============================================================================

class RecipeRequest(BaseModel):
    ingredients: Optional[List[str]] = []


@app.post("/api/recipe/generate")
async def generate_recipe(
    request: RecipeRequest,
    user: dict = Depends(get_current_user),
):
    """
    Generate a recipe from ingredients.
    """
    user_id = user["sub"]
    
    if not request.ingredients:
        raise HTTPException(status_code=400, detail="No ingredients provided")
    
    # Get nutrition for each ingredient
    total_nutrition = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}
    
    for ingredient in request.ingredients:
        # Check against food database
        ingredient_lower = ingredient.lower()
        for food, data in FOOD_DATABASE.items():
            if food in ingredient_lower:
                total_nutrition["calories"] += data.get("calories", 0)
                total_nutrition["protein_g"] += data.get("protein_g", 0)
                total_nutrition["carbs_g"] += data.get("carbs_g", 0)
                total_nutrition["fat_g"] += data.get("fat_g", 0)
                break
    
    rag_service = get_rag_service()
    profile = rag_service.get_medical_profile(user_id)
    
    # Try AI-powered generation first
    llm_service = get_llm_service()
    if llm_service.is_available:
        print(f"[RECIPE] Generating AI recipe for ingredients: {request.ingredients}")
        llm_response = llm_service.generate_recipe(request.ingredients, profile)
        
        if llm_response.success:
            content = llm_response.content.strip()
            
            # --- Robust Markdown/JSON Parser ---
            recipe_name = "AI-Generated Healthy Dish"
            ingredients_list = request.ingredients
            instructions_list = []
            nutrition_data = total_nutrition.copy()
            health_note = "Recipe tailored to your health profile."
            
            # 1. Try JSON parsing as fallback
            try:
                # Find JSON block
                json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                    recipe_name = (data.get("name") or data.get("recipe_name") or recipe_name).replace('**', '').strip()
                    
                    raw_ingredients = data.get("ingredients") or []
                    ingredients_list = [re.sub(r'^[ \-*•◦∙]+', '', str(i)).replace('**', '').strip() for i in raw_ingredients if str(i).strip()]
                    
                    raw_instructions = data.get("instructions") or data.get("steps") or []
                    instructions_list = [re.sub(r'^([ \-*•◦∙]+|\d+[\.\)\:]\s*)', '', str(i)).replace('**', '').strip() for i in raw_instructions if str(i).strip()]
                    
                    nutrition_data = data.get("nutrition") or nutrition_data
                    health_note = (data.get("health_note") or health_note).replace('**', '').strip()
            except:
                pass

            # 2. Markdown Parsing (Extracting by headers)
            if not instructions_list:
                # Extract Title (# Title)
                title_match = re.search(r'^#\s*(.*)', content, re.MULTILINE)
                if title_match:
                    recipe_name = title_match.group(1).strip()
                
                # Split content into sections
                sections = re.split(r'\n##?\s+', content)
                
                for section in sections:
                    # Clean lines but don't strip leading digits globally (only for instructions)
                    lines = [l.strip() for l in section.split('\n') if l.strip()]
                    if not lines: continue
                    
                    header = lines[0].lower()
                    
                    if "ingredient" in header:
                        # For ingredients: Strip symbols and also markdown bold markers (**)
                        ingredients_list = [re.sub(r'^[ \-*•◦∙]+', '', l).replace('**', '').strip() for l in lines[1:] if l.strip()]
                    elif "instruction" in header or "preparation" in header or "method" in header or "step" in header:
                        # For instructions: Strip leading list markers AND markdown bold markers (**)
                        instructions_list = [re.sub(r'^([ \-*•◦∙]+|\d+[\.\)\:]\s*)', '', l).replace('**', '').strip() for l in lines[1:] if l.strip()]
                    elif "nutrition" in header:
                        # Extract basic metrics from lines
                        for line in lines[1:]:
                            line_lower = line.lower()
                            val_match = re.search(r'(\d+)', line)
                            if val_match:
                                val = float(val_match.group(1))
                                if "cal" in line_lower: nutrition_data["calories"] = val
                                elif "protein" in line_lower: nutrition_data["protein_g"] = val
                                elif "carb" in line_lower: nutrition_data["carbs_g"] = val
                                elif "fat" in line_lower: nutrition_data["fat_g"] = val
                
                # Health Note (extract intro paragraph)
                if "##" in content:
                    intro = content.split("##")[0].split("\n")
                    if len(intro) > 1:
                        health_note = " ".join([l.strip() for l in intro[1:] if l.strip()])
            
            # Final fallback: if still no instructions, use raw lines
            if not instructions_list:
                instructions_list = [l.strip() for l in content.split('\n') if l.strip()]

            return {
                "status": "success",
                "recipe_name": recipe_name,
                "ingredients": ingredients_list,
                "instructions": instructions_list,
                "nutrition": nutrition_data,
                "serves": 2,
                "powered_by": "ollama_gemma",
                "health_note": health_note
            }
        else:
            print(f"[RECIPE] AI generation failed, falling back: {llm_response.error}")

    # Fallback to simple template-based recipe
    recipe_name = f"Healthy {request.ingredients[0].split()[-1].title()} Dish"
    
    instructions = [
        f"Prepare all ingredients: {', '.join(request.ingredients[:3])}",
        "Heat a pan with a little olive oil over medium heat",
        "Add proteins first and cook until done (5-7 minutes)",
        "Add vegetables and stir-fry for 3-4 minutes",
        "Season with salt, pepper, and your preferred herbs",
        "Serve hot and enjoy your healthy meal!"
    ]
    
    return {
        "status": "success",
        "recipe_name": recipe_name,
        "ingredients": request.ingredients,
        "instructions": instructions,
        "nutrition": total_nutrition,
        "serves": 2,
        "fallback_mode": True
    }


# (Duplicate /api/meal/fix removed - the correct endpoint is defined above in the ROUTES: FIX MEAL section)

# =============================================================================
# ROUTES: ANALYTICS
# =============================================================================

@app.get("/api/analytics/summary")
async def get_analytics_summary(user: dict = Depends(get_current_user)):
    """
    Get dynamic analytics summary for user based on actual logs.
    """
    user_id = user["sub"]
    
    # Get today's stats
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_stats = DailyLogRepository.get_or_create(user_id, today_str)
    
    # Get last 7 days stats for weekly totals and trends
    weekly_logs = []
    end_date = datetime.now()
    for i in range(7):
        date_str = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
        log = DailyLogRepository.get_or_create(user_id, date_str)
        weekly_logs.append(log)
    
    # Calculate Weekly Metrics
    total_actual_cals = sum(l.get("calories_consumed", 0) for l in weekly_logs)
    total_target_cals = sum(l.get("calories_target", 2000) for l in weekly_logs)
    
    # Health Score Calculation (Simplified)
    # Start with 100 and deduct for missed targets or imbalances
    adherence_penalty = 0
    days_with_logs = 0
    for log in weekly_logs:
        if log.get("calories_consumed", 0) > 0:
            days_with_logs += 1
            target = log.get("calories_target", 2000)
            dev = abs(log["calories_consumed"] - target) / target
            adherence_penalty += dev * 20 # Max 20 points penalty per day for 100% deviation
            
    avg_penalty = adherence_penalty / days_with_logs if days_with_logs > 0 else 25
    health_score = max(0, min(100, int(100 - avg_penalty)))
    
    # Grade mapping
    if health_score >= 90: grade = "A+"
    elif health_score >= 80: grade = "A"
    elif health_score >= 70: grade = "B"
    elif health_score >= 60: grade = "C"
    else: grade = "D"
    
    # Macro Balance (Based on Today)
    p = today_stats.get("protein_g", 0) * 4
    c = today_stats.get("carbs_g", 0) * 4
    f = today_stats.get("fat_g", 0) * 9
    total_macro_kcal = (p + c + f) or 1
    
    # Trends (Comparing last 3 days to previous 4 days)
    recent_avg = sum(l.get("calories_consumed", 0) for l in weekly_logs[:3]) / 3
    older_avg = sum(l.get("calories_consumed", 0) for l in weekly_logs[3:]) / 4
    
    cal_trend = "stable"
    if recent_avg < older_avg * 0.9: cal_trend = "decreasing"
    elif recent_avg > older_avg * 1.1: cal_trend = "increasing"

    return {
        "health_score": health_score,
        "health_grade": grade,
        "weekly_calories": {
            "target": int(total_target_cals),
            "actual": int(total_actual_cals),
            "status": "good" if total_actual_cals <= total_target_cals * 1.05 else "over"
        },
        "macro_balance": {
            "protein": {"target": 25, "actual": int(p / total_macro_kcal * 100)},
            "carbs": {"target": 50, "actual": int(c / total_macro_kcal * 100)},
            "fat": {"target": 25, "actual": int(f / total_macro_kcal * 100)}
        },
        "streaks": {
            "logging": days_with_logs,
            "safe_meals": max(0, days_with_logs * 2 - 1) # Approximation
        },
        "trends": {
            "calories": cal_trend,
            "protein": "stable",
            "sugar": "decreasing" if health_score > 75 else "stable"
        }
    }


@app.get("/api/analytics/weight-forecast")
async def get_weight_forecast(user: dict = Depends(get_current_user)):
    """
    Predict 30-day weight forecast based on user's current metrics.
    """
    user_id = user["sub"]
    
    # Get user profile for current weight and activity level
    profile = MedicalProfileRepository.get_by_user_id(user_id)
    if not profile or not profile.get("weight_kg"):
        # Use defaults if profile incomplete
        current_weight = 75.0
        activity_level = 1.2
    else:
        current_weight = float(profile.get("weight_kg"))
        activity_level_map = {
            "sedentary": 1.2,
            "lightly_active": 1.375,
            "moderately_active": 1.55,
            "very_active": 1.725,
            "extra_active": 1.9
        }
        activity_level = activity_level_map.get(profile.get("activity_level", "sedentary"), 1.2)
    
    # Get recent calorie delta from daily logs
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6)
    
    total_delta = 0
    days_with_data = 0
    
    for i in range(7):
        date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        log = DailyLogRepository.get_or_create(user_id, date_str)
        if log.get("calories_consumed", 0) > 0:
            # Delta = Consumed - (Target + Burned)
            delta = log["calories_consumed"] - (log["calories_target"] + log["calories_burned"])
            total_delta += delta
            days_with_data += 1
            
    avg_delta = total_delta / days_with_data if days_with_data > 0 else 0
    
    # Run prediction
    forecast = weight_forecaster.predict_30_day_forecast(
        current_weight=current_weight,
        avg_daily_delta=avg_delta,
        activity_level=activity_level
    )
    
    return {
        "forecast": forecast,
        "input_summary": {
            "avg_daily_delta": round(avg_delta, 1),
            "activity_level": activity_level,
            "days_analyzed": days_with_data
        }
    }


# =============================================================================
# ROUTES: MEAL PLANNING & GROCERY LISTS
# =============================================================================

from auth.database import WeeklyPlanRepository
meal_planner = WeeklyPlanGenerator()
grocery_gen = GroceryGenerator()

@app.post("/api/meal-plan/generate")
async def generate_meal_plan(user: dict = Depends(get_current_user)):
    """Generate a new weekly meal plan for the user."""
    user_id = user["sub"]
    try:
        plan = meal_planner.generate_plan(user_id)
        return {"status": "success", "plan": plan}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"[API] Meal plan generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate meal plan")

@app.get("/api/meal-plan/current")
async def get_current_meal_plan(user: dict = Depends(get_current_user)):
    """Retrieve the latest weekly meal plan."""
    user_id = user["sub"]
    plan_record = WeeklyPlanRepository.get_latest(user_id)
    
    if not plan_record:
        return {"has_plan": False, "message": "No active meal plan found."}
    
    return {
        "has_plan": True, 
        "plan": plan_record["plan"],
        "start_date": plan_record["start_date"],
        "id": plan_record["id"]
    }

@app.get("/api/grocery-list")
async def get_grocery_list(user: dict = Depends(get_current_user)):
    """Generate a grocery list from the current meal plan."""
    user_id = user["sub"]
    plan_record = WeeklyPlanRepository.get_latest(user_id)
    
    if not plan_record:
        raise HTTPException(status_code=404, detail="No active meal plan found")
    
    grocery_list = grocery_gen.generate_list(plan_record["plan"])
    return {"status": "success", "grocery_list": grocery_list}


# =============================================================================
# ROUTES: ANALYTICS (EXTENDED)
# =============================================================================

@app.get("/api/analytics/meal-prediction")
async def get_meal_prediction(user: dict = Depends(get_current_user)):
    """Predict next meal time and type."""
    engine = AnalyticsEngine(user["sub"])
    prediction = engine.predict_next_meal()
    return prediction

@app.get("/api/analytics/calorie-budget")
async def get_calorie_budget(user: dict = Depends(get_current_user)):
    """Predict current calorie budget status."""
    engine = AnalyticsEngine(user["sub"])
    return engine.get_calorie_forecast()

@app.get("/api/analytics/weight-trajectory")
async def get_weight_trajectory(user: dict = Depends(get_current_user)):
    """Predict weight trajectory."""
    engine = AnalyticsEngine(user["sub"])
    return engine.predict_weight_trajectory()


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)