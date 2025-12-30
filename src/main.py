"""
FastAPI Application for AI Nutrition

Provides REST API endpoints for:
- Authentication (login, register)
- Virtual Coach (chat)
- Analytics (health score, trends, insights)
- Food logging
- Medical report upload with OCR
"""

# Load environment variables FIRST (before other imports that may need them)
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import sys
import uuid
import tempfile

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from models.food import Food, NutritionInfo, FoodCategory
from models.user import UserProfile, HealthCondition, DailyTargets, DailyIntake
from rules.engine import RuleEngine
from coach.virtual_coach import VirtualCoach
from analytics.analytics_service import AnalyticsService, MealLogStore
from feedback.feedback_service import FeedbackService, FeedbackStore
from auth.auth_service import auth_service
from auth.database import MedicalProfileRepository, UploadRepository
from services.llm_service import get_mistral_service, get_llm_service
from services.rag_service import get_rag_service


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="AI Nutrition API",
    description="Context-aware nutrition guidance with medical safety",
    version="3.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")


# =============================================================================
# GLOBAL STATE (In-memory for demo - use DB in production)
# =============================================================================

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
async def chat(request: ChatRequest):
    """Chat with the Virtual Coach using RAG + Mistral LLM."""
    try:
        user_id = "demo_user_123"  # Demo mode
        
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
        llm_service = get_llm_service()
        rag_service = get_rag_service()
        
        # Build comprehensive RAG context from user data
        rag_context = rag_service.build_context(
            user_id=user_id,
            meal_logs=user_meal_logs,
            current_food=food_data,
            user_question=request.message
        )
        print(f"[RAG] Built context: {len(rag_context)} chars")
        
        # Try LLM-powered response if available
        if llm_service.is_available:
            print(f"[Coach] LLM is available, calling chat...")
            llm_response = llm_service.chat(
                prompt=request.message,
                system_prompt="nutrition_coach",
                rag_context=rag_context
            )
            
            if llm_response.success:
                print(f"[Coach] Using Ollama/Gemma LLM with RAG context")
                
                # Get user profile data to show in response
                profile = rag_service.get_medical_profile(user_id)
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
                
                # Still run rule engine for safety checks
                violations = []
                safety_level = "safe"
                if food:
                    from rules.engine import RuleEngine
                    rule_engine = RuleEngine()
                    rule_violations = rule_engine.evaluate(food, demo_user)
                    violations = [v.to_dict() for v in rule_violations]
                    verdict = rule_engine.get_final_verdict(rule_violations)
                    safety_level = verdict.value
                
                return ChatResponse(
                    message=profile_header + llm_response.content,
                    safety_level=safety_level,
                    confidence=0.9,
                    violations=violations,
                    suggestions=[{"type": "llm_powered", "source": "ollama_gemma", "rag_enabled": True}],
                )
            else:
                print(f"[Coach] LLM response failed: {llm_response.error}")
        else:
            print(f"[Coach] LLM service not available, using fallback")
        
        # Fallback to original rule-based coach - STILL SHOW PROFILE DATA
        profile = rag_service.get_medical_profile(user_id)
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
async def exercise_chat(request: ExerciseRequest):
    """
    Exercise Guidance AI - provides general fitness advice.
    Uses Ollama/Gemma LLM for intelligent responses.
    Shows user health profile for personalized context.
    """
    user_id = "demo_user_123"
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
async def fix_meal(request: FixMealRequest, user: dict = None):
    """
    Clinical Nutrition AI - analyzes meal and suggests improvements.
    Uses detected food data and user medical profile.
    Does NOT ask for re-entry of data.
    """
    if not user:
        user = {"sub": "demo_user_123"}
    
    user_id = user["sub"]
    
    # Get food data - from request or auto-fetch from context
    food_name = request.food_name
    nutrition = request.nutrition
    
    if not food_name or not nutrition:
        # Auto-fetch from current_food_context
        if user_id in current_food_context:
            ctx = current_food_context[user_id]
            food_name = ctx.get("food_name", "Unknown Food")
            nutrition = ctx.get("nutrition", {})
        else:
            return {
                "verdict": "No Data",
                "message": "No food data available. Please scan a food item first.",
                "problems": [],
                "suggestions": {}
            }
    
    # Get user medical profile
    profile = MedicalProfileRepository.get_by_user_id(user_id)
    conditions = []
    if profile:
        conditions = profile.get("conditions", [])
    
    # Analyze the meal
    problems = []
    remove_items = []
    reduce_items = []
    replace_items = []
    improvements = {}
    
    # Extract nutrition values
    calories = nutrition.get("calories", 0)
    carbs = nutrition.get("carbs_g", 0)
    sugar = nutrition.get("sugar_g", 0)
    fat = nutrition.get("fat_g", 0)
    sodium = nutrition.get("sodium_mg", 0)
    protein = nutrition.get("protein_g", 0)
    fiber = nutrition.get("fiber_g", 0)
    
    # Check against conditions
    for condition in conditions:
        condition_lower = condition.lower()
        
        # Diabetes checks
        if "diabetes" in condition_lower:
            if sugar > 10:
                problems.append({
                    "ingredient": food_name,
                    "reason": f"High sugar ({sugar}g) can cause glucose spikes",
                    "condition": "Diabetes"
                })
                reduce_items.append(f"Sugar content by {sugar - 10}g")
                improvements["sugar_reduction"] = f"-{sugar - 10}g sugar"
            if carbs > 45:
                problems.append({
                    "ingredient": food_name,
                    "reason": f"High carbohydrates ({carbs}g) affects blood sugar",
                    "condition": "Diabetes"
                })
        
        # Hypertension checks
        if "hypertension" in condition_lower or "blood pressure" in condition_lower:
            if sodium > 500:
                problems.append({
                    "ingredient": food_name,
                    "reason": f"High sodium ({sodium}mg) raises blood pressure",
                    "condition": "Hypertension"
                })
                reduce_items.append(f"Sodium by {sodium - 500}mg")
                replace_items.append("Use herbs/spices instead of salt")
                improvements["sodium_reduction"] = f"-{sodium - 500}mg sodium"
        
        # Cholesterol checks
        if "cholesterol" in condition_lower:
            if fat > 15:
                problems.append({
                    "ingredient": food_name,
                    "reason": f"High fat content ({fat}g) affects cholesterol levels",
                    "condition": "High Cholesterol"
                })
                replace_items.append("Choose lean proteins or grilled options")
                improvements["fat_reduction"] = f"-{fat - 15}g fat"
        
        # Kidney disease checks
        if "kidney" in condition_lower:
            if sodium > 400:
                problems.append({
                    "ingredient": food_name,
                    "reason": f"High sodium ({sodium}mg) strains kidneys",
                    "condition": "Kidney Disease"
                })
            if protein > 25:
                problems.append({
                    "ingredient": food_name,
                    "reason": f"High protein ({protein}g) may strain kidney function",
                    "condition": "Kidney Disease"
                })
    
    # General health checks (for users without specific conditions)
    if not conditions or len(problems) == 0:
        if calories > 600:
            problems.append({
                "ingredient": food_name,
                "reason": f"High calorie meal ({calories} cal) - consider portion control",
                "condition": "General Health"
            })
            improvements["calorie_reduction"] = "Consider smaller portion"
        
        if sugar > 25:
            problems.append({
                "ingredient": food_name,
                "reason": f"High sugar content ({sugar}g) - exceeds recommended limits",
                "condition": "General Health"
            })
            replace_items.append("Choose unsweetened alternatives")
        
        if sodium > 800:
            problems.append({
                "ingredient": food_name,
                "reason": f"High sodium ({sodium}mg) - may affect blood pressure over time",
                "condition": "General Health"
            })
    
    # Determine verdict
    if len(problems) == 0:
        verdict = "Healthy"
        message = f"{food_name} appears suitable for your dietary needs."
        formatted_response = f"✅ This meal is suitable for your dietary needs.\n\nNo problematic ingredients detected for your conditions."
        llm_suggestions = None
    else:
        verdict = "Needs Fix"
        message = f"{food_name} has {len(problems)} issue(s) based on your health profile."
        
        # Try LLM-powered suggestions first with RAG context
        llm_suggestions = None
        llm_service = get_llm_service()
        rag_service = get_rag_service()
        
        if llm_service.is_available:
            # Build RAG context for personalized suggestions
            rag_context = rag_service.build_context(
                user_id=user_id,
                meal_logs=user_meal_logs,
                current_food=nutrition,
                user_question="How can I make this meal healthier?"
            )
            
            # Build detailed prompt for meal fixing
            problems_str = "\n".join(
                f"- {p.get('ingredient', 'Unknown')}: {p.get('reason', 'Unknown')} ({p.get('condition', '')})"
                for p in problems
            )
            
            fix_prompt = f"""Based on the user's health profile and the meal analysis, provide specific suggestions to fix this meal.

DETECTED PROBLEMS:
{problems_str}

Provide actionable suggestions in these categories:
1. REMOVE: Items to remove entirely
2. REDUCE: Items to reduce quantity
3. REPLACE: Healthier alternatives
4. ADD: Beneficial additions

Be specific and practical for someone with these health conditions."""
            
            llm_response = llm_service.chat(
                prompt=fix_prompt,
                system_prompt="meal_fixer",
                rag_context=rag_context
            )
            if llm_response.success:
                llm_suggestions = llm_response.content
        
        # Build formatted response with emoji template
        formatted_response = f"This meal is NOT suitable based on your health profile.\n\n"
        
        # Problematic items
        formatted_response += "❌ Problematic items:\n"
        for p in problems:
            formatted_response += f"  • {p['ingredient']} - {p['reason']} ({p['condition']})\n"
        
        # Use LLM suggestions if available, otherwise fallback to rule-based
        if llm_suggestions:
            formatted_response += f"\n🤖 AI-Powered Suggestions:\n{llm_suggestions}\n"
        else:
            # Fallback to rule-based suggestions
            # Fix suggestions
            formatted_response += "\n🔧 Fix suggestions:\n"
            if remove_items:
                formatted_response += "  REMOVE:\n"
                for item in remove_items:
                    formatted_response += f"    • {item}\n"
            if reduce_items:
                formatted_response += "  REDUCE:\n"
                for item in reduce_items:
                    formatted_response += f"    • {item}\n"
            if replace_items:
                formatted_response += "  REPLACE:\n"
                for item in replace_items:
                    formatted_response += f"    • {item}\n"
            
            # Healthier alternatives
            formatted_response += "\n✅ Healthier alternatives:\n"
            if "diabetes" in str(conditions).lower() and sugar > 10:
                formatted_response += "  • Choose whole fruits instead of sugary items\n"
                formatted_response += "  • Use stevia or monk fruit as sweetener\n"
            if "hypertension" in str(conditions).lower() and sodium > 500:
                formatted_response += "  • Use herbs, garlic, or lemon for flavor\n"
                formatted_response += "  • Choose fresh food over processed\n"
            if "cholesterol" in str(conditions).lower() and fat > 15:
                formatted_response += "  • Choose grilled or baked instead of fried\n"
                formatted_response += "  • Use olive oil instead of butter\n"
            if not any(k in str(conditions).lower() for k in ["diabetes", "hypertension", "cholesterol"]):
                formatted_response += "  • Choose smaller portion sizes\n"
                formatted_response += "  • Add more vegetables to balance the meal\n"
    
    return {
        "verdict": verdict,
        "message": message,
        "formatted_response": formatted_response,
        "food_name": food_name,
        "nutrition": nutrition,
        "problems": problems,
        "suggestions": {
            "remove": remove_items,
            "reduce": reduce_items,
            "replace_with": replace_items
        },
        "expected_improvements": improvements,
        "conditions_checked": conditions
    }


# =============================================================================
# ROUTES: ANALYTICS
# =============================================================================

@app.get("/api/analytics/score")
async def get_health_score():
    """Get current health score."""
    score = analytics_service.compute_health_score(
        demo_user.user_id,
        demo_user.daily_targets,
    )
    return score.to_dict()


@app.get("/api/analytics/trends")
async def get_trends(days: int = 7):
    """Get nutrient trends."""
    trends = analytics_service.compute_daily_trends(
        demo_user.user_id,
        demo_user.daily_targets,
        days=days,
    )
    return {k: v.to_dict() for k, v in trends.items()}


@app.get("/api/analytics/patterns")
async def get_patterns(days: int = 14):
    """Get detected behavior patterns."""
    patterns = analytics_service.detect_patterns(
        demo_user.user_id,
        demo_user.daily_targets,
        days=days,
    )
    return [p.to_dict() for p in patterns]


@app.get("/api/analytics/insight")
async def get_insight():
    """Get daily actionable insight."""
    insight = analytics_service.generate_insight(
        demo_user.user_id,
        demo_user.daily_targets,
    )
    return {"insight": insight}


@app.get("/api/analytics/snapshot")
async def get_snapshot():
    """Get complete analytics snapshot."""
    snapshot = analytics_service.get_snapshot(
        demo_user.user_id,
        demo_user.daily_targets,
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
async def analyze_food(food: FoodInput):
    """Analyze a food item for safety."""
    # Ensure carbs >= sugar + fiber (validation requirement)
    carbs_g = food.carbs_g
    if carbs_g < food.sugar_g + food.fiber_g:
        carbs_g = food.sugar_g + food.fiber_g
    
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
    
    violations = rule_engine.evaluate(food_obj, demo_user)
    verdict = rule_engine.get_final_verdict(violations)
    
    return {
        "food": food.name,
        "verdict": verdict.value,
        "violations": [v.to_dict() for v in violations],
        "formatted": rule_engine.format_violations(violations),
    }


@app.post("/api/food/log")
async def log_meal(request: LogMealRequest):
    """Log a meal."""
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
        user_id=demo_user.user_id,
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
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on a response."""
    feedback = feedback_service.submit_feedback(
        user_id=demo_user.user_id,
        context_type=request.context_type,
        context_id=request.context_id,
        rating=request.rating,
        comment=request.comment,
    )
    return {"feedback_id": feedback.feedback_id, "status": "submitted"}


# =============================================================================
# ROUTES: USER
# =============================================================================

@app.get("/api/profile")
async def get_profile(user: dict = None):
    """
    Get current user's medical profile from database.
    TEMP: Auth optional for testing.
    """
    # For testing: use demo user if no auth
    if not user:
        user = {"sub": "demo_user_123"}
    
    user_id = user["sub"]
    print(f"[API] Loading profile for user: {user_id}")
    
    # Get profile from database (from OCR extraction)
    profile = MedicalProfileRepository.get_by_user_id(user_id)
    
    if not profile:
        print(f"[API] No profile found for user: {user_id}")
        # Return empty profile instead of 404
        return {
            "user_id": user_id,
            "conditions": [],
            "allergies": [],
            "medications": [],
            "daily_targets": {},
            "vitals": {
                "glucose_level": None,
                "cholesterol": None
            },
            "source": "none",
            "message": "No medical profile yet. Upload a report."
        }
    
    print(f"[API] Found profile: conditions={profile.get('conditions')}, allergens={profile.get('allergens')}")
    
    # Extract vitals from daily_targets if available
    daily_targets = profile.get("daily_targets", {})
    vitals = {
        "glucose_level": daily_targets.get("glucose_level") or daily_targets.get("sugar_level"),
        "cholesterol": daily_targets.get("cholesterol"),
    }
    
    return {
        "user_id": user_id,
        "conditions": profile.get("conditions", []),
        "allergies": profile.get("allergens", []),
        "medications": profile.get("medications", []),
        "daily_targets": daily_targets,
        "vitals": vitals,
        "source": "medical_report_ocr",
        "source_file": profile.get("source_file"),
        "created_at": profile.get("created_at"),
    }


# Keep old endpoint for backward compatibility
@app.get("/api/user/profile")
async def get_user_profile_legacy(user: dict = Depends(get_current_user)):
    """Legacy endpoint - redirects to /api/profile"""
    return await get_profile(user)


# =============================================================================
# ROUTES: MEDICAL REPORT UPLOAD
# =============================================================================

@app.post("/api/medical-report/upload")
async def upload_medical_report(
    file: UploadFile = File(...),
    user: dict = None,  # TEMP: Auth optional for testing
):
    """
    Upload a medical report and extract conditions, allergens, and vitals.
    Uses OCR to parse the document.
    """
    if not user:
        user = {"sub": "demo_user_123"}
    
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
        
        print(f"[MEDICAL REPORT] Conditions: {result.get('conditions', [])}")
        print(f"[MEDICAL REPORT] Allergens: {result.get('allergens', [])}")
        print(f"[MEDICAL REPORT] Vitals: {result.get('vitals', {})}")
        
        # Build daily_targets with vitals
        daily_targets = result.get('vitals', {})
        
        # Save to database
        import uuid
        profile_id = str(uuid.uuid4())
        
        MedicalProfileRepository.create(
            profile_id=profile_id,
            user_id=user_id,
            conditions=result.get('conditions', []),
            allergens=result.get('allergens', []),
            medications=[],
            daily_targets=daily_targets,  # Contains vitals
            raw_ocr_text=result.get('raw_text', '')[:5000],  # Limit size
            source_file=file.filename,
        )
        
        print(f"[MEDICAL REPORT] ✓ Saved profile: {profile_id}")
        
        return {
            "success": True,
            "profile_id": profile_id,
            "conditions": result.get('conditions', []),
            "allergens": result.get('allergens', []),
            "vitals": result.get('vitals', {}),
            "message": f"Extracted {len(result.get('conditions', []))} conditions, {len(result.get('allergens', []))} allergens, and vitals from report."
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


# =============================================================================
# ROUTES: PROFILE (Authenticated)
# =============================================================================

@app.get("/profile")
async def get_medical_profile(user: dict = Depends(get_current_user)):
    """Get authenticated user's medical profile."""
    profile = MedicalProfileRepository.get_by_user_id(user["sub"])
    
    if not profile:
        return {"message": "No profile found", "profile": None}
    
    return profile


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
# ROUTES: FOOD IMAGE UPLOAD
# =============================================================================

# Load food database from Indian_Food_Nutrition_Processed.csv
def _load_food_database():
    """Load food database from CSV for nutrition lookups."""
    import csv
    food_db = {}
    
    # Try Indian food nutrition dataset first
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "Indian_Food_Nutrition_Processed.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "healthy_eating_dataset.csv")
    
    # Fallback defaults
    food_db["default"] = {"calories": 200, "protein_g": 8, "carbs_g": 25, "fat_g": 8, "sugar_g": 5}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support multiple formats
                dish_name = row.get('Dish Name') or row.get('meal_name', '')
                if dish_name:
                    # Extract food type keywords for fallback matches
                    keywords = ['pasta', 'rice', 'salad', 'soup', 'curry', 'dal', 'roti', 'paratha', 'dosa', 'idli', 'biryani', 'pulao', 'kheer', 'halwa', 'sandwich', 'chapati', 'paneer', 'samosa', 'pakora']
                    for keyword in keywords:
                        if keyword in dish_name.lower():
                            if keyword not in food_db:
                                food_db[keyword] = {
                                    "calories": float(row.get('Calories (kcal)', 0) or row.get('calories', 0) or 0),
                                    "protein_g": float(row.get('Protein (g)', 0) or row.get('protein_g', 0) or 0),
                                    "carbs_g": float(row.get('Carbohydrates (g)', 0) or row.get('carbs_g', 0) or 0),
                                    "fat_g": float(row.get('Fats (g)', 0) or row.get('fat_g', 0) or 0),
                                    "sugar_g": float(row.get('Free Sugar (g)', 0) or row.get('sugar_g', 0) or 0),
                                    "sodium_mg": float(row.get('Sodium (mg)', 0) or row.get('sodium_mg', 0) or 0),
                                    "fiber_g": float(row.get('Fibre (g)', 0) or row.get('fiber_g', 0) or 0),
                                }
                    
                    # Also add full dish name for exact matches
                    food_db[dish_name.lower()] = {
                        "calories": float(row.get('Calories (kcal)', 0) or row.get('calories', 0) or 0),
                        "protein_g": float(row.get('Protein (g)', 0) or row.get('protein_g', 0) or 0),
                        "carbs_g": float(row.get('Carbohydrates (g)', 0) or row.get('carbs_g', 0) or 0),
                        "fat_g": float(row.get('Fats (g)', 0) or row.get('fat_g', 0) or 0),
                        "sugar_g": float(row.get('Free Sugar (g)', 0) or row.get('sugar_g', 0) or 0),
                        "sodium_mg": float(row.get('Sodium (mg)', 0) or row.get('sodium_mg', 0) or 0),
                        "fiber_g": float(row.get('Fibre (g)', 0) or row.get('fiber_g', 0) or 0),
                    }
        print(f"[FOOD DB] Loaded {len(food_db)} food entries from Indian_Food_Nutrition_Processed.csv")
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


@app.post("/api/food/upload")
async def upload_food_image(
    file: UploadFile = File(...),
    user: dict = None,  # TEMP: Auth optional for testing
):
    """
    Upload food image, run OCR, and return nutrition data.
    
    Flow:
    1. Accept image
    2. Run OCR to detect food/nutrition label
    3. If nutrition found, use it
    4. If not, detect food name and use USDA fallback
    5. Apply medical safety rules
    6. Return structured data
    """
    # Use demo user if no auth (matches meals endpoint)
    if not user:
        user = {"sub": "demo_user_123"}
    
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
            
            print(f"[FOOD UPLOAD] Calling food recognition pipeline...")
            
            # STEP 1: Try YOLO food recognition first
            try:
                from services.yolo_service import get_yolo_recognizer
                yolo = get_yolo_recognizer()
                
                if yolo.is_available:
                    print(f"[FOOD UPLOAD] Using YOLO model for recognition...")
                    yolo_result = yolo.predict(tmp_path)
                    
                    if yolo_result.get("success") and yolo_result.get("food_name"):
                        food_name = yolo_result["food_name"]
                        confidence = yolo_result.get("confidence", 0.8)
                        source = "yolo_model"
                        print(f"[FOOD UPLOAD] ✓ YOLO detected: {food_name} ({confidence:.0%})")
                        
                        # Look up nutrition data for detected food
                        for db_food, data in FOOD_DATABASE.items():
                            if db_food.lower() in food_name.lower() or food_name.lower() in db_food.lower():
                                nutrition = data.copy()
                                print(f"[FOOD UPLOAD] ✓ Found nutrition for: {db_food}")
                                break
                    else:
                        print(f"[FOOD UPLOAD] YOLO: No food detected, trying OCR...")
                else:
                    print(f"[FOOD UPLOAD] YOLO model not available, trying OCR...")
            except ImportError as ie:
                print(f"[FOOD UPLOAD] YOLO import error (install ultralytics): {ie}")
            except Exception as e:
                print(f"[FOOD UPLOAD] YOLO error: {e}")
            
            # STEP 2: Try OCR if YOLO didn't find nutrition
            if not nutrition:
                try:
                    from ocr.food_recognition import parse_nutrition_label
                    ocr_result = parse_nutrition_label(tmp_path)
                    print(f"[FOOD UPLOAD] OCR result: {ocr_result}")
                    
                    if ocr_result and ocr_result.get("nutrition"):
                        nutrition = ocr_result["nutrition"]
                        food_name = ocr_result.get("food_name", food_name)
                        source = ocr_result.get("source", "database")
                        confidence = ocr_result.get("confidence", 0.7)
                        print(f"[FOOD UPLOAD] ✓ OCR recognized: {food_name} via {source} ({confidence:.0%})")
                    else:
                        print(f"[FOOD UPLOAD] ✗ No nutrition data from OCR")
                except Exception as e:
                    print(f"[FOOD UPLOAD] ✗ OCR error: {e}")
            
            # Fallback: detect food from filename or use default
            if not nutrition:
                # Try to match food name from filename
                filename_lower = file.filename.lower() if file.filename else ""
                for food, data in FOOD_DATABASE.items():
                    if food in filename_lower:
                        food_name = food.title()
                        nutrition = data.copy()
                        source = "usda_fallback"
                        confidence = 0.6
                        break
                
                # Default if nothing matched
                if not nutrition:
                    food_name = "Mixed Meal"
                    nutrition = FOOD_DATABASE["default"].copy()
                    source = "estimated"
                    confidence = 0.3
            
            # Create Food object for rule checking
            # Ensure we handle None values properly
            cal = float(nutrition.get("calories") or 0)
            prot = float(nutrition.get("protein_g") or 0)
            carb = float(nutrition.get("carbs_g") or 0)
            fat = float(nutrition.get("fat_g") or 0)
            sugar = float(nutrition.get("sugar_g") or 0)
            fiber = float(nutrition.get("fiber_g") or 0)
            sodium = float(nutrition.get("sodium_mg") or 0)
            
            # Ensure carbs >= sugar + fiber for validation
            if carb < sugar + fiber:
                carb = sugar + fiber
            
            nutrition_info = NutritionInfo(
                calories=cal,
                protein_g=prot,
                carbs_g=carb,
                fat_g=fat,
                sugar_g=sugar,
                fiber_g=fiber,
                sodium_mg=sodium,
            )
            
            food = Food(
                food_id=str(uuid.uuid4()),
                name=food_name,
                nutrition=nutrition_info,
                serving_size=100.0,
                serving_unit="g",
            )
            
            # Apply safety rules
            violations = rule_engine.evaluate(food, demo_user)
            final_verdict = rule_engine.get_final_verdict(violations)
            
            # Determine verdict
            verdict = "safe"
            if final_verdict.value == "block":
                verdict = "danger"
            elif final_verdict.value in ["warn", "alert"]:
                verdict = "warning"
            
            # Store the meal in user's log BEFORE returning
            user_id = user["sub"]
            if user_id not in user_meal_logs:
                user_meal_logs[user_id] = []
            
            meal_data = {
                "food_name": food_name,
                "nutrition": {
                    "calories": cal,
                    "protein_g": prot,
                    "carbs_g": carb,
                    "fat_g": fat,
                    "sugar_g": sugar,
                    "fiber_g": fiber,
                    "sodium_mg": sodium,
                },
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "confidence": confidence,
            }
            
            user_meal_logs[user_id].append(meal_data)
            
            # Set as current food context for immediate AI coach use
            current_food_context[user_id] = meal_data
            
            print(f"[FOOD UPLOAD] ✓ Stored meal in logs (total: {len(user_meal_logs[user_id])})")
            print(f"[FOOD UPLOAD] ✓ Set as current context for AI coach")
            
            # Now return the response
            response_data = {
                "status": "success",
                "food_name": food_name,
                "confidence": confidence,
                "nutrition": {
                    "calories": nutrition_info.calories,
                    "protein_g": nutrition_info.protein_g,
                    "carbs_g": nutrition_info.carbs_g,
                    "fat_g": nutrition_info.fat_g,
                    "sugar_g": nutrition_info.sugar_g,
                    "fiber_g": nutrition_info.fiber_g,
                    "sodium_mg": nutrition_info.sodium_mg,
                },
                "safety": {
                    "verdict": verdict,
                    "level": final_verdict.value,
                    "violations": [
                        {
                            "rule_id": v.rule_id,
                            "severity": v.severity.value,
                            "message": v.message,
                        }
                        for v in violations
                    ],
                },
                "source": source,
            }
            
            return response_data
            
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
async def get_today_meals(user: dict = None):
    """
    Get all meals logged today by the current user.
    TEMP: Auth optional for testing.
    """
    # For testing: use demo user if no auth
    if not user:
        user = {"sub": "demo_user_123"}
    
    user_id = user["sub"]
    today = datetime.now().date()
    
    if user_id not in user_meal_logs:
        return {"meals": [], "message": "No meals logged yet"}
    
    # Filter to today's meals only
    today_meals = []
    for meal in user_meal_logs[user_id]:
        try:
            meal_date = datetime.fromisoformat(meal["timestamp"]).date()
            if meal_date == today:
                today_meals.append(meal)
        except:
            continue
    
    return {
        "meals": today_meals,
        "count": len(today_meals),
        "date": today.isoformat(),
    }


@app.delete("/api/meals/clear")
async def clear_meals(user: dict = None):
    """Clear all meals for the current user (for testing)."""
    if not user:
        user = {"sub": "demo_user_123"}
    user_id = user["sub"]
    if user_id in user_meal_logs:
        user_meal_logs[user_id] = []
    return {"message": "Meals cleared"}


@app.get("/api/food/current")
async def get_current_food(user: dict = None):
    """
    Get the most recently scanned food for immediate AI coach context.
    
    This is THE PRIMARY endpoint the AI coach should use.
    Returns the food that was just analyzed, with full nutrition data.
    """
    if not user:
        user = {"sub": "demo_user_123"}
    
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
    user: dict = Depends(get_optional_user),
):
    """
    Generate a recipe from ingredients.
    
    Flow:
    1. Parse ingredients
    2. Check medical constraints
    3. Generate recipe
    4. Validate against rules
    """
    # Demo mode fallback
    if not user:
        user = {"sub": "demo_user_123"}
    
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
    
    # Generate simple recipe
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
    }


# =============================================================================
# ROUTES: FIX MY MEAL
# =============================================================================

class FixMealRequest(BaseModel):
    food_name: str
    nutrition: Dict[str, float]
    violations: Optional[List[str]] = None


@app.post("/api/meal/fix")
async def fix_meal(
    request: FixMealRequest,
    user: dict = Depends(get_current_user),
):
    """
    Suggest fixes for a meal with violations.
    """
    suggestions = []
    
    # Analyze and suggest fixes based on common issues
    if request.nutrition.get("sugar_g", 0) > 20:
        suggestions.append({
            "issue": "High sugar content",
            "fix": "Replace with sugar-free alternative or reduce portion by half",
            "impact": f"Reduce sugar from {request.nutrition.get('sugar_g', 0)}g to ~{request.nutrition.get('sugar_g', 0)/2}g"
        })
    
    if request.nutrition.get("sodium_mg", 0) > 500:
        suggestions.append({
            "issue": "High sodium",
            "fix": "Use low-sodium version or add potassium-rich sides (banana, spinach)",
            "impact": f"Balance sodium with potassium"
        })
    
    if request.nutrition.get("fat_g", 0) > 20:
        suggestions.append({
            "issue": "High fat content", 
            "fix": "Choose grilled instead of fried, or trim visible fat",
            "impact": f"Reduce fat from {request.nutrition.get('fat_g', 0)}g to ~{request.nutrition.get('fat_g', 0)*0.6}g"
        })
    
    if request.nutrition.get("calories", 0) > 500:
        suggestions.append({
            "issue": "High calorie meal",
            "fix": "Reduce portion size or add more vegetables to feel full",
            "impact": f"Reduce calories by 30%"
        })
    
    if not suggestions:
        suggestions.append({
            "issue": "No major issues",
            "fix": "This meal looks reasonable! Consider adding more fiber.",
            "impact": "Improved digestion"
        })
    
    return {
        "status": "success",
        "original_food": request.food_name,
        "suggestions": suggestions,
        "fixed_meal": {
            "name": f"Healthier {request.food_name}",
            "estimated_reduction": "~30% fewer calories/sugar/sodium"
        }
    }


# =============================================================================
# ROUTES: ANALYTICS
# =============================================================================

@app.get("/api/analytics/summary")
async def get_analytics_summary(user: dict = Depends(get_current_user)):
    """
    Get analytics summary for user.
    """
    # Demo analytics data
    return {
        "health_score": 72,
        "health_grade": "B",
        "weekly_calories": {
            "target": 14000,
            "actual": 12500,
            "status": "good"
        },
        "macro_balance": {
            "protein": {"target": 25, "actual": 22},
            "carbs": {"target": 50, "actual": 55},
            "fat": {"target": 25, "actual": 23}
        },
        "streaks": {
            "logging": 5,
            "safe_meals": 12
        },
        "trends": {
            "calories": "decreasing",
            "protein": "stable",
            "sugar": "decreasing"
        }
    }


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
