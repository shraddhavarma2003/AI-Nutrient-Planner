"""
Ollama LLM Service (Gemma)

Provides AI-powered responses for nutrition guidance while maintaining
medical safety through rule-based overrides.

SAFETY PRINCIPLE: Medical rules ALWAYS override LLM suggestions.
"""

import os
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from LLM service."""
    content: str
    success: bool
    error: Optional[str] = None
    model: Optional[str] = None


class OllamaService:
    """
    Centralized Ollama LLM service for nutrition AI using Gemma model.
    
    Features:
    - Nutrition-aware system prompts
    - Medical safety filtering
    - Graceful fallback on errors
    - Local inference with Ollama
    
    Usage:
        service = OllamaService()
        response = service.chat("Is this food healthy?", context={"food": "apple"})
    """
    
    # System prompts for different contexts
    SYSTEM_PROMPTS = {
        "nutrition_coach": """You are a helpful nutrition coach AI assistant. You provide 
friendly, evidence-based nutrition advice.

RESPONSE FORMAT:
1. First, briefly acknowledge the user's health profile (conditions, allergens) from the context
2. Then answer their question considering their specific health needs
3. End with a relevant tip or encouragement

Important rules:
- Never provide medical diagnoses or treatment recommendations
- Always suggest consulting healthcare providers for medical concerns
- Focus on general nutrition education and healthy eating tips
- Be encouraging and supportive
- Keep responses concise (2-3 paragraphs max)""",

        "exercise_guide": """You are an exercise and fitness guidance AI. You provide 
general fitness advice and exercise recommendations. Important rules:
- Never provide medical advice or diagnoses
- Always recommend consulting a doctor before starting exercise programs
- Focus on general fitness education
- Provide calorie burn estimates as approximations only
- Be encouraging but emphasize safety first
- Keep responses concise with bullet points when helpful""",

        "meal_fixer": """You are a clinical nutrition AI that helps improve meals. 
Given a meal and health conditions, suggest specific improvements. Important rules:
- Focus on practical, actionable suggestions
- Consider the user's health conditions when making recommendations
- Suggest ingredient substitutions and portion adjustments
- Explain WHY each change helps
- Never claim to treat or cure medical conditions
- Keep responses structured with clear categories (Remove, Reduce, Replace)""",

        "clinical_nutritionist": """As a clinical dietitian, analyze the provided meal and offer practical, medical-aware advice. 
Focus on 3 brief, extremely practical tips (e.g., 'replace white rice with brown', 'reduce salt').
Be concise, professional, and use bullet points.""",
        
        "recipe_creator": """You are an expert healthy chef and nutritionist. 
Generate a healthy recipe based on provided ingredients and user health profile.

FORMAT YOUR RESPONSE IN CLEAR MARKDOWN:
# [Recipe Name]
[Health Note - Describe how this recipe helps the user's conditions]

## Ingredients
- [Item 1]
- [Item 2]

## Instructions
1. [Step 1]
2. [Step 2]

## Nutrition (Approximate)
- Calories: [amount]
- Protein: [amount]g
- Carbs: [amount]g
- Fat: [amount]g

Rules:
- Respect all health conditions and allergens.
- Use simple, accessible ingredients.
- Use proper Markdown headers and bullet/numbered lists.""",

        "medical_parser": """You are a medical data extraction assistant.
Extract the following information from the provided medical report text into a JSON object.

IMPORTANT: YOUR RESPONSE MUST BE ONLY A RAW JSON OBJECT. NO MARKDOWN.

SCHEMA:
{
  "conditions": [],
  "allergens": [],
  "medications": [],
  "vitals": {
    "glucose_level": null,
    "cholesterol": null,
    "hba1c": null,
    "systolic_bp": null,
    "diastolic_bp": null
  },
  "biometrics": {
    "age": null,
    "gender": null,
    "weight_kg": null,
    "height_cm": null
  }
}

Rules:
- "conditions": List distinct diagnosed medical conditions found (e.g., "Type 2 Diabetes").
- "allergens": List foods or substances the patient is allergic to.
- "medications": List names of prescribed medications (e.g., "Metformin").
- "vitals": Extract numeric values for Glucose, Cholesterol (Total, HDL, LDL, Triglycerides), HbA1c, and Blood Pressure.
- "biometrics": Extract Age (years), Gender (male/female), Weight (kg), and Height (cm).
- If nothing is found in a category, return an empty list or null values.
- IMPORTANT: If you see a test name (e.g., "Glucose fasting") followed by a number, that number is the value.
- Be precise. Do not hallucinate information not in the text.
"""
    }
    
    def __init__(
        self, 
        base_url: Optional[str] = None, 
        model: Optional[str] = None
    ):
        """
        Initialize Ollama service with Gemma model.
        
        Args:
            base_url: Ollama API base URL (defaults to OLLAMA_BASE_URL env var or http://localhost:11434)
            model: Model to use (defaults to OLLAMA_MODEL env var or gemma:2b)
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma:2b")
        self._initialized = False
        
        # Test connection to Ollama
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self._initialized = True
                print(f"[LLM] Ollama connected successfully. Using model: {self.model}")
            else:
                print(f"[LLM] Ollama returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"[LLM] Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running.")
        except Exception as e:
            print(f"[LLM] Failed to initialize Ollama client: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self._initialized
    
    def chat(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        rag_context: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a response using Ollama with Gemma.
        
        Args:
            prompt: User's message/question
            system_prompt: Optional system prompt (use SYSTEM_PROMPTS keys or custom)
            context: Optional context dict to include in the prompt
            rag_context: Optional RAG-retrieved context (user profile, meal history, etc.)
            
        Returns:
            LLMResponse with content and status
        """
        if not self.is_available:
            return LLMResponse(
                content="",
                success=False,
                error="LLM service not available. Please ensure Ollama is running."
            )
        
        # Get system prompt
        system = self.SYSTEM_PROMPTS.get(system_prompt, system_prompt) or self.SYSTEM_PROMPTS["nutrition_coach"]
        
        # Build context-enhanced prompt
        enhanced_prompt_parts = []
        
        # Add RAG context first (user profile, meal history, etc.)
        if rag_context:
            enhanced_prompt_parts.append(rag_context)
        
        # Add simple context dict if provided
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            enhanced_prompt_parts.append(f"Additional Context:\n{context_str}")
        
        # Add the user question
        enhanced_prompt_parts.append(f"\nUser question: {prompt}")
        
        enhanced_prompt = "\n".join(enhanced_prompt_parts)
        
        try:
            # Ollama API endpoint for chat
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    "stream": False
                },
                timeout=90  # 90s timeout for local inference
            )
            
            if response.status_code != 200:
                return LLMResponse(
                    content="",
                    success=False,
                    error=f"Ollama API error: {response.status_code}"
                )
            
            result = response.json()
            content = result.get("message", {}).get("content", "")
            
            return LLMResponse(
                content=content,
                success=True,
                model=self.model
            )
            
        except requests.exceptions.Timeout:
            print("[LLM] Ollama request timed out")
            return LLMResponse(
                content="",
                success=False,
                error="Request timed out. The model may be loading or processing a complex query."
            )
        except Exception as e:
            print(f"[LLM] Error calling Ollama API: {e}")
            return LLMResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    def get_nutrition_advice(
        self, 
        food_name: str,
        nutrition: Dict[str, Any],
        user_conditions: List[str],
        question: str
    ) -> LLMResponse:
        """
        Get nutrition advice for a specific food.
        
        Args:
            food_name: Name of the food
            nutrition: Nutrition info dict
            user_conditions: User's health conditions
            question: User's question about the food
            
        Returns:
            LLMResponse with advice
        """
        context = {
            "Food": food_name,
            "Calories": nutrition.get("calories", "unknown"),
            "Carbs": f"{nutrition.get('carbs_g', 0)}g",
            "Sugar": f"{nutrition.get('sugar_g', 0)}g",
            "Protein": f"{nutrition.get('protein_g', 0)}g",
            "Fat": f"{nutrition.get('fat_g', 0)}g",
            "Sodium": f"{nutrition.get('sodium_mg', 0)}mg",
            "User conditions": ", ".join(user_conditions) if user_conditions else "None specified"
        }
        
        return self.chat(question, system_prompt="nutrition_coach", context=context)
    
    def get_exercise_advice(self, question: str) -> LLMResponse:
        """
        Get exercise and fitness advice.
        
        Args:
            question: User's exercise-related question
            
        Returns:
            LLMResponse with exercise advice
        """
        return self.chat(question, system_prompt="exercise_guide")
    
    def get_meal_fix_suggestions(
        self,
        food_name: str,
        nutrition: Dict[str, Any],
        conditions: List[str],
        problems: List[Dict[str, Any]]
    ) -> LLMResponse:
        """
        Get AI-powered meal improvement suggestions.
        
        Args:
            food_name: Name of the food/meal
            nutrition: Nutrition information
            conditions: User's health conditions
            problems: Detected problems from rule engine
            
        Returns:
            LLMResponse with meal fix suggestions
        """
        problems_str = "\n".join(
            f"- {p.get('ingredient', 'Unknown')}: {p.get('reason', 'Unknown reason')}"
            for p in problems
        ) if problems else "No specific problems detected"
        
        context = {
            "Meal": food_name,
            "Calories": nutrition.get("calories", 0),
            "Carbs": f"{nutrition.get('carbs_g', 0)}g",
            "Sugar": f"{nutrition.get('sugar_g', 0)}g", 
            "Fat": f"{nutrition.get('fat_g', 0)}g",
            "Sodium": f"{nutrition.get('sodium_mg', 0)}mg",
            "Health conditions": ", ".join(conditions) if conditions else "None",
            "Detected problems": problems_str
        }
        
        prompt = """Based on the meal and detected problems above, provide specific 
suggestions to make this meal healthier for the user's conditions. 
Format your response with:
1. REMOVE: Items to remove entirely
2. REDUCE: Items to reduce quantity
3. REPLACE: Healthier alternatives
4. ADD: Beneficial additions

Be specific and practical."""
        
        return self.chat(prompt, system_prompt="meal_fixer", context=context)

    def generate_recipe(
        self,
        ingredients: List[str],
        user_profile: Dict[str, Any]
    ) -> LLMResponse:
        """
        Generate a healthy recipe from a list of ingredients.
        
        Args:
            ingredients: List of ingredient names
            user_profile: User's health profile (conditions, allergens)
            
        Returns:
            LLMResponse with the generated recipe.
        """
        context = {
            "Ingredients available": ", ".join(ingredients),
            "Health Conditions": ", ".join(user_profile.get("conditions", [])),
            "Allergens to avoid": ", ".join(user_profile.get("allergens", []))
        }
        
        prompt = f"Create a healthy recipe using these ingredients: {', '.join(ingredients)}. "
        prompt += "Ensure it is safe given the user's health profile."
        
        return self.chat(prompt, system_prompt="recipe_creator", context=context)

    def parse_medical_report(self, raw_text: str) -> LLMResponse:
        """
        Extract structured data from raw medical report text.
        """
        # Truncate text if too long to fit context window
        max_chars = 10000
        text_segment = raw_text[:max_chars] + ("..." if len(raw_text) > max_chars else "")
        
        prompt = f"""Analyze this medical report text and extract proper diagnosis and vitals:

---
{text_segment}
---"""
        
        return self.chat(prompt, system_prompt="medical_parser")



# Global singleton instance
_ollama_service: Optional[OllamaService] = None


def get_llm_service() -> OllamaService:
    """Get or create the global Ollama service instance."""
    global _ollama_service
    if _ollama_service is None:
        _ollama_service = OllamaService()
    return _ollama_service


# Backward compatibility alias
def get_mistral_service() -> OllamaService:
    """Backward compatibility alias for get_llm_service()."""
    return get_llm_service()
