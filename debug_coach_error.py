
import sys
import os
import traceback
import json

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from auth.database import init_database, UserRepository, MealRepository
from main import app
from fastapi.testclient import TestClient
from services.llm_service import get_llm_service

def debug_coach_chat():
    print("Initializing database...")
    init_database()
    
    # Ensure we have a user
    user_id = "debug_user"
    UserRepository.create(user_id, "debug@example.com", "hash", "Debug User")
    
    # Create valid token (mocking auth is harder with full integration test, 
    # so let's try to invoke the underlying logic or verify if LLM service is the culprit first)
    
    # Actually, using TestClient is better to hit the endpoint directly
    client = TestClient(app)
    
    # We need to mock the auth dependency overrides or generate a valid token
    # But for a 500 error, it often happens even with valid inputs.
    # Let's see if we can trigger it by calling the internal services directly first, 
    # mimicking what the endpoint does.
    
    # The endpoint logic:
    # 1. get recent logs
    # 2. build RAG context
    # 3. call LLM
    
    print("\n--- Testing Internal Service Logic ---")
    try:
        from services.rag_service import get_rag_service
        rag = get_rag_service()
        
        print("Building Context...")
        # Mock pulling logs
        user_meal_logs = [{"food_name": "apple", "nutrition": {"calories": 95}}]
        
        ctx = rag.build_context(
            user_id=user_id,
            meal_logs=user_meal_logs,
            current_food=None,
            user_question="Is an apple healthy?"
        )
        print(f"Context built: {len(ctx)} chars")
        
        llm = get_llm_service()
        print(f"LLM Available: {llm.is_available}")
        
        print("Calling LLM chat...")
        response = llm.chat(
            prompt="Is an apple healthy?",
            system_prompt="nutrition_coach",
            rag_context=ctx
        )
        print(f"LLM Response success: {response.success}")
        if response.success:
            print(f"Content: {response.content[:100]}...")
        else:
            print(f"Error: {response.error}")
            
    except Exception as e:
        print(f"CRITICAL ERROR in Service Logic: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_coach_chat()
