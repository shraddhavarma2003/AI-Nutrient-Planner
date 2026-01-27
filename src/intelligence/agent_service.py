import os
import sys
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama

# Robust import strategy for AgentExecutor and related tools
AgentExecutor = None
create_structured_chat_agent = None

try:
    # Primary path for LangChain 1.2.7+ (Legacy components)
    from langchain_classic.agents import AgentExecutor, create_structured_chat_agent, create_react_agent
except ImportError:
    try:
        # Standard path for older LangChain versions
        from langchain.agents import AgentExecutor, create_structured_chat_agent, create_react_agent
    except ImportError:
        # Fallback search for other paths
        paths_to_try = [
            ("langchain.agents.agent_executor", "AgentExecutor"),
            ("langchain.agents.agent", "AgentExecutor"),
            ("langchain.agents.executor", "AgentExecutor"),
            ("langchain.agents.structured_chat.base", "create_structured_chat_agent"),
            ("langchain.agents.react.base", "create_react_agent"),
        ]
        
        import importlib
        for module_path, attr_name in paths_to_try:
            try:
                mod = importlib.import_module(module_path)
                val = getattr(mod, attr_name)
                if attr_name == "AgentExecutor":
                    AgentExecutor = val
                elif attr_name == "create_structured_chat_agent":
                    create_structured_chat_agent = val
                elif attr_name == "create_react_agent":
                    create_react_agent = val
            except (ImportError, AttributeError):
                continue

# Final check
if AgentExecutor is None or create_structured_chat_agent is None:
    # Diagnostic info to help user
    import langchain
    pkg_path = getattr(langchain, '__file__', 'unknown')
    raise ImportError(
        f"Could not find recommended LangChain components. \n"
        f"Please install 'langchain-classic': pip install langchain-classic\n"
        f"Current LangChain: {pkg_path}"
    )
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from auth.database import DailyLogRepository, MedicalProfileRepository, MealRepository
from services.rag_service import get_rag_service
from intelligence.meal_fixer import MealFixer
from intelligence.recipe_generator import RecipeGenerator
from rules.engine import RuleEngine

class NutritionAgent:
    """
    Intelligent Agent that coordinates nutrition-related tasks.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.model = os.getenv("OLLAMA_MODEL", "gemma:2b")
        self.llm = ChatOllama(model=self.model, temperature=0)
        self.tools = self._setup_tools()
        self.agent_executor = self._setup_agent()

    def _setup_tools(self):
        
        @tool
        def nutrition_lookup(food_name: Any) -> str:
            """Look up nutrition information for a given food name."""
            # Handle possible dict input from structured agent
            if isinstance(food_name, dict):
                food_name = food_name.get('food_name', list(food_name.values())[0])
            
            clean_name = str(food_name).replace('food_name=', '').replace('"', '').strip()
            
            rag = get_rag_service()
            result = rag.get_food_info(clean_name)
            if not result:
                return f"Could not find nutrition info for {clean_name}."
            return str(result)

        @tool
        def get_daily_stats(date: Any = None) -> str:
            """Get user's nutrition statistics (calories, macros) for a specific date (YYYY-MM-DD). Use 'today' if unsure."""
            # Handle possible dict input from structured agent
            if isinstance(date, dict):
                date = date.get('date', list(date.values())[0] if date else None)
            
            if date:
                date = str(date).replace('date=', '').replace('"', '').strip()
            
            # Robustly handle empty/None/null inputs from LLM
            from datetime import datetime
            if not date or str(date).lower() in ["none", "null", "today", "undefined"]:
                date = datetime.now().strftime("%Y-%m-%d")
            
            stats = DailyLogRepository.get_or_create(self.user_id, date)
            return f"User statistics for {date}: {str(stats)}"

        @tool
        def get_medical_profile() -> str:
            """Retrieve the user's medical conditions, allergies, and health targets."""
            profile = MedicalProfileRepository.get_by_user_id(self.user_id)
            if not profile:
                return "No medical profile found for this user."
            return str(profile)

        @tool
        def suggest_meal_fix(food_name: Any) -> str:
            """Suggest improvements for a specific meal based on user's health profile."""
            if isinstance(food_name, dict):
                food_name = food_name.get('food_name', list(food_name.values())[0])
            
            clean_name = str(food_name).replace('food_name=', '').replace('"', '').strip()
            
            rag = get_rag_service()
            nutrition = rag.get_food_info(clean_name)
            if not nutrition:
                return f"Could not find nutrition for {clean_name} to suggest fixes."
            
            from models.food import Food, NutritionInfo, FoodCategory, DataSource
            food_obj = Food(
                food_id="temp",
                name=clean_name,
                serving_size=100.0,
                serving_unit="g",
                nutrition=NutritionInfo(
                    calories=nutrition.get("calories", 0),
                    protein_g=nutrition.get("protein_g", 0),
                    carbs_g=nutrition.get("carbs_g", 0),
                    fat_g=nutrition.get("fat_g", 0),
                    sugar_g=nutrition.get("sugar_g", 0),
                    sodium_mg=nutrition.get("sodium_mg", 0)
                ),
                allergens=[],
                category=FoodCategory.OTHER,
                data_source=DataSource.USER_VERIFIED
            )
            
            profile_data = MedicalProfileRepository.get_by_user_id(self.user_id)
            from models.user import UserProfile
            user_profile = UserProfile(
                user_id=self.user_id,
                name="User",
                conditions=profile_data.get("conditions", []),
                allergens=profile_data.get("allergens", [])
            )
            
            engine = RuleEngine()
            fixer = MealFixer(engine)
            from models.user import DailyIntake
            fixes = fixer.analyze_and_fix([food_obj], user_profile, DailyIntake())
            return "\n".join([f.to_dict()["fix_description"] for f in fixes]) if fixes else "No fixes suggested. Your meal looks good!"

        @tool
        def generate_recipe(ingredients: Any) -> str:
            """Generate a healthy recipe using a list of ingredients."""
            # Handle various model output patterns
            if isinstance(ingredients, dict):
                ingredients = ingredients.get('ingredients', list(ingredients.values())[0])
            
            if isinstance(ingredients, str):
                 ingredients = [i.strip() for i in ingredients.replace('ingredients=', '').replace('[', '').replace(']', '').replace('"', '').split(',')]

            profile_data = MedicalProfileRepository.get_by_user_id(self.user_id)
            engine = RuleEngine()
            generator = RecipeGenerator(engine)
            
            from models.user import UserProfile
            user_profile = UserProfile(
                user_id=self.user_id,
                conditions=profile_data.get("conditions", []),
                allergens=profile_data.get("allergens", [])
            )
            
            recipe = generator.generate(ingredients, user_profile, None)
            if not recipe:
                return "Could not generate a safe recipe with those ingredients."
            return str(recipe.to_dict())

        return [nutrition_lookup, get_daily_stats, get_medical_profile, suggest_meal_fix, generate_recipe]

    def _setup_agent(self):
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Create a clean system message without f-string layer to avoid brace confusion
        system_message = (
            "You are a helpful and knowledgeable AI Nutrition Coach. You have access to user data via tools but can also give general advice.\n\n"
            "STRICT RULES:\n"
            "1. You HAVE access to all user data (meals, profiles, nutritional stats) via tools.\n"
            "2. To answer ANY data-related question, you MUST use the provided tools.\n"
            "3. NEVER use parentheses in your Action calls. Example: use 'Action: get_daily_stats', NOT 'Action: get_daily_stats()'.\n"
            "4. NEVER say 'I don't have access', 'I am an AI', or 'Check your diary'. \n"
            "5. If a tool fails, inform the user Concisely.\n"
            "6. GENERAL ADVICE & TOOLS: If you use a tool, use its output to help answer the user's ORIGINAL question. For example, if they ask for exercise tips and you see they have 2000 calories left, suggest a brisk walk.\n"
            "7. MAPPING: You MUST still follow the format below: Thought -> Final Answer. NEVER output a direct answer without headers.\n\n"
            "TOOLS:\n"
            "------\n"
            "{tools}\n\n"
            "EXAMPLE OF CORRECT BEHAVIOR (TOOL USE):\n"
            "----------------------------\n"
            "Question: How many calories did I eat today?\n"
            "Thought: I need to check the daily stats for today's date.\n"
            "Action: get_daily_stats\n"
            "Action Input: " + current_date + "\n"
            "Observation: (User statistics retrieved: calories_consumed=500, target=2000)\n"
            "Thought: I now have the user's calorie count which is 500. This is well below their target.\n"
            "Final Answer: You have consumed 500 calories today, which is 1500 below your goal. You have plenty of room for a healthy dinner!\n"
            "----------------------------\n"
            "EXAMPLE OF CORRECT BEHAVIOR (NO TOOL):\n"
            "----------------------------\n"
            "Question: What exercise is good for cardio?\n"
            "Thought: I do not have a specific tool for exercise exercises, I will answer based on general knowledge.\n"
            "Final Answer: Walking, running, and swimming are great forms of cardio exercise to improve your heart health.\n"
            "----------------------------\n\n"
            "FORMAT:\n"
            "-------\n"
            "Use the following format for EVERY response:\n\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}] (NO PARENTHESES!)\n"
            "Action Input: the input to the action (or 'None')\n"
            "Observation: the result of the action\n"
            "... (repeat Thought/Action/Action Input/Observation if needed)\n"
            "Thought: I now have the answer to the user's original question\n"
            "Final Answer: the final answer to the original input question\n\n"
            "Begin!\n\n"
            "Today's Date: " + current_date
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}\n\n{agent_scratchpad}"),
        ])
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors="Check your output and make sure it conforms! Do not output an XML code block! 'Thought' must be followed by 'Action' or 'Final Answer'.",
            max_iterations=30
        )

    def chat(self, user_input: str, chat_history: List[Any] = None) -> str:
        """Process user input through the agent."""
        try:
            # We no longer pass user_id to invoke because it's not in the template.
            # The tools already have self.user_id.
            response = self.agent_executor.invoke({
                "input": user_input + "\n\n(IMPORTANT: You MUST start with 'Thought:' and use the headers 'Action:' or 'Final Answer:' as required by the ReAct format.)",
                "chat_history": chat_history or []
            })
            return response["output"]
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"
