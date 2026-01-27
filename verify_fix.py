try:
    from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
    print("SUCCESS: Imported AgentExecutor and create_structured_chat_agent from langchain_classic")
except ImportError as e:
    print(f"FAILED: {e}")

try:
    from intelligence.agent_service import NutritionAgent
    print("SUCCESS: Imported NutritionAgent from intelligence.agent_service")
except ImportError as e:
    print(f"FAILED to import NutritionAgent: {e}")
