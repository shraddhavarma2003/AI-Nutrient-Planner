from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

def test_prompt():
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Simulating the prompt in agent_service.py
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a deterministic Data Retrieval Bot for the AI Nutrition system.
        
        STRICT RULES:
        1. You HAVE access to all user data (meals, profiles, nutritional stats) via tools.
        2. To answer ANY data-related question, you MUST use the provided tools.
        3. NEVER say "I don't have access", "I am an AI", or "Check your diary". 
        4. If a tool fails or no information is found via the tool, inform the user Concisely.

        TOOLS:
        ------
        {{tools}}

        EXAMPLE OF CORRECT BEHAVIOR:
        ----------------------------
        Question: How many calories did I eat today?
        Thought: I need to check the daily stats for today's date.
        Action: get_daily_stats
        Action Input: 2026-01-27
        Observation: {{{{'user_id': '...', 'calories_consumed': 500, ...}}}}
        Thought: I now have the user's calorie count which is 500.
        Final Answer: You have consumed 500 calories today.
        ----------------------------

        FORMAT:
        -------
        Use the following format for EVERY response:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{{tool_names}}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (repeat Thought/Action/Action Input/Observation if needed)
        Thought: I now have the answer
        Final Answer: the final answer to the original input question

        Begin!

        User ID: {{user_id}}
        Current Date: {current_date}
"""),
        ("human", "{input}\n\n{agent_scratchpad}"),
    ])

    print("Input variables identified by LangChain:")
    print(prompt.input_variables)
    with open("test_prompt_output.txt", "w") as f:
        f.write(str(prompt.input_variables))

if __name__ == "__main__":
    test_prompt()
