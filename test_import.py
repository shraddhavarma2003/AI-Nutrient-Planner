try:
    import langchain
    print(f"langchain version: {getattr(langchain, '__version__', 'unknown')}")
    print(f"langchain path: {langchain.__file__}")
except ImportError:
    print("langchain not installed")

try:
    from langchain.agents import AgentExecutor
    print("SUCCESS: from langchain.agents import AgentExecutor")
except ImportError as e:
    print(f"FAILED: from langchain.agents import AgentExecutor - {e}")

try:
    from langchain.agents.agent_executor import AgentExecutor
    print("SUCCESS: from langchain.agents.agent_executor import AgentExecutor")
except ImportError as e:
    print(f"FAILED: from langchain.agents.agent_executor import AgentExecutor - {e}")

try:
    from langchain.agents.agent import AgentExecutor
    print("SUCCESS: from langchain.agents.agent import AgentExecutor")
except ImportError as e:
    print(f"FAILED: from langchain.agents.agent import AgentExecutor - {e}")

try:
    import langchain_community
    print(f"langchain_community version: {getattr(langchain_community, '__version__', 'unknown')}")
except ImportError:
    print("langchain_community not installed")

try:
    from langchain_community.agent_toolkits import create_spark_sql_agent # Just a test
    print("SUCCESS: from langchain_community.agent_toolkits import ...")
except ImportError as e:
    print(f"FAILED: from langchain_community.agent_toolkits import ... - {e}")
