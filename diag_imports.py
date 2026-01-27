import sys
import os

results = []

def test_import(module_name, attribute_name):
    try:
        mod = __import__(module_name, fromlist=[attribute_name])
        attr = getattr(mod, attribute_name)
        return True, f"SUCCESS: {module_name}.{attribute_name}"
    except ImportError as e:
        return False, f"ImportError: {module_name}.{attribute_name} - {str(e)}"
    except AttributeError:
        return False, f"AttributeError: {module_name}.{attribute_name}"
    except Exception as e:
        return False, f"ERROR: {module_name}.{attribute_name} - {str(e)}"

# Test variations
paths = [
    ("langchain.agents", "AgentExecutor"),
    ("langchain.agents.agent_executor", "AgentExecutor"),
    ("langchain.agents.agent", "AgentExecutor"),
    ("langchain.agents", "create_structured_chat_agent"),
]

for mod, attr in paths:
    success, msg = test_import(mod, attr)
    results.append(msg)

# Write results to file
with open("import_diag.txt", "w") as f:
    f.write("\n".join(results))
    f.write("\n\nPython Version: " + sys.version)
    f.write("\nCWD: " + os.getcwd())
