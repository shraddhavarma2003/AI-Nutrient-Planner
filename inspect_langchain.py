import langchain
import os
import inspect

print(f"Version: {getattr(langchain, '__version__', 'N/A')}")
print(f"File: {langchain.__file__}")
print(f"Doc: {langchain.__doc__}")

print("\nSubmodules in langchain:")
for name, obj in inspect.getmembers(langchain):
    if inspect.ismodule(obj):
        print(f" - {name}")

try:
    import langchain.agents as agents
    print("\nExports in langchain.agents:")
    print(dir(agents))
except ImportError as e:
    print(f"\nCould not import langchain.agents: {e}")

# Check for specific files in the directory
base_path = os.path.dirname(langchain.__file__)
print(f"\nContents of {base_path}:")
try:
    print(os.listdir(base_path))
    agents_path = os.path.join(base_path, 'agents')
    if os.path.exists(agents_path):
        print(f"\nContents of {agents_path}:")
        print(os.listdir(agents_path))
except Exception as e:
    print(f"Error listing dir: {e}")
