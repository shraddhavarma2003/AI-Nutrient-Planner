try:
    with open(r'c:\Users\hp\AI Nutrition\venv\Lib\site-packages\langchain\agents\__init__.py', 'r', encoding='utf-8') as f:
        print(f.read())
except Exception as e:
    print(f"Error: {e}")
