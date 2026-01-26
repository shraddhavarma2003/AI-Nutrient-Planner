import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

def generate_steps_with_ollama(ingredients):
    prompt = f"""
    Create a healthy recipe using these ingredients: {ingredients}

    Format your response in Markdown:
    ## Ingredients
    - [item]
    
    ## Instructions
    1. [step]
    
    Rules:
    - Use minimal oil
    - Indian home-style cooking
    - Healthy and simple
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()

    text = result.get("response", "")
    steps = [line.strip() for line in text.split("\n") if line.strip()]

    return steps


def generate_recipe_with_llm(ingredients):
    return {
        "recipe_name": "LLM Generated Healthy Recipe",
        "ingredients": ingredients,
        "instructions": generate_steps_with_ollama(ingredients),
        "nutrition": {"calories": 250, "protein_g": 15, "carbs_g": 30, "fat_g": 8},
        "source": "Ollama LLM"
    }
