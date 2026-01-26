from flask import Flask, request, jsonify
from recipe_api import fetch_recipe_from_api
from ai_engine import generate_recipe_with_llm

app = Flask(__name__)

@app.route("/recipe", methods=["POST"])
def get_recipe():
    data = request.json
    ingredients = data.get("ingredients", [])

    api_recipe = fetch_recipe_from_api(ingredients)

    if api_recipe:
        return jsonify({
            "recipe_name": api_recipe.get("title") or api_recipe.get("recipe_name") or "Healthy Recipe",
            "ingredients": api_recipe.get("ingredients", ingredients),
            "instructions": api_recipe.get("instructions") or api_recipe.get("steps") or [],
            "nutrition": api_recipe.get("nutrition") if isinstance(api_recipe.get("nutrition"), dict) else {"calories": 0},
            "source": "Recipe API"
        })

    # Fallback to LLM
    return jsonify(generate_recipe_with_llm(ingredients))


if __name__ == "__main__":
    app.run(debug=True)
