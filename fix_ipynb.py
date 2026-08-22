import json

filename = "kenyan_food_prices.ipynb"

with open(filename, "r", encoding="utf-8") as f:
    content = f.read()

# If it's pure JSON with single quotes, evaluate it safely as a python literal or replace keys
# Let's try loading it or cleaning it:
try:
    data = json.loads(content)
except Exception as e:
    print(f"JSON load error: {e}")
    # Fallback: parse via ast if single quotes were used like a python dict
    import ast
    data = ast.literal_eval(content)

with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Successfully fixed and converted to valid JSON!")
