"""Canonical ingredients list for bootstrapping the ingredient database.

This list contains common ingredients that serve as the foundation
for ingredient canonicalization. New ingredients will be matched
against these canonical entries using semantic similarity.
"""

# Flat list of canonical ingredient names
# These are the "ground truth" ingredients that extracted ingredients
# will be mapped to or compared against
CANONICAL_INGREDIENTS: list[str] = [
    # Proteins - Meat
    "beef",
    "pork",
    "chicken",
    "lamb",
    "duck",
    "turkey",
    "bacon",
    "ham",
    "sausage",
    "ground meat",
    
    # Proteins - Seafood
    "fish",
    "salmon",
    "tuna",
    "shrimp",
    "crab",
    "lobster",
    "squid",
    "octopus",
    "clam",
    "mussel",
    "oyster",
    
    # Proteins - Other
    "egg",
    "tofu",
    
    # Grains & Starches
    "rice",
    "bread",
    "noodles",
    "pasta",
    "flour",
    "oats",
    "corn",
    "potato",
    "sweet potato",
    
    # Vegetables - Fresh
    "fresh vegetables",
    "lettuce",
    "spinach",
    "cabbage",
    "broccoli",
    "cauliflower",
    "carrot",
    "celery",
    "cucumber",
    "tomato",
    "bell pepper",
    "zucchini",
    "eggplant",
    "mushroom",
    "onion",
    "garlic",
    "ginger",
    "green onion",
    "leek",
    
    # Vegetables - Pickled/Preserved
    "pickled vegetables",
    "kimchi",
    "sauerkraut",
    "pickle",
    
    # Legumes
    "beans",
    "lentils",
    "chickpeas",
    "peas",
    "edamame",
    
    # Dairy
    "milk",
    "cream",
    "butter",
    "cheese",
    "yogurt",
    "sour cream",
    
    # Oils & Fats
    "oil",
    "olive oil",
    "sesame oil",
    "vegetable oil",
    "coconut oil",
    "fat",
    
    # Seasonings & Spices
    "salt",
    "pepper",
    "sugar",
    "honey",
    "soy sauce",
    "fish sauce",
    "vinegar",
    "lemon",
    "lime",
    "chili",
    "paprika",
    "cumin",
    "coriander",
    "turmeric",
    "cinnamon",
    "oregano",
    "basil",
    "thyme",
    "rosemary",
    "parsley",
    "cilantro",
    "mint",
    "dill",
    "herbs",
    
    # Sauces & Condiments
    "ketchup",
    "mayonnaise",
    "mustard",
    "hot sauce",
    "tomato sauce",
    "oyster sauce",
    "hoisin sauce",
    
    # Nuts & Seeds
    "peanut",
    "almond",
    "walnut",
    "cashew",
    "sesame seeds",
    
    # Fruits
    "apple",
    "banana",
    "orange",
    "lemon",
    "lime",
    "mango",
    "pineapple",
    "coconut",
    "berries",
    "grapes",
    "avocado",
    "papaya",
    
    # Baking
    "baking powder",
    "baking soda",
    "yeast",
    "vanilla",
    "chocolate",
    
    # Broths & Stocks
    "broth",
    "stock",
    "water",
    "wine",
]
