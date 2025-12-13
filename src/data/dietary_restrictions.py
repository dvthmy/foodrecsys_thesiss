"""Dietary restrictions data for seeding the database.

Each restriction defines:
- name: The restriction identifier (e.g., "dairy-allergy")
- description: Human-readable description
- not_suited_ingredients: Ingredients that violate this restriction
- suited_ingredients: Ingredients explicitly suited (optional, for clarity)
"""

from typing import TypedDict


class DietaryRestrictionData(TypedDict):
    """Type definition for dietary restriction seed data."""

    name: str
    description: str
    not_suited_ingredients: list[str]
    suited_ingredients: list[str]


DIETARY_RESTRICTIONS: list[DietaryRestrictionData] = [
    # =========================================================================
    # Allergy-based restrictions
    # =========================================================================
    {
        "name": "dairy-allergy",
        "description": "Avoids all dairy products including milk, cheese, and butter",
        "not_suited_ingredients": [
            "milk",
            "cream",
            "butter",
            "cheese",
            "yogurt",
            "sour cream",
            "cream cheese",
            "whipped cream",
            "condensed milk",
            "evaporated milk",
            "ghee",
            "paneer",
            "ricotta",
            "mozzarella",
            "parmesan",
            "cheddar",
            "feta",
            "brie",
            "cottage cheese",
        ],
        "suited_ingredients": [
            "coconut milk",
            "almond milk",
            "soy milk",
            "oat milk",
            "coconut cream",
        ],
    },
    {
        "name": "seafood-allergy",
        "description": "Avoids all seafood including fish and shellfish",
        "not_suited_ingredients": [
            "fish",
            "salmon",
            "tuna",
            "cod",
            "tilapia",
            "mackerel",
            "sardine",
            "anchovy",
            "shrimp",
            "prawn",
            "crab",
            "lobster",
            "squid",
            "octopus",
            "clam",
            "mussel",
            "oyster",
            "scallop",
            "fish sauce",
            "oyster sauce",
            "shrimp paste",
            "caviar",
            "roe",
        ],
        "suited_ingredients": [],
    },
    {
        "name": "peanut-allergy",
        "description": "Avoids peanuts and peanut-derived products",
        "not_suited_ingredients": [
            "peanut",
            "peanut butter",
            "peanut oil",
            "groundnut",
        ],
        "suited_ingredients": [
            "almond",
            "cashew",
            "walnut",
            "almond butter",
            "sunflower seed butter",
        ],
    },
    # =========================================================================
    # Diet-based restrictions
    # =========================================================================
    {
        "name": "vegetarian",
        "description": "No meat or fish, but eggs and dairy are allowed",
        "not_suited_ingredients": [
            # Meat
            "beef",
            "pork",
            "chicken",
            "lamb",
            "duck",
            "turkey",
            "veal",
            "venison",
            "rabbit",
            "goat",
            # Processed meat
            "bacon",
            "ham",
            "sausage",
            "salami",
            "pepperoni",
            "prosciutto",
            "chorizo",
            "hot dog",
            "ground beef",
            "ground pork",
            # Seafood
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
            "anchovy",
            "sardine",
            # Animal-derived cooking ingredients
            "fish sauce",
            "oyster sauce",
            "chicken broth",
            "beef broth",
            "lard",
            "tallow",
        ],
        "suited_ingredients": [
            "tofu",
            "tempeh",
            "seitan",
            "egg",
            "cheese",
            "milk",
            "yogurt",
            "legume",
            "lentil",
            "chickpea",
            "bean",
        ],
    },
    {
        "name": "low-carb",
        "description": "Limits high-carbohydrate foods for keto or low-carb diets",
        "not_suited_ingredients": [
            # Grains and starches
            "rice",
            "bread",
            "noodles",
            "pasta",
            "flour",
            "wheat",
            "oats",
            "oatmeal",
            "cereal",
            "corn",
            "cornstarch",
            "cornmeal",
            # Starchy vegetables
            "potato",
            "sweet potato",
            "yam",
            "taro",
            "cassava",
            # Sugars
            "sugar",
            "brown sugar",
            "honey",
            "maple syrup",
            "corn syrup",
            "molasses",
            # High-carb fruits
            "banana",
            "grape",
            "mango",
            "pineapple",
        ],
        "suited_ingredients": [
            # Proteins
            "beef",
            "pork",
            "chicken",
            "fish",
            "egg",
            "bacon",
            # Dairy
            "cheese",
            "butter",
            "cream",
            # Fats
            "olive oil",
            "coconut oil",
            "avocado",
            # Low-carb vegetables
            "spinach",
            "kale",
            "broccoli",
            "cauliflower",
            "zucchini",
            "cucumber",
            "bell pepper",
            "mushroom",
        ],
    },
]
