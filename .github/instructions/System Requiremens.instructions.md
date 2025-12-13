---
applyTo: '**'
---
# System Requirements
A Food Recommendation System that suggests the best dishes based on user preferences and dietary restrictions (e.g., vegan, gluten-free).

## Components
1. A data pipeline to collect and preprocess food data from various sources.
 - The ingredients are extracted from the descriptions using AI models (e.g., Gemini API, CLIP).
 - The dish's image is embedded using image embedding models (e.g., CLIP) to capture visual features.
2. The user data, dish and ingredients are stored in Graph Database (Neo4j).
3. A recommendation engine that uses collaborative filtering and content-based filtering to suggest dishes.
4. A web application interface for users to input their preferences and view recommendations (Streamlit).

## Technologies Used
- Data Storage: Graph Database (Neo4j)
- Recommendation Engine: Collaborative Filtering, Content-Based Filtering (Surprise, Scikit-learn)
- Web Application: Streamlit, FastAPI
- AI Models:
  - World Models: Gemini API
  - Image Embedding: CLIP

## Database Design

### Entities
- User: Represents a user of the system.
  - Attributes: user_id, name, email
- Dish: Represents a food dish.
  - Attributes: dish_id, name, description, image_url, image_embedding
- Ingredient: Represents an ingredient used in dishes.
  - Attributes: ingredient_id, name
- Country / Region: Represents the origin of the dish.
  - Attributes: country_id, name
- Dietary Restriction: Represents dietary restrictions (e.g., vegan, gluten-free).
  - Attributes: restriction_id, name

### Relationships
- User -[BOOKED]-> Dish: Represents a user booking a dish.
- User -[RATED]-> Dish: Represents a user rating a dish.
  - Attributes: rating (1-5), review (default to 4 if a user has booked a dish but not rated it)
- Dish -[CONTAINS]-> Ingredient: Represents the ingredients used in a dish.
- Dish -[ORIGINATES_FROM]-> Country / Region: Represents the origin of the dish.
- User -[PREFERS]-> Ingredient: Represents a user's preference for certain ingredients.
- User -[AVOIDS]-> Ingredient: Represents a user's dietary restrictions.
- Ingredient -[SUITED_FOR]-> Dietary Restriction: Represents which ingredients are suitable for certain dietary restrictions.
- Ingredient -[NOT_SUITED_FOR]-> Dietary Restriction: Represents which ingredients are not suitable for certain dietary restrictions.
- User -[FOLLOWS]-> Dietary Restriction: Represents a user's dietary restrictions.

## Use Cases:

### 1. Dish Registration
- Input: Dish image, name, description, origin country/region
- Process:
  - Extract ingredients using Gemini API.
  - Generate image embedding using CLIP.
  - Store dish, ingredients, and relationships in Neo4j.
- Output: Confirmation of dish registration.