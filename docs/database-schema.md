# Database Schema

The system uses Neo4j graph database to store dishes, ingredients, and their relationships.

## Node Labels

### Dish

Represents a food dish.

| Property | Type | Description |
|----------|------|-------------|
| `dish_id` | String | Unique identifier (UUID) |
| `name` | String | Name of the dish |
| `description` | String | Optional description |
| `image_url` | String | Path to the dish image |
| `image_embedding` | List[Float] | CLIP embedding vector (512 dimensions) |
| `created_at` | DateTime | When the dish was created |
| `updated_at` | DateTime | When the dish was last updated |

### Ingredient

Represents an ingredient used in dishes.

| Property | Type | Description |
|----------|------|-------------|
| `name` | String | Ingredient name (lowercase, trimmed) |
| `embedding` | List[Float] | Embedding vector (512 dimensions) |
| `created_at` | DateTime | When the ingredient was created |

### Country

Represents the origin country/cuisine of a dish.

| Property | Type | Description |
|----------|------|-------------|
| `name` | String | Country or cuisine name |
| `created_at` | DateTime | When created |

### User (Planned)

Represents a system user.

| Property | Type | Description |
|----------|------|-------------|
| `user_id` | String | Unique identifier |
| `name` | String | User's name |
| `email` | String | User's email |

### DietaryRestriction (Planned)

Represents dietary restrictions (e.g., vegan, gluten-free).

| Property | Type | Description |
|----------|------|-------------|
| `name` | String | Restriction name (e.g., "vegetarian", "dairy-allergy") |
| `description` | String | Human-readable description |
| `created_at` | DateTime | When created |
| `updated_at` | DateTime | When last updated |

## Relationships

### Currently Implemented

```
(Dish)-[:CONTAINS]->(Ingredient)
(Dish)-[:ORIGINATES_FROM]->(Country)
(Ingredient)-[:SUITED_FOR]->(DietaryRestriction)
(Ingredient)-[:NOT_SUITED_FOR]->(DietaryRestriction)
```

### Planned for Recommendation System

```
(User)-[:RATED {rating: 1-5}]->(Dish)
(User)-[:FOLLOWS]->(DietaryRestriction)
```

## Seeded Dietary Restrictions

The following dietary restrictions are pre-seeded with ingredient relationships:

| Name | Description |
|------|-------------|
| `dairy-allergy` | Avoids all dairy products including milk, cheese, and butter |
| `seafood-allergy` | Avoids all seafood including fish and shellfish |
| `peanut-allergy` | Avoids peanuts and peanut-derived products |
| `vegetarian` | No meat or fish, but eggs and dairy are allowed |
| `low-carb` | Limits high-carbohydrate foods for keto or low-carb diets |

Run `python main.py --seed-restrictions` to seed these restrictions.

## Constraints

Unique constraints ensure data integrity and enable efficient MERGE operations:

```cypher
CREATE CONSTRAINT dish_id_unique IF NOT EXISTS 
  FOR (d:Dish) REQUIRE d.dish_id IS UNIQUE;

CREATE CONSTRAINT ingredient_name_unique IF NOT EXISTS 
  FOR (i:Ingredient) REQUIRE i.name IS UNIQUE;

CREATE CONSTRAINT country_name_unique IF NOT EXISTS 
  FOR (c:Country) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT user_id_unique IF NOT EXISTS 
  FOR (u:User) REQUIRE u.user_id IS UNIQUE;

CREATE CONSTRAINT restriction_name_unique IF NOT EXISTS 
  FOR (r:DietaryRestriction) REQUIRE r.name IS UNIQUE;
```

Run `python main.py --init-db` to create these constraints.

## Example Queries

### Find all ingredients for a dish

```cypher
MATCH (d:Dish {dish_id: $dish_id})-[:CONTAINS]->(i:Ingredient)
RETURN d.name AS dish, collect(i.name) AS ingredients
```

### Find dishes containing a specific ingredient

```cypher
MATCH (d:Dish)-[:CONTAINS]->(i:Ingredient {name: "garlic"})
RETURN d.name, d.description
```

### Find all Italian dishes

```cypher
MATCH (d:Dish)-[:ORIGINATES_FROM]->(c:Country {name: "Italian"})
RETURN d.name, d.description
```

### Count ingredients per dish

```cypher
MATCH (d:Dish)-[:CONTAINS]->(i:Ingredient)
RETURN d.name, count(i) AS ingredient_count
ORDER BY ingredient_count DESC
```

### Find common ingredients between dishes

```cypher
MATCH (d1:Dish {name: "Pizza"})-[:CONTAINS]->(i:Ingredient)<-[:CONTAINS]-(d2:Dish)
WHERE d1 <> d2
RETURN d2.name, collect(i.name) AS common_ingredients
```

## Data Model Diagram

```
                    ┌─────────────┐
                    │    User     │
                    └─────────────┘
                          │
         ┌────────────────┼────────────────┐
         │ BOOKED         │ RATED          │ FOLLOWS
         │                │                │
         ▼                ▼                ▼
┌─────────────┐    ┌─────────────┐  ┌─────────────────┐
│    Dish     │    │    Dish     │  │DietaryRestriction│
└─────────────┘    └─────────────┘  └─────────────────┘
         │                                    ▲
         │ CONTAINS                           │ SUITED_FOR /
         ▼                                    │ NOT_SUITED_FOR
┌─────────────┐                               │
│ Ingredient  │───────────────────────────────┘
└─────────────┘
         │
         │ ORIGINATES_FROM
         ▼
┌─────────────┐
│   Country   │
└─────────────┘
```
