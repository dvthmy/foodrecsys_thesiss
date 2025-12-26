"""Similarity metrics for dish comparison.

Implements three different similarity measures:
1. Jaccard similarity based on ingredient sets
2. Cosine similarity based on averaged ingredient embeddings
3. Cosine similarity based on dish image embeddings
"""

import numpy as np
from numpy.typing import NDArray
from src.services.neo4j_service import Neo4jService
from src.visualization.dish_aggregator import get_aggregator

aggregator = get_aggregator(method='tfidf')


def get_tfidf_weights_for_dish(
    ingredients: list[str],
) -> list[dict[str, float]]:
    """Get TF-IDF weights for each ingredient in a dish.
    
    Args:
        ingredients: List of ingredient names.
        
    Returns:
        List of dicts with 'ingredient' and 'tfidf_weight' keys, sorted by weight descending.
    """
    weights = []
    for name in ingredients:
        clean_name = name.lower().strip()
        weight = aggregator.idf_map.get(clean_name, 1.0)
        weights.append({
            "ingredient": name,
            "tfidf_weight": weight,
        })
    
    # Sort by weight descending
    weights.sort(key=lambda x: x["tfidf_weight"], reverse=True)
    return weights

def initialize_aggregator():
    neo4j = Neo4jService()
    aggregator = get_aggregator(method='tfidf')

    # 1. Fetch all recipes (dish ingredients) for IDF calculation
    # (You likely already have this part)
    dishes_data = neo4j.get_all_dishes_ingredients()
    all_recipes = [d['ingredients'] for d in dishes_data]

    # 2. Fetch all unique ingredient embeddings for Global Mean Centering
    print("Fetching ingredient embeddings from Neo4j...")
    all_names, all_embeddings = neo4j.get_all_ingredient_embeddings()
    
    # 3. Fit the aggregator with both datasets
    # This calculates IDF *and* the Global Mean vector
    aggregator.fit_idf(
        all_recipes=all_recipes, 
        all_ingredient_embeddings=all_embeddings
    )
    print(f"Aggregator initialized. Global mean computed from {len(all_embeddings)} ingredients.")
    neo4j.close()

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set_a: First set of elements.
        set_b: Second set of elements.

    Returns:
        Jaccard similarity coefficient (0.0 to 1.0).
    """
    if not set_a and not set_b:
        return 1.0  # Two empty sets are identical
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def cosine_similarity(vec_a: NDArray[np.float64], vec_b: NDArray[np.float64]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity (-1.0 to 1.0).
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_jaccard_matrix(
    dishes: list[dict],
) -> tuple[NDArray[np.float64], list[str]]:
    """Compute pairwise Jaccard similarity matrix for dishes based on ingredients.

    Args:
        dishes: List of dish dictionaries with 'name' and 'ingredients' keys.
            Each dish must have 'ingredients' as a list of ingredient names.

    Returns:
        Tuple of:
            - NxN similarity matrix as numpy array
            - List of dish names (labels for matrix axes)
    """
    n = len(dishes)
    matrix = np.zeros((n, n), dtype=np.float64)
    names = [dish["name"] for dish in dishes]

    # Convert ingredient lists to sets for efficient comparison
    ingredient_sets = [set(dish.get("ingredients", [])) for dish in dishes]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                sim = jaccard_similarity(ingredient_sets[i], ingredient_sets[j])
                matrix[i, j] = sim
                matrix[j, i] = sim

    return matrix, names


def compute_ingredient_embedding_matrix(
    dishes: list[dict],
) -> tuple[NDArray[np.float64], list[str], list[str]]:
    """Compute pairwise cosine similarity based on TF-IDF weighted ingredient embeddings.

    For each dish, computes the TF-IDF weighted aggregation of all ingredient embeddings,
    then calculates pairwise cosine similarity between dishes.

    Args:
        dishes: List of dish dictionaries with keys:
            - 'name': Dish name
            - 'ingredients': List of ingredient names (list[str])
            - 'ingredient_embeddings': List of embedding vectors (list[list[float]])

    Returns:
        Tuple of:
            - NxN similarity matrix as numpy array
            - List of dish names included in the matrix
            - List of dish names skipped (missing embeddings)
    """
    valid_dishes = []
    skipped_names = []

    for dish in dishes:
        embeddings = dish.get("ingredient_embeddings", [])
        ingredients = dish.get("ingredients", [])
        
        # Filter out None or empty embeddings, keeping aligned ingredient names
        valid_embeddings = []
        valid_ingredients = []
        for i, e in enumerate(embeddings):
            if e is not None and len(e) > 0:
                valid_embeddings.append(e)
                if i < len(ingredients):
                    valid_ingredients.append(ingredients[i])
                else:
                    valid_ingredients.append(f"ingredient_{i}")

        if valid_embeddings:
            # Use TF-IDF weighted aggregation instead of simple average
            dish_embedding = aggregator.aggregate(valid_embeddings, valid_ingredients)
            valid_dishes.append({
                "name": dish["name"],
                "embedding": np.array(dish_embedding, dtype=np.float64),
            })
        else:
            skipped_names.append(dish["name"])

    n = len(valid_dishes)
    if n == 0:
        return np.array([]), [], skipped_names

    matrix = np.zeros((n, n), dtype=np.float64)
    names = [dish["name"] for dish in valid_dishes]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                sim = cosine_similarity(
                    valid_dishes[i]["embedding"],
                    valid_dishes[j]["embedding"],
                )
                matrix[i, j] = sim
                matrix[j, i] = sim

    return matrix, names, skipped_names


def compute_image_embedding_matrix(
    dishes: list[dict],
) -> tuple[NDArray[np.float64], list[str], list[str]]:
    """Compute pairwise cosine similarity based on dish image embeddings.

    Args:
        dishes: List of dish dictionaries with keys:
            - 'name': Dish name
            - 'image_embedding': CLIP image embedding vector (list[float])

    Returns:
        Tuple of:
            - NxN similarity matrix as numpy array
            - List of dish names included in the matrix
            - List of dish names skipped (missing embeddings)
    """
    valid_dishes = []
    skipped_names = []

    for dish in dishes:
        embedding = dish.get("image_embedding")
        if embedding is not None and len(embedding) > 0:
            valid_dishes.append({
                "name": dish["name"],
                "embedding": np.array(embedding, dtype=np.float64),
            })
        else:
            skipped_names.append(dish["name"])

    n = len(valid_dishes)
    if n == 0:
        return np.array([]), [], skipped_names

    matrix = np.zeros((n, n), dtype=np.float64)
    names = [dish["name"] for dish in valid_dishes]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                sim = cosine_similarity(
                    valid_dishes[i]["embedding"],
                    valid_dishes[j]["embedding"],
                )
                matrix[i, j] = sim
                matrix[j, i] = sim

    return matrix, names, skipped_names


def get_ingredient_embeddings_batch(
    neo4j_service: Neo4jService,
    ingredient_names: list[str],
) -> dict[str, list[float]]:
    """Fetch embeddings for multiple ingredients in batch.
    
    Args:
        neo4j_service: Neo4j service instance.
        ingredient_names: List of ingredient names to fetch embeddings for.
        
    Returns:
        Dictionary mapping ingredient name to embedding vector.
        Only includes ingredients that have embeddings.
    """
    if not ingredient_names:
        return {}
    
    query = """
    MATCH (i:Ingredient)
    WHERE i.name IN $names AND i.embedding IS NOT NULL
    RETURN i.name AS name, i.embedding AS embedding
    """
    
    with neo4j_service.session() as session:
        result = session.run(query, names=[name.lower().strip() for name in ingredient_names])
        return {record["name"]: record["embedding"] for record in result}


def compute_ingredient_comparison_jaccard(
    dish_A: dict,
    dish_B: dict,
) -> dict:
    """Compute ingredient comparison for Jaccard similarity.
    
    For Jaccard, we only show:
    - Shared ingredients (exact matches)
    - Unique ingredients to each dish
    
    Args:
        dish_A: Dish dictionary with 'name' and 'ingredients' keys.
        dish_B: Dish dictionary with 'name' and 'ingredients' keys.
        
    Returns:
        Dictionary with:
            - shared_ingredients: List of shared ingredient names
            - unique_to_A: List of ingredients only in dish A
            - unique_to_B: List of ingredients only in dish B
            - overall_similarity: Jaccard similarity score
    """
    ingredients_A = set(dish_A.get("ingredients", []))
    ingredients_B = set(dish_B.get("ingredients", []))
    
    shared = ingredients_A & ingredients_B
    unique_A = ingredients_A - shared
    unique_B = ingredients_B - shared
    
    # Calculate Jaccard similarity
    overall_similarity = jaccard_similarity(ingredients_A, ingredients_B)
    
    return {
        "shared_ingredients": sorted(list(shared)),
        "unique_to_A": sorted(list(unique_A)),
        "unique_to_B": sorted(list(unique_B)),
        "overall_similarity": overall_similarity,
    }


def compute_ingredient_comparison_embedding(
    dish_A: dict,
    dish_B: dict,
    neo4j_service: Neo4jService,
    similarity_threshold: float = 0.85,
    max_similar_pairs: int = 15,
) -> dict:
    """Compute ingredient comparison for Ingredient Embedding similarity.
    
    For Ingredient Embedding, we show:
    - Shared ingredients (exact matches)
    - Similar ingredient pairs (using embeddings)
    - Unique ingredients remaining after matching
    
    Note: Default threshold 0.85 is higher than typical (0.7) because ingredients
    are embedded with generic "food ingredient" description, causing false positives
    between unrelated ingredients (e.g., "chicken" vs "spices").
    
    Args:
        dish_A: Dish dictionary with 'name' and 'ingredients' keys.
        dish_B: Dish dictionary with 'name' and 'ingredients' keys.
        neo4j_service: Neo4j service instance for fetching embeddings.
        similarity_threshold: Minimum similarity score to consider ingredients similar.
            Default 0.85 to filter out false positives.
        max_similar_pairs: Maximum number of similar pairs to return.
        
    Returns:
        Dictionary with:
            - shared_ingredients: List of shared ingredient names
            - similar_pairs: List of dicts with 'ingredient_A', 'ingredient_B', 'similarity'
            - unique_to_A: List of ingredients only in dish A (after matching similar pairs)
            - unique_to_B: List of ingredients only in dish B (after matching similar pairs)
            - overall_similarity: Overall dish similarity score (if available)
    """
    ingredients_A = set(dish_A.get("ingredients", []))
    ingredients_B = set(dish_B.get("ingredients", []))
    
    # Find exact matches
    shared = ingredients_A & ingredients_B
    
    # Remaining ingredients after removing exact matches
    remaining_A = ingredients_A - shared
    remaining_B = ingredients_B - shared
    
    # Fetch embeddings for remaining ingredients
    all_remaining = list(remaining_A | remaining_B)
    embeddings_map = get_ingredient_embeddings_batch(neo4j_service, all_remaining)
    
    # Find similar pairs using embeddings
    similar_pairs = []
    matched_A = set()
    matched_B = set()
    
    for ing_A in remaining_A:
        ing_A_normalized = ing_A.lower().strip()
        if ing_A_normalized not in embeddings_map:
            continue
            
        embedding_A = np.array(embeddings_map[ing_A_normalized], dtype=np.float64)
        # Normalize embedding to ensure consistent cosine similarity calculation
        norm_A = np.linalg.norm(embedding_A)
        if norm_A > 0:
            embedding_A = embedding_A / norm_A
        else:
            continue  # Skip zero vectors
        
        best_match = None
        best_score = 0.0
        
        for ing_B in remaining_B:
            ing_B_normalized = ing_B.lower().strip()
            if ing_B_normalized not in embeddings_map:
                continue
            if ing_B in matched_B:
                continue  # Already matched
                
            embedding_B = np.array(embeddings_map[ing_B_normalized], dtype=np.float64)
            # Normalize embedding to ensure consistent cosine similarity calculation
            norm_B = np.linalg.norm(embedding_B)
            if norm_B > 0:
                embedding_B = embedding_B / norm_B
            else:
                continue  # Skip zero vectors
                
            # Since embeddings are already normalized (norm=1.0), 
            # cosine similarity = dot product (no need to divide by norms)
            # But we use cosine_similarity() for safety and consistency
            sim = cosine_similarity(embedding_A, embedding_B)
            
            if sim > best_score and sim >= similarity_threshold:
                best_score = sim
                best_match = ing_B
        
        if best_match:
            similar_pairs.append({
                "ingredient_A": ing_A,
                "ingredient_B": best_match,
                "similarity": best_score,
            })
            matched_A.add(ing_A)
            matched_B.add(best_match)
    
    # Sort by similarity descending
    similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    similar_pairs = similar_pairs[:max_similar_pairs]
    
    # Remaining unique ingredients after matching
    unique_A = remaining_A - matched_A
    unique_B = remaining_B - matched_B
    
    return {
        "shared_ingredients": sorted(list(shared)),
        "similar_pairs": similar_pairs,
        "unique_to_A": sorted(list(unique_A)),
        "unique_to_B": sorted(list(unique_B)),
        "overall_similarity": None,  # Will be set by caller if available
    }


def compute_ingredient_comparison(
    dish_A: dict,
    dish_B: dict,
    metric: str,
    neo4j_service: Neo4jService | None = None,
    similarity_threshold: float = 0.8,
    overall_similarity: float | None = None,
) -> dict:
    """Compute ingredient comparison between two dishes.
    
    Args:
        dish_A: Dish dictionary with 'name' and 'ingredients' keys.
        dish_B: Dish dictionary with 'name' and 'ingredients' keys.
        metric: Similarity metric ('jaccard' or 'ingredient_embedding').
        neo4j_service: Neo4j service instance (required for ingredient_embedding metric).
        similarity_threshold: Minimum similarity for ingredient pairs (ingredient_embedding only).
            Default 0.85 to filter out false positives from generic "food ingredient" descriptions.
        overall_similarity: Overall dish similarity score to include in result.
        
    Returns:
        Dictionary with comparison results (structure depends on metric).
    """
    if metric == "jaccard":
        result = compute_ingredient_comparison_jaccard(dish_A, dish_B)
    elif metric == "ingredient_embedding":
        if neo4j_service is None:
            raise ValueError("neo4j_service is required for ingredient_embedding metric")
        result = compute_ingredient_comparison_embedding(
            dish_A, dish_B, neo4j_service, similarity_threshold
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    if overall_similarity is not None:
        result["overall_similarity"] = overall_similarity
    
    return result