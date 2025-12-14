"""Similarity metrics for dish comparison.

Implements three different similarity measures:
1. Jaccard similarity based on ingredient sets
2. Cosine similarity based on averaged ingredient embeddings
3. Cosine similarity based on dish image embeddings
"""

import numpy as np
from numpy.typing import NDArray

from src.visualization.dish_aggregator import get_aggregator

aggregator = get_aggregator(method='tfidf')

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
