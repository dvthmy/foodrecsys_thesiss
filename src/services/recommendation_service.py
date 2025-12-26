


"""Recommendation service for collaborative filtering.

Implements user-based collaborative filtering to recommend dishes
based on similar users' preferences.
"""

import logging
from dataclasses import dataclass, field
from typing import Any
logger = logging.getLogger(__name__)

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from src.services.neo4j_service import Neo4jService
from src.visualization.dish_aggregator import get_aggregator


# Minimum similarity threshold for considering users as "similar"
SIMILARITY_THRESHOLD = 0.5

# Minimum number of ratings for collaborative filtering (cold start threshold)
COLD_START_THRESHOLD = 3


@dataclass
class SimilarUser:
    """Represents a similar user with similarity score."""

    user_id: str
    name: str
    similarity: float
    shared_dishes: int = 0


@dataclass
class DishRecommendation:
    """Represents a recommended dish with prediction details."""

    dish_id: str
    name: str
    predicted_score: float
    description: str | None = None
    ingredients: list[str] = field(default_factory=list)
    recommender_count: int = 0
    reason: str = "collaborative_filtering"


class RecommendationService:
    """Service for generating dish recommendations using collaborative filtering."""

    def __init__(self, neo4j_service: Neo4jService | None = None):
        """Initialize the recommendation service.

        Args:
            neo4j_service: Neo4j service instance. Creates one if not provided.
        """
        self._neo4j = neo4j_service

        # Cached data for similarity computation
        self._user_ids: list[str] = []
        self._dish_ids: list[str] = []
        self._user_index: dict[str, int] = {}
        self._dish_index: dict[str, int] = {}
        self._rating_matrix: NDArray[np.float64] | None = None
        self._user_similarity_matrix: NDArray[np.float64] | None = None
        self._user_names: dict[str, str] = {}
        
        # Dish aggregator for ingredient-based recommendations
        self._aggregator = None
        self._aggregator_fitted = False

    @property
    def neo4j(self) -> Neo4jService:
        """Get or create Neo4j service instance."""
        if self._neo4j is None:
            self._neo4j = Neo4jService()
        return self._neo4j
    
    def _ensure_aggregator_fitted(self) -> None:
        """Ensure dish aggregator is initialized and fitted with IDF."""
        if self._aggregator is None:
            self._aggregator = get_aggregator(method='tfidf')
        
        if not self._aggregator_fitted:
            # Fetch all recipes for IDF calculation
            dishes_data = self.neo4j.get_all_dishes_ingredients()
            all_recipes = [d.get('ingredients', []) for d in dishes_data]
            
            # Fetch all ingredient embeddings for global mean
            _, all_embeddings = self.neo4j.get_all_ingredient_embeddings()
            
            # Fit the aggregator
            self._aggregator.fit_idf(
                all_recipes=all_recipes,
                all_ingredient_embeddings=all_embeddings if all_embeddings else None
            )
            self._aggregator_fitted = True
            logging.info("Dish aggregator fitted with IDF scores")

    def _build_rating_matrix(self) -> None:
        """Build user-dish rating matrix from database."""
        ratings = self.neo4j.get_all_ratings()

        if not ratings:
            logging.warning("No ratings found in database")
            self._rating_matrix = np.array([])
            return

        # Build indices
        self._user_ids = list(set(r["user_id"] for r in ratings))
        self._dish_ids = list(set(r["dish_id"] for r in ratings))
        self._user_index = {uid: i for i, uid in enumerate(self._user_ids)}
        self._dish_index = {did: i for i, did in enumerate(self._dish_ids)}

        # Initialize matrix with zeros (unrated)
        n_users = len(self._user_ids)
        n_dishes = len(self._dish_ids)
        self._rating_matrix = np.zeros((n_users, n_dishes), dtype=np.float64)

        # Fill in ratings
        for r in ratings:
            user_idx = self._user_index[r["user_id"]]
            dish_idx = self._dish_index[r["dish_id"]]
            self._rating_matrix[user_idx, dish_idx] = r["score"]

        logging.info(
            "Built rating matrix: %d users x %d dishes, %d ratings",
            n_users,
            n_dishes,
            len(ratings),
        )

    
    def _compute_user_similarity(self) -> None:
        """Compute user-user similarity matrix using Cosine similarity."""
        if self._rating_matrix is None or self._rating_matrix.size == 0:
            self._build_rating_matrix()

        if self._rating_matrix is None or self._rating_matrix.size == 0:
            self._user_similarity_matrix = np.array([])
            return

        n_users = len(self._user_ids)
        self._user_similarity_matrix = np.zeros((n_users, n_users), dtype=np.float64)

        # Compute cosine similarity between users
        for i in range(n_users):
            for j in range(i, n_users):
                if i == j:
                    self._user_similarity_matrix[i, j] = 1.0
                else:
                    sim = self._cosine_similarity(
                        self._rating_matrix[i],
                        self._rating_matrix[j],
                    )
                    self._user_similarity_matrix[i, j] = sim
                    self._user_similarity_matrix[j, i] = sim

        logging.info("Computed user similarity matrix (cosine): %d x %d", n_users, n_users)

    def _cosine_similarity(
        self,
        vec1: NDArray[np.float64],
        vec2: NDArray[np.float64],
    ) -> float:
        """Compute cosine similarity between two rating vectors.

        Only considers items that both users have rated (non-zero).

        Args:
            vec1: First user's rating vector.
            vec2: Second user's rating vector.

        Returns:
            Cosine similarity score between 0 and 1.
        """
        # Find common rated items (both non-zero)
        common_mask = (vec1 > 0) & (vec2 > 0)

        if not np.any(common_mask):
            return 0.0

        v1 = vec1[common_mask]
        v2 = vec2[common_mask]

        # Mean-center the ratings for better similarity
        v1_centered = v1 - np.mean(v1)
        v2_centered = v2 - np.mean(v2)

        norm1 = np.linalg.norm(v1_centered)
        norm2 = np.linalg.norm(v2_centered)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1_centered, v2_centered) / (norm1 * norm2))

    def refresh_cache(self) -> None:
        """Refresh the cached rating matrix and similarity matrix."""
        self._rating_matrix = None
        self._user_similarity_matrix = None
        self._build_rating_matrix()
        self._compute_user_similarity()

        # Cache user names
        users = self.neo4j.get_all_users(limit=1000)
        self._user_names = {u["user_id"]: u["name"] for u in users}

    def find_similar_users(
        self,
        user_id: str,
        k: int = 3,
        min_similarity: float = SIMILARITY_THRESHOLD,
    ) -> list[SimilarUser]:
        """Find users most similar to the target user using Cosine similarity.

        Args:
            user_id: Target user ID.
            k: Maximum number of similar users to return.
            min_similarity: Minimum similarity threshold (default > 0.5).

        Returns:
            List of SimilarUser objects sorted by similarity.
        """
        # Ensure similarity matrix is computed
        if self._user_similarity_matrix is None:
            self.refresh_cache()
        
        self._compute_user_similarity()

        if self._user_similarity_matrix is None or self._rating_matrix is None:
            logging.warning("Failed to build similarity matrix")
            return []

        if user_id not in self._user_index:
            logging.warning("User %s not found in rating matrix", user_id)
            return []

        user_idx = self._user_index[user_id]
        similarities = self._user_similarity_matrix[user_idx]

        # Get user's rating vector for counting shared dishes
        user_ratings = self._rating_matrix[user_idx]

        similar_users = []
        for other_idx, sim in enumerate(similarities):
            if other_idx == user_idx:
                continue
            if sim > min_similarity:
                other_user_id = self._user_ids[other_idx]

                # Count shared rated dishes
                other_ratings = self._rating_matrix[other_idx]
                shared = int(np.sum((user_ratings > 0) & (other_ratings > 0)))

                similar_users.append(
                    SimilarUser(
                        user_id=other_user_id,
                        name=self._user_names.get(other_user_id, "Unknown"),
                        similarity=float(sim),
                        shared_dishes=shared,
                    )
                )

        # Sort by similarity descending
        similar_users.sort(key=lambda x: x.similarity, reverse=True)

        return similar_users[:k]

    def get_user_rating_count(self, user_id: str) -> int:
        """Get the number of ratings a user has made.

        Args:
            user_id: The user identifier.

        Returns:
            Number of ratings.
        """
        if self._rating_matrix is None:
            self.refresh_cache()

        if self._rating_matrix is None:
            return 0

        if user_id not in self._user_index:
            return 0

        user_idx = self._user_index[user_id]
        return int(np.sum(self._rating_matrix[user_idx] > 0))

    def recommend_dishes(
        self,
        user_id: str,
        k: int = 3,
        apply_dietary_filter: bool = True,
        metric: str = "cosine",  
    ) -> list[DishRecommendation]:
        """Recommend dishes for a user using collaborative filtering (Cosine similarity).

        For users with few ratings (cold start), falls back to popular dishes
        filtered by dietary restrictions.

        Args:
            user_id: Target user ID.
            k: Maximum number of recommendations.
            apply_dietary_filter: Whether to filter by user's dietary restrictions.
            metric: Deprecated, only Cosine similarity is used.

        Returns:
            List of DishRecommendation objects.
        """
        # Check for cold start
        rating_count = self.get_user_rating_count(user_id)

        if rating_count < COLD_START_THRESHOLD:
            logging.info(
                "User %s has only %d ratings (< %d), using popular dish fallback",
                user_id,
                rating_count,
                COLD_START_THRESHOLD,
            )
            return self._recommend_popular_dishes(
                user_id=user_id,
                k=k,
                apply_dietary_filter=apply_dietary_filter,
            )

        return self._recommend_collaborative(
            user_id=user_id,
            k=k,
            apply_dietary_filter=apply_dietary_filter,
        )

    def recommend_image_by_image(
        self,
        user_id: str,
        k: int = 3,
        apply_dietary_filter: bool = True,
    ) -> list[DishRecommendation]:
        """Recommend dishes using image-based similarity (CLIP embeddings).

        Uses CLIP image embeddings to find dishes visually similar to what the user likes.

        Args:
            user_id: Target user ID.
            k: Maximum number of recommendations.
            apply_dietary_filter: Whether to filter by dietary restrictions.

        Returns:
            List of DishRecommendation objects.
        """
        # 1. Get image embeddings of dishes liked by user
        liked_embeddings = self.neo4j.get_user_liked_dish_embeddings(user_id, min_rating=4)
        
        if not liked_embeddings:
            logging.info("No liked dishes with image embeddings found for %s", user_id)
            return []

        # 2. Compute user profile vector (mean of liked dish image embeddings)
        user_profile = np.mean(liked_embeddings, axis=0).tolist()

        # 3. Find similar dishes using vector search on image embeddings
        # We fetch more candidates to allow for filtering
        raw_recs = self.neo4j.find_similar_dishes_by_embedding(
            embedding=user_profile,
            k=k * 3,
            threshold=0.0, # Return top k regardless of threshold
        )
        
        # Filter out dishes the user has already rated
        rated_ids = self.neo4j.get_rated_dish_ids(user_id)
        raw_recs = [d for d in raw_recs if d["dish_id"] not in rated_ids]

        if not raw_recs:
            logging.info("No image-based recommendations found for %s", user_id)
            return []

        # Apply dietary restriction filter
        if apply_dietary_filter:
            restrictions = self.neo4j.get_user_dietary_restrictions(user_id)
            if restrictions:
                candidate_dish_ids = [d["dish_id"] for d in raw_recs]
                safe_dish_ids = set(
                    self.neo4j.filter_dishes_by_restrictions(
                        dish_ids=candidate_dish_ids,
                        restriction_names=restrictions,
                    )
                )
                raw_recs = [d for d in raw_recs if d["dish_id"] in safe_dish_ids]

        recommendations = []
        for dish in raw_recs[:k]:
            recommendations.append(
                DishRecommendation(
                    dish_id=dish["dish_id"],
                    name=dish["name"],
                    predicted_score=round(dish["score"], 3),
                    description=dish.get("description"),
                    ingredients=dish.get("ingredients", []),
                    recommender_count=0,
                    reason="image_based",
                )
            )

        return recommendations

    def recommend_content_based(
        self,
        user_id: str,
        k: int = 3,
        apply_dietary_filter: bool = True,
        metric: str = "jaccard",
    ) -> list[DishRecommendation]:
        """Recommend dishes using content-based filtering.

        Args:
            user_id: Target user ID.
            k: Maximum number of recommendations.
            apply_dietary_filter: Whether to filter by dietary restrictions.
            metric: Similarity metric:
                - 'jaccard': Jaccard similarity on ingredient sets
                - 'ingredient_embedding': Semantic similarity using Gemma ingredient embeddings + TF-IDF aggregation

        Returns:
            List of DishRecommendation objects.
        """
        if metric == "ingredient_embedding":
            # Ensure aggregator is fitted
            self._ensure_aggregator_fitted()
            
            # 1. Get liked dishes with ingredient embeddings
            liked_dishes = self.neo4j.get_user_liked_dishes_with_ingredient_embeddings(
                user_id, min_rating=4
            )
            
            if not liked_dishes:
                logging.info("No liked dishes with ingredient embeddings found for %s", user_id)
                return []
            
            # 2. Aggregate ingredient embeddings for each liked dish to create dish embeddings
            dish_embeddings = []
            for dish in liked_dishes:
                ingredient_embeddings = dish.get("ingredient_embeddings", [])
                ingredients = dish.get("ingredients", [])
                
                # Filter out None or empty embeddings
                valid_embeddings = []
                valid_ingredients = []
                for i, emb in enumerate(ingredient_embeddings):
                    if emb is not None and len(emb) > 0:
                        valid_embeddings.append(emb)
                        if i < len(ingredients):
                            valid_ingredients.append(ingredients[i])
                
                if valid_embeddings:
                    # Aggregate ingredient embeddings into dish embedding
                    dish_emb = self._aggregator.aggregate(valid_embeddings, valid_ingredients)
                    dish_embeddings.append(dish_emb)
            
            if not dish_embeddings:
                logging.info("No valid dish embeddings computed for %s", user_id)
                return []
            
            # 3. Compute user profile vector (mean of aggregated dish embeddings)
            user_profile = np.array(np.mean(dish_embeddings, axis=0))
            
            # 4. Get all candidate dishes with ingredient embeddings
            # Fetch all dishes that user hasn't rated
            rated_ids = self.neo4j.get_rated_dish_ids(user_id)
            all_dishes = self.neo4j.get_all_dishes_with_ingredient_embeddings()
            candidate_dishes = [d for d in all_dishes if d["dish_id"] not in rated_ids]
            
            if not candidate_dishes:
                logging.info("No candidate dishes found for %s", user_id)
                return []
            
            # 5. Compute similarity scores in Python
            similarities = []
            for dish in candidate_dishes:
                ingredient_embeddings = dish.get("ingredient_embeddings", [])
                ingredients = dish.get("ingredients", [])
                
                # Filter and aggregate
                valid_embeddings = []
                valid_ingredients = []
                for i, emb in enumerate(ingredient_embeddings):
                    if emb is not None and len(emb) > 0:
                        valid_embeddings.append(emb)
                        if i < len(ingredients):
                            valid_ingredients.append(ingredients[i])
                
                if valid_embeddings:
                    dish_emb = np.array(self._aggregator.aggregate(valid_embeddings, valid_ingredients))
                    # Compute cosine similarity
                    norm_user = np.linalg.norm(user_profile)
                    norm_dish = np.linalg.norm(dish_emb)
                    if norm_user > 0 and norm_dish > 0:
                        similarity = float(np.dot(user_profile, dish_emb) / (norm_user * norm_dish))
                        similarities.append({
                            "dish_id": dish["dish_id"],
                            "name": dish["name"],
                            "description": dish.get("description"),
                            "ingredients": dish.get("ingredients", []),
                            "score": similarity
                        })
            
            # Sort by similarity descending
            similarities.sort(key=lambda x: x["score"], reverse=True)
            raw_recs = similarities[:k * 5]  # Get more candidates for filtering
            
        else:
            # Default to Jaccard on ingredients
            raw_recs = self.neo4j.get_content_based_recommendations(
                user_id=user_id,
                limit=k * 5,
                min_rating=4,
            )

        if not raw_recs:
            logging.info("No content-based recommendations found for %s", user_id)
            return []

        # Apply dietary restriction filter
        if apply_dietary_filter:
            restrictions = self.neo4j.get_user_dietary_restrictions(user_id)
            if restrictions:
                candidate_dish_ids = [d["dish_id"] for d in raw_recs]
                safe_dish_ids = set(
                    self.neo4j.filter_dishes_by_restrictions(
                        dish_ids=candidate_dish_ids,
                        restriction_names=restrictions,
                    )
                )
                raw_recs = [d for d in raw_recs if d["dish_id"] in safe_dish_ids]

        recommendations = []
        for dish in raw_recs[:k]:
            recommendations.append(
                DishRecommendation(
                    dish_id=dish["dish_id"],
                    name=dish["name"],
                    predicted_score=round(dish["score"], 3),
                    description=dish.get("description"),
                    ingredients=dish.get("ingredients", []),
                    recommender_count=0,
                    reason=f"content_based_{metric}",
                )
            )

        return recommendations
    

    def _recommend_popular_dishes(
        self,
        user_id: str,
        k: int = 3,
        apply_dietary_filter: bool = True,
    ) -> list[DishRecommendation]:
        """Recommend popular dishes for cold start users.

        Args:
            user_id: Target user ID.
            k: Maximum number of recommendations.
            apply_dietary_filter: Whether to filter by dietary restrictions.

        Returns:
            List of popular DishRecommendation objects.
        """
        # Get user's dietary restrictions
        restrictions = []
        if apply_dietary_filter:
            restrictions = self.neo4j.get_user_dietary_restrictions(user_id)

        # Get popular dishes filtered by restrictions
        if restrictions:
            popular = self.neo4j.get_popular_dishes_with_restriction_filter(
                restriction_names=restrictions,
                limit=k * 5,  # Get extra in case some are already rated
                min_ratings=1,
            )
        else:
            popular = self.neo4j.get_popular_dishes(
                limit=k * 5,
                min_ratings=1,
            )

        # Get dishes user already rated to exclude
        user_ratings = self.neo4j.get_user_ratings(user_id)
        rated_dish_ids = {r["dish_id"] for r in user_ratings}

        recommendations = []
        logger.info("Found %d popular dishes for fallback recommendation", len(popular))
        for dish in popular:
            if dish["dish_id"] in rated_dish_ids:
                continue

            recommendations.append(
                DishRecommendation(
                    dish_id=dish["dish_id"],
                    name=dish["name"],
                    predicted_score=float(dish["avg_rating"]),
                    description=dish.get("description"),
                    ingredients=dish.get("ingredients", []),
                    recommender_count=int(dish["rating_count"]),
                    reason="popular_dish",
                )
            )

            if len(recommendations) >= k:
                break

        return recommendations

    def _recommend_collaborative(
        self,
        user_id: str,
        k: int = 3,
        apply_dietary_filter: bool = True,
    ) -> list[DishRecommendation]:
        """Recommend dishes using user-based collaborative filtering (Cosine similarity).

        Args:
            user_id: Target user ID.
            k: Maximum number of recommendations.
            apply_dietary_filter: Whether to filter by dietary restrictions.

        Returns:
            List of DishRecommendation objects.
        """
        # Find similar users using Cosine similarity
        similar_users = self.find_similar_users(
            user_id=user_id,
            k=10,  # Consider top 10 similar users
            min_similarity=SIMILARITY_THRESHOLD,
        )

        if not similar_users:
            logging.info(
                "No similar users found for %s with threshold > %.2f, falling back to popular",
                user_id,
                SIMILARITY_THRESHOLD,
            )
            return self._recommend_popular_dishes(
                user_id=user_id,
                k=k,
                apply_dietary_filter=apply_dietary_filter,
            )

        similar_user_ids = [u.user_id for u in similar_users]
        similarity_map = {u.user_id: u.similarity for u in similar_users}

        # Get dishes rated by similar users that target hasn't rated
        candidate_dishes = self.neo4j.get_dishes_rated_by_users(
            user_ids=similar_user_ids,
            exclude_user_id=user_id,
            limit=k * 5,
        )
        logger.info("Found %d candidate dishes from similar users", len(candidate_dishes))
        if not candidate_dishes:
            logger.info("No candidate dishes from similar users, falling back to popular")
            return self._recommend_popular_dishes(
                user_id=user_id,
                k=k,
                apply_dietary_filter=apply_dietary_filter,
            )

        # Apply dietary restriction filter
        if apply_dietary_filter:
            restrictions = self.neo4j.get_user_dietary_restrictions(user_id)
            if restrictions:
                candidate_dish_ids = [d["dish_id"] for d in candidate_dishes]
                safe_dish_ids = set(
                    self.neo4j.filter_dishes_by_restrictions(
                        dish_ids=candidate_dish_ids,
                        restriction_names=restrictions,
                    )
                )
                candidate_dishes = [
                    d for d in candidate_dishes if d["dish_id"] in safe_dish_ids
                ]

        # Compute weighted predicted scores
        recommendations = []
        for dish in candidate_dishes:
            ratings = dish.get("ratings", [])

            # Compute weighted average using similarity scores
            weighted_sum = 0.0
            weight_total = 0.0
            rater_count = 0

            for rating in ratings:
                rater_id = rating["user_id"]
                score = rating["score"]

                if rater_id in similarity_map:
                    sim = similarity_map[rater_id]
                    weighted_sum += sim * score
                    weight_total += sim
                    rater_count += 1

            if weight_total > 0:
                predicted_score = weighted_sum / weight_total
            else:
                predicted_score = dish.get("avg_score", 3.0)

            recommendations.append(
                DishRecommendation(
                    dish_id=dish["dish_id"],
                    name=dish["name"],
                    predicted_score=round(predicted_score, 2),
                    description=dish.get("description"),
                    ingredients=dish.get("ingredients", []),
                    recommender_count=rater_count,
                    reason="collaborative_filtering",
                )
            )

        # Sort by predicted score descending
        recommendations.sort(key=lambda x: x.predicted_score, reverse=True)

        return recommendations[:k]


# Global service instance
_recommendation_service: RecommendationService | None = None


def get_recommendation_service() -> RecommendationService:
    """Get or create the global recommendation service instance.

    Returns:
        RecommendationService singleton instance.
    """
    global _recommendation_service
    if _recommendation_service is None:
        _recommendation_service = RecommendationService()
    return _recommendation_service
