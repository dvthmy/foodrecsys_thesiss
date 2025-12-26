"""Offline evaluation for recommendation system.

This module provides comprehensive evaluation metrics for the food recommendation system,
including nDCG@k, Hit Rate, MRR, and comparison between different algorithms.

Evaluation Methods:
1. Train-Test Split: Split ratings into training and test sets
2. Leave-One-Out: Hold out one rating per user for testing

IMPORTANT: This evaluator builds its own rating matrices from training data only,
ensuring proper train/test separation. It does NOT use the recommendation service
directly (which would use all database ratings), but instead replicates the
recommendation logic using only training ratings.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from src.services.neo4j_service import Neo4jService
from src.services.ingredient_embedder import get_ingredient_embedder
from src.visualization.dish_aggregator import get_aggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Holds evaluation metrics for a recommendation method."""
    
    method_name: str
    metric_type: str  # 'cosine', 'jaccard', 'embedding'
    ndcg_at_k: float = 0.0
    hit_rate: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    coverage: float = 0.0
    avg_popularity: float = 0.0
    num_users_evaluated: int = 0
    k: int = 10
    
    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"Method: {self.method_name} ({self.metric_type})\n"
            f"{'='*60}\n"
            f"  nDCG@{self.k}:      {self.ndcg_at_k:.4f}\n"
            f"  Hit Rate:          {self.hit_rate:.4f}\n"
            f"  MRR:               {self.mrr:.4f}\n"
            f"  Coverage:          {self.coverage:.4f}\n"
            f"  Avg Popularity:    {self.avg_popularity:.4f}\n"
            f"  Users Evaluated:   {self.num_users_evaluated}\n"
        )


@dataclass
class UserEvaluation:
    """Holds evaluation results for a single user."""
    
    user_id: str
    ndcg: float = 0.0
    hit: bool = False
    reciprocal_rank: float = 0.0
    recommended_dishes: list[str] = field(default_factory=list)
    relevant_dishes: list[str] = field(default_factory=list)


class TrainOnlyRecommender:
    """A recommender that uses ONLY training data for making predictions.
    
    This class replicates the recommendation logic but uses a custom rating
    matrix built from training data only, ensuring proper evaluation.
    """
    
    def __init__(self, train_ratings: list[dict[str, Any]], neo4j_service: Neo4jService):
        """Initialize with training ratings only.
        
        Args:
            train_ratings: List of rating dictionaries from training set.
            neo4j_service: Neo4j service for fetching dish metadata.
        """
        self.neo4j = neo4j_service
        self._train_ratings = train_ratings
        
        # Build indices
        self._user_ids: list[str] = []
        self._dish_ids: list[str] = []
        self._user_index: dict[str, int] = {}
        self._dish_index: dict[str, int] = {}
        self._rating_matrix: NDArray[np.float64] | None = None
        self._user_similarity_matrix: NDArray[np.float64] | None = None
        
        # User's rated dishes (from training only)
        self._user_rated_dishes: dict[str, set[str]] = defaultdict(set)
        
        # User's liked dishes (rating >= 4) for content-based
        self._user_liked_dishes: dict[str, set[str]] = defaultdict(set)
        
        # Dish aggregator for ingredient-based recommendations
        self._aggregator = None
        self._aggregator_fitted = False
        
        self._build_rating_matrix()
    
    def _normalize_id(self, value: Any) -> str:
        """Normalize ID value to string."""
        if isinstance(value, list):
            return str(value[0]) if value else ""
        return str(value)
    
    def _build_rating_matrix(self) -> None:
        """Build rating matrix from training data only."""
        if not self._train_ratings:
            logger.warning("No training ratings provided")
            self._rating_matrix = np.array([])
            return
        
        # Build indices
        self._user_ids = list(set(self._normalize_id(r["user_id"]) for r in self._train_ratings))
        self._dish_ids = list(set(self._normalize_id(r["dish_id"]) for r in self._train_ratings))
        self._user_index = {uid: i for i, uid in enumerate(self._user_ids)}
        self._dish_index = {did: i for i, did in enumerate(self._dish_ids)}
        
        # Initialize matrix with zeros (unrated)
        n_users = len(self._user_ids)
        n_dishes = len(self._dish_ids)
        self._rating_matrix = np.zeros((n_users, n_dishes), dtype=np.float64)
        
        # Fill in ratings and track user's rated dishes
        for r in self._train_ratings:
            user_id = self._normalize_id(r["user_id"])
            dish_id = self._normalize_id(r["dish_id"])
            user_idx = self._user_index[user_id]
            dish_idx = self._dish_index[dish_id]
            self._rating_matrix[user_idx, dish_idx] = r["score"]
            self._user_rated_dishes[user_id].add(dish_id)
            
            # Track liked dishes for content-based
            if r["score"] >= 4.0:
                self._user_liked_dishes[user_id].add(dish_id)
        
        logger.info(
            f"Built training rating matrix: {n_users} users x {n_dishes} dishes, "
            f"{len(self._train_ratings)} ratings"
        )
    
    def _cosine_similarity(
        self,
        vec1: NDArray[np.float64],
        vec2: NDArray[np.float64],
    ) -> float:
        """Compute cosine similarity between two rating vectors."""
        common_mask = (vec1 > 0) & (vec2 > 0)
        
        if not np.any(common_mask):
            return 0.0
        
        v1 = vec1[common_mask]
        v2 = vec2[common_mask]
        
        v1_centered = v1 - np.mean(v1)
        v2_centered = v2 - np.mean(v2)
        
        norm1 = np.linalg.norm(v1_centered)
        norm2 = np.linalg.norm(v2_centered)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1_centered, v2_centered) / (norm1 * norm2))
    
    def _jaccard_similarity(
        self,
        vec1: NDArray[np.float64],
        vec2: NDArray[np.float64],
    ) -> float:
        """Compute Jaccard similarity between two rating vectors."""
        set1 = set(np.where(vec1 > 0)[0])
        set2 = set(np.where(vec2 > 0)[0])
        
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    def _compute_user_similarity(self, metric: str = "cosine") -> None:
        """Compute user-user similarity matrix."""
        if self._rating_matrix is None or self._rating_matrix.size == 0:
            self._user_similarity_matrix = np.array([])
            return
        
        n_users = len(self._user_ids)
        self._user_similarity_matrix = np.zeros((n_users, n_users), dtype=np.float64)
        
        for i in range(n_users):
            for j in range(i, n_users):
                if i == j:
                    self._user_similarity_matrix[i, j] = 1.0
                else:
                    if metric == "jaccard":
                        sim = self._jaccard_similarity(
                            self._rating_matrix[i],
                            self._rating_matrix[j],
                        )
                    else:
                        sim = self._cosine_similarity(
                            self._rating_matrix[i],
                            self._rating_matrix[j],
                        )
                    self._user_similarity_matrix[i, j] = sim
                    self._user_similarity_matrix[j, i] = sim
    
    def recommend_collaborative(
        self,
        user_id: str,
        k: int = 10,
        metric: str = "cosine",
        min_similarity: float = 0.1,
    ) -> list[tuple[str, float]]:
        """Generate recommendations using collaborative filtering.
        
        Args:
            user_id: Target user ID.
            k: Number of recommendations.
            metric: Similarity metric ('cosine' or 'jaccard').
            min_similarity: Minimum similarity threshold.
            
        Returns:
            List of (dish_id, predicted_score) tuples.
        """
        if self._rating_matrix is None or self._rating_matrix.size == 0:
            return []
        
        if user_id not in self._user_index:
            # User not in training data - return popular dishes
            return self._get_popular_dishes(k, user_id)
        
        # Compute similarity if not done
        if self._user_similarity_matrix is None:
            self._compute_user_similarity(metric)
        
        if self._user_similarity_matrix is None or self._user_similarity_matrix.size == 0:
            return []
        
        user_idx = self._user_index[user_id]
        similarities = self._user_similarity_matrix[user_idx]
        
        # Find similar users
        similar_users = []
        for other_idx, sim in enumerate(similarities):
            if other_idx != user_idx and sim > min_similarity:
                similar_users.append((other_idx, sim))
        
        if not similar_users:
            return self._get_popular_dishes(k, user_id)
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)
        top_similar = similar_users[:20]  # Use top 20 similar users
        
        # Predict scores for unrated dishes
        # NOTE: We include dishes in test set (not in training) for evaluation
        dish_scores: dict[str, float] = {}
        
        for dish_idx, dish_id in enumerate(self._dish_ids):
            # Skip if user already rated this dish IN TRAINING SET
            if dish_id in self._user_rated_dishes[user_id]:
                continue
            
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for other_idx, sim in top_similar:
                other_rating = self._rating_matrix[other_idx, dish_idx]
                if other_rating > 0:
                    weighted_sum += sim * other_rating
                    weight_sum += abs(sim)
            
            if weight_sum > 0:
                dish_scores[dish_id] = weighted_sum / weight_sum
        
        # Sort by predicted score
        sorted_dishes = sorted(dish_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_dishes[:k]
    
    def _get_popular_dishes(
        self,
        k: int,
        exclude_user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Get popular dishes as fallback."""
        if self._rating_matrix is None or self._rating_matrix.size == 0:
            return []
        
        # Count ratings and average score for each dish
        dish_stats: dict[str, tuple[int, float]] = {}
        
        for dish_idx, dish_id in enumerate(self._dish_ids):
            # Skip dishes rated by user in training
            if exclude_user_id and dish_id in self._user_rated_dishes.get(exclude_user_id, set()):
                continue
            
            ratings = self._rating_matrix[:, dish_idx]
            rated_mask = ratings > 0
            count = int(np.sum(rated_mask))
            
            if count > 0:
                avg_score = float(np.mean(ratings[rated_mask]))
                dish_stats[dish_id] = (count, avg_score)
        
        # Sort by count * avg_score (popularity weighted by quality)
        sorted_dishes = sorted(
            dish_stats.items(),
            key=lambda x: x[1][0] * x[1][1],
            reverse=True,
        )
        
        return [(dish_id, stats[1]) for dish_id, stats in sorted_dishes[:k]]
    
    def recommend_content_based(
        self,
        user_id: str,
        k: int = 10,
        metric: str = "jaccard",
    ) -> list[tuple[str, float]]:
        """Generate recommendations using content-based filtering.
        
        Args:
            user_id: Target user ID.
            k: Number of recommendations.
            metric: 
                - 'jaccard' for ingredient set similarity
                - 'ingredient_embedding' for semantic similarity using Gemma ingredient embeddings + TF-IDF aggregation
            
        Returns:
            List of (dish_id, predicted_score) tuples.
        """
        if metric == "ingredient_embedding":
            return self._recommend_by_ingredient_embedding(user_id, k)
        else:
            return self._recommend_by_ingredients(user_id, k)
    
    def _recommend_by_ingredients(
        self,
        user_id: str,
        k: int,
    ) -> list[tuple[str, float]]:
        """Recommend based on ingredient similarity (Jaccard).
        
        NOTE: We compute Jaccard similarity in Python instead of using the Neo4j query
        because the Neo4j query filters out ALL rated dishes (including test set),
        which would make proper evaluation impossible.
        """
        # Get dishes liked by user from TRAINING data only
        liked_dishes = self._user_liked_dishes.get(user_id, set())
        
        if not liked_dishes:
            return self._get_popular_dishes(k, user_id)
        
        # Get ingredients for liked dishes (from training)
        liked_ingredients: set[str] = set()
        for dish_id in liked_dishes:
            dish_info = self.neo4j.get_dish_by_id(dish_id)
            if dish_info and dish_info.get("ingredients"):
                liked_ingredients.update(dish_info.get("ingredients", []))
        
        if not liked_ingredients:
            return self._get_popular_dishes(k, user_id)
        
        # Get all dishes and compute Jaccard similarity
        # Only exclude dishes rated in TRAINING (not test set)
        user_rated_in_train = self._user_rated_dishes.get(user_id, set())
        dish_scores: list[tuple[str, float]] = []
        
        # Fetch all dishes with ingredients
        all_dishes = self.neo4j.get_all_dishes_ingredients()
        
        for dish in all_dishes:
            dish_id = self._normalize_id(dish.get("dish_id", ""))
            
            # Skip dishes rated in TRAINING only (allow test dishes to be recommended)
            if dish_id in user_rated_in_train:
                continue
            
            # Get dish ingredients
            dish_ingredients = set(dish.get("ingredients", []))
            
            if not dish_ingredients:
                continue
            
            # Jaccard similarity
            intersection = len(liked_ingredients.intersection(dish_ingredients))
            union = len(liked_ingredients.union(dish_ingredients))
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0:  # Only include non-zero scores
                    dish_scores.append((dish_id, similarity))
        
        if not dish_scores:
            return self._get_popular_dishes(k, user_id)
        
        # Sort by similarity
        dish_scores.sort(key=lambda x: x[1], reverse=True)
        
        return dish_scores[:k]
    
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
            logger.info("Dish aggregator fitted with IDF scores")
    
    def _recommend_by_ingredient_embedding(
        self,
        user_id: str,
        k: int,
    ) -> list[tuple[str, float]]:
        """Recommend based on ingredient embedding similarity using dish_aggregator.
        
        Uses Gemma ingredient embeddings with TF-IDF aggregation to compute
        semantic similarity between dishes.
        """
        # Ensure aggregator is fitted
        self._ensure_aggregator_fitted()
        
        # Get dishes liked by user from training data
        liked_dishes = self._user_liked_dishes.get(user_id, set())
        
        if not liked_dishes:
            logger.debug(f"User {user_id}: No liked dishes, using popular fallback")
            return self._get_popular_dishes(k, user_id)
        
        # Get liked dishes with ingredient embeddings efficiently using batch query
        liked_dishes_list = list(liked_dishes)
        
        if not liked_dishes_list:
            logger.debug(f"User {user_id}: No liked dishes, using popular fallback")
            return self._get_popular_dishes(k, user_id)
        
        # Batch query to get all dishes with ingredient embeddings at once
        query = """
        MATCH (d:Dish)
        WHERE d.dish_id IN $dish_ids
        OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
        WHERE i.embedding IS NOT NULL
        WITH d, 
             collect(DISTINCT i.name) AS ingredient_names,
             collect(DISTINCT i.embedding) AS ingredient_embeddings
        WHERE size(ingredient_embeddings) > 0
        RETURN d.dish_id AS dish_id,
               ingredient_names AS ingredients,
               [emb IN ingredient_embeddings WHERE emb IS NOT NULL] AS ingredient_embeddings
        """
        
        with self.neo4j.session() as session:
            result = session.run(query, dish_ids=liked_dishes_list)
            liked_dishes_data = [dict(record) for record in result]
        
        if not liked_dishes_data:
            logger.debug(f"User {user_id}: No liked dishes with ingredient embeddings, using popular fallback")
            return self._get_popular_dishes(k, user_id)
        
        # Aggregate ingredient embeddings for each liked dish to create dish embeddings
        dish_embeddings = []
        for dish in liked_dishes_data:
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
            logger.debug(f"User {user_id}: No valid dish embeddings computed, using popular fallback")
            return self._get_popular_dishes(k, user_id)
        
        # Compute user profile vector (mean of aggregated dish embeddings)
        user_profile = np.array(np.mean(dish_embeddings, axis=0))
        
        # Normalize user profile
        user_norm = np.linalg.norm(user_profile)
        if user_norm > 0:
            user_profile = user_profile / user_norm
        
        # Find similar dishes by comparing ingredient embeddings
        user_rated = self._user_rated_dishes.get(user_id, set())
        dish_scores: list[tuple[str, float]] = []
        
        # Get all dishes with ingredient embeddings efficiently
        # With only 66 dishes in DB, we don't need a limit
        all_dishes = self.neo4j.get_all_dishes_with_ingredient_embeddings(limit=None)
        
        logger.debug(f"User {user_id}: Found {len(liked_dishes_data)} liked dishes with embeddings, "
                    f"computing similarity with {len(all_dishes)} candidate dishes "
                    f"(user has rated {len(user_rated)} dishes)")
        
        for dish in all_dishes:
            dish_id = self._normalize_id(dish.get("dish_id", ""))
            
            # Skip dishes rated in training
            if dish_id in user_rated:
                continue
            
            ingredient_embeddings = dish.get("ingredient_embeddings", [])
            ingredients = dish.get("ingredients", [])
            
            if not ingredient_embeddings or not ingredients:
                continue
            
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
                # Normalize dish embedding
                dish_norm = np.linalg.norm(dish_emb)
                if dish_norm > 0:
                    dish_emb = dish_emb / dish_norm
                
                # Compute cosine similarity (both vectors are normalized)
                similarity = float(np.dot(user_profile, dish_emb))
                dish_scores.append((dish_id, similarity))
        
        # Sort by similarity
        dish_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not dish_scores:
            logger.debug(f"User {user_id}: No dishes found with ingredient embeddings, using popular fallback")
            return self._get_popular_dishes(k, user_id)
        
        # Log similarity distribution for debugging
        if dish_scores:
            top_similarities = [s[1] for s in dish_scores[:10]]
            logger.debug(f"User {user_id}: Top-10 similarities range: {min(top_similarities):.4f} to {max(top_similarities):.4f}")
        
        return dish_scores[:k]
    
    def recommend_random(
        self,
        user_id: str,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Generate random recommendations as baseline.
        
        Args:
            user_id: Target user ID.
            k: Number of recommendations.
            
        Returns:
            List of (dish_id, random_score) tuples.
        """
        if self._rating_matrix is None or self._rating_matrix.size == 0:
            return []
        
        # Get all dishes from database
        all_dishes = self.neo4j.get_dishes(limit=1000)
        
        # Filter out dishes rated by user in training
        user_rated = self._user_rated_dishes.get(user_id, set())
        candidate_dishes = [
            self._normalize_id(dish.get("dish_id", ""))
            for dish in all_dishes
            if self._normalize_id(dish.get("dish_id", "")) not in user_rated
        ]
        
        if not candidate_dishes:
            return []
        
        # Randomly sample k dishes
        random.seed(hash(user_id) % (2**32))  # Deterministic randomness per user
        sampled = random.sample(candidate_dishes, min(k, len(candidate_dishes)))
        
        # Return with random scores (for consistency with other methods)
        return [(dish_id, random.random()) for dish_id in sampled]
    
    def recommend_popular(
        self,
        user_id: str,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Generate popular dish recommendations as baseline.
        
        Args:
            user_id: Target user ID.
            k: Number of recommendations.
            
        Returns:
            List of (dish_id, score) tuples.
        """
        return self._get_popular_dishes(k, user_id)


class RecommendationEvaluator:
    """Evaluator for recommendation system using offline evaluation."""
    
    def __init__(
        self,
        neo4j_service: Neo4jService | None = None,
        min_rating_for_relevant: float = 4.0,
        random_seed: int = 42,
    ):
        """Initialize the evaluator.
        
        Args:
            neo4j_service: Neo4j service instance.
            min_rating_for_relevant: Minimum rating to consider a dish as relevant.
            random_seed: Random seed for reproducibility.
        """
        self._neo4j = neo4j_service
        self.min_rating_for_relevant = min_rating_for_relevant
        self.random_seed = random_seed
        
        # Data containers
        self._all_ratings: list[dict[str, Any]] = []
        self._train_ratings: list[dict[str, Any]] = []
        self._test_ratings: list[dict[str, Any]] = []
        self._user_test_dishes: dict[str, set[str]] = {}  # user_id -> set of relevant dish_ids in test
        self._all_dish_ids: set[str] = set()
        self._dish_popularity: dict[str, int] = {}  # dish_id -> rating count
        
        # Recommender built from training data only
        self._train_recommender: TrainOnlyRecommender | None = None
        
    @property
    def neo4j(self) -> Neo4jService:
        """Get or create Neo4j service instance."""
        if self._neo4j is None:
            self._neo4j = Neo4jService()
        return self._neo4j
    
    def load_data(self) -> None:
        """Load all ratings from database."""
        logger.info("Loading ratings from database...")
        self._all_ratings = self.neo4j.get_all_ratings()
        
        # Build dish set and popularity
        self._all_dish_ids = set()
        self._dish_popularity = defaultdict(int)
        
        for r in self._all_ratings:
            dish_id = self._normalize_id(r["dish_id"])
            self._all_dish_ids.add(dish_id)
            self._dish_popularity[dish_id] += 1
            
        logger.info(f"Loaded {len(self._all_ratings)} ratings for {len(self._all_dish_ids)} dishes")
    
    def _normalize_id(self, value: Any) -> str:
        """Normalize ID value to string."""
        if isinstance(value, list):
            return str(value[0]) if value else ""
        return str(value)
    
    def leave_one_out_split(self) -> None:
        """Leave-one-out split: hold out the last rating per user for testing."""
        if not self._all_ratings:
            self.load_data()
            
        # Group ratings by user
        user_ratings: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in self._all_ratings:
            user_id = self._normalize_id(r["user_id"])
            user_ratings[user_id].append(r)
        
        self._train_ratings = []
        self._test_ratings = []
        self._user_test_dishes = defaultdict(set)
        
        random.seed(self.random_seed)
        
        for user_id, ratings in user_ratings.items():
            if len(ratings) < 2:
                self._train_ratings.extend(ratings)
                continue
            
            # Randomly select one rating for test
            test_idx = random.randint(0, len(ratings) - 1)
            
            for i, r in enumerate(ratings):
                if i == test_idx:
                    self._test_ratings.append(r)
                    if r["score"] >= self.min_rating_for_relevant:
                        dish_id = self._normalize_id(r["dish_id"])
                        self._user_test_dishes[user_id].add(dish_id)
                else:
                    self._train_ratings.append(r)
        
        # Build recommender from training data only
        self._train_recommender = TrainOnlyRecommender(self._train_ratings, self.neo4j)
        
        logger.info(
            f"Leave-one-out split: {len(self._train_ratings)} train, "
            f"{len(self._test_ratings)} test"
        )
    
    def _dcg_at_k(self, relevances: list[float], k: int) -> float:
        """Compute Discounted Cumulative Gain at k.
        
        Args:
            relevances: List of relevance scores (1 if relevant, 0 if not).
            k: Number of items to consider.
            
        Returns:
            DCG@k score.
        """
        relevances = relevances[:k]
        if not relevances:
            return 0.0
        
        dcg = relevances[0]
        for i in range(1, len(relevances)):
            dcg += relevances[i] / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        return float(dcg)
    
    def _ndcg_at_k(self, relevances: list[float], k: int) -> float:
        """Compute Normalized DCG at k.
        
        Args:
            relevances: List of relevance scores.
            k: Number of items to consider.
            
        Returns:
            nDCG@k score between 0 and 1.
        """
        dcg = self._dcg_at_k(relevances, k)
        
        # Ideal DCG: all relevant items at the top
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self._dcg_at_k(ideal_relevances, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_user(
        self,
        user_id: str,
        recommendations: list[tuple[str, float]],
        relevant_dishes: set[str],
        k: int = 10,
    ) -> UserEvaluation:
        """Evaluate recommendations for a single user.
        
        Args:
            user_id: User identifier.
            recommendations: List of (dish_id, score) tuples.
            relevant_dishes: Set of dish IDs that are relevant (liked in test set).
            k: Number of recommendations to consider.
            
        Returns:
            UserEvaluation with nDCG, hit rate, and MRR scores.
        """
        rec_dish_ids = [r[0] for r in recommendations[:k]]
        
        # Hits: recommended dishes that are in the relevant set
        hits = [1 if dish_id in relevant_dishes else 0 for dish_id in rec_dish_ids]
        num_hits = sum(hits)
        
        # nDCG@k
        ndcg = self._ndcg_at_k(hits, k)
        
        # Hit: at least one recommendation is relevant
        hit = num_hits > 0
        
        # Reciprocal Rank (position of first hit)
        rr = 0.0
        for i, h in enumerate(hits):
            if h == 1:
                rr = 1.0 / (i + 1)
                break
        
        return UserEvaluation(
            user_id=user_id,
            ndcg=ndcg,
            hit=hit,
            reciprocal_rank=rr,
            recommended_dishes=rec_dish_ids,
            relevant_dishes=list(relevant_dishes),
        )
    
    def evaluate_collaborative_filtering(
        self,
        k: int = 10,
        metric: str = "cosine",
    ) -> EvaluationResult:
        """Evaluate collaborative filtering method.
        
        Args:
            k: Number of recommendations.
            metric: Similarity metric ('cosine' or 'jaccard').
            
        Returns:
            EvaluationResult with aggregated metrics.
        """
        logger.info(f"Evaluating Collaborative Filtering ({metric})...")
        
        if self._train_recommender is None:
            logger.error("No training data available. Run leave_one_out_split first.")
            return EvaluationResult(
                method_name="Collaborative Filtering",
                metric_type=metric,
                k=k,
            )
        
        # Compute similarity matrix for the requested metric
        self._train_recommender._compute_user_similarity(metric=metric)
        
        user_evals: list[UserEvaluation] = []
        all_recommended_dishes: set[str] = set()
        popularity_scores: list[float] = []
        
        for user_id, relevant_dishes in self._user_test_dishes.items():
            if not relevant_dishes:
                continue
            
            try:
                # Get recommendations from training-only recommender
                recommendations = self._train_recommender.recommend_collaborative(
                    user_id=user_id,
                    k=k,
                    metric=metric,
                )
                
                if not recommendations:
                    continue
                
                # Evaluate
                user_eval = self.evaluate_user(user_id, recommendations, relevant_dishes, k)
                user_evals.append(user_eval)
                
                # Track coverage and popularity
                for dish_id, score in recommendations[:k]:
                    all_recommended_dishes.add(dish_id)
                    popularity_scores.append(self._dish_popularity.get(dish_id, 0))
                    
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
        
        if not user_evals:
            logger.warning("No users were evaluated!")
            return EvaluationResult(
                method_name="Collaborative Filtering",
                metric_type=metric,
                k=k,
            )
        
        # Aggregate metrics
        avg_ndcg = np.mean([e.ndcg for e in user_evals])
        hit_rate = np.mean([1 if e.hit else 0 for e in user_evals])
        mrr = np.mean([e.reciprocal_rank for e in user_evals])
        coverage = len(all_recommended_dishes) / len(self._all_dish_ids) if self._all_dish_ids else 0
        avg_popularity = np.mean(popularity_scores) if popularity_scores else 0
        
        return EvaluationResult(
            method_name="Collaborative Filtering",
            metric_type=metric,
            ndcg_at_k=float(avg_ndcg),
            hit_rate=float(hit_rate),
            mrr=float(mrr),
            coverage=float(coverage),
            avg_popularity=float(avg_popularity),
            num_users_evaluated=len(user_evals),
            k=k,
        )
    
    def evaluate_content_based(
        self,
        k: int = 10,
        metric: str = "jaccard",
    ) -> EvaluationResult:
        """Evaluate content-based filtering method.
        
        Args:
            k: Number of recommendations.
            metric: Similarity metric:
                - 'jaccard' for ingredient set similarity
                - 'ingredient_embedding' for semantic similarity using Gemma + TF-IDF aggregation
            
        Returns:
            EvaluationResult with aggregated metrics.
        """
        logger.info(f"Evaluating Content-Based Filtering ({metric})...")
        
        if self._train_recommender is None:
            logger.error("No training data available. Run leave_one_out_split first.")
            return EvaluationResult(
                method_name="Content-Based Filtering",
                metric_type=metric,
                k=k,
            )
        
        user_evals: list[UserEvaluation] = []
        all_recommended_dishes: set[str] = set()
        popularity_scores: list[float] = []
        
        for user_id, relevant_dishes in self._user_test_dishes.items():
            if not relevant_dishes:
                continue
            
            try:
                # Get recommendations from training-only recommender
                recommendations = self._train_recommender.recommend_content_based(
                    user_id=user_id,
                    k=k,
                    metric=metric,
                )
                
                if not recommendations:
                    continue
                
                # Evaluate
                user_eval = self.evaluate_user(user_id, recommendations, relevant_dishes, k)
                user_evals.append(user_eval)
                
                # Track coverage and popularity
                for dish_id, score in recommendations[:k]:
                    all_recommended_dishes.add(dish_id)
                    popularity_scores.append(self._dish_popularity.get(dish_id, 0))
                    
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
        
        if not user_evals:
            logger.warning("No users were evaluated!")
            return EvaluationResult(
                method_name="Content-Based Filtering",
                metric_type=metric,
                k=k,
            )
        
        # Aggregate metrics
        avg_ndcg = np.mean([e.ndcg for e in user_evals])
        hit_rate = np.mean([1 if e.hit else 0 for e in user_evals])
        mrr = np.mean([e.reciprocal_rank for e in user_evals])
        coverage = len(all_recommended_dishes) / len(self._all_dish_ids) if self._all_dish_ids else 0
        avg_popularity = np.mean(popularity_scores) if popularity_scores else 0
        
        return EvaluationResult(
            method_name="Content-Based Filtering",
            metric_type=metric,
            ndcg_at_k=float(avg_ndcg),
            hit_rate=float(hit_rate),
            mrr=float(mrr),
            coverage=float(coverage),
            avg_popularity=float(avg_popularity),
            num_users_evaluated=len(user_evals),
            k=k,
        )
    
    def evaluate_popular_baseline(
        self,
        k: int = 10,
    ) -> EvaluationResult:
        """Evaluate popular dishes baseline method.
        
        Args:
            k: Number of recommendations.
            
        Returns:
            EvaluationResult with aggregated metrics.
        """
        logger.info("Evaluating Popular Baseline...")
        
        if self._train_recommender is None:
            logger.error("No training data available. Run leave_one_out_split first.")
            return EvaluationResult(
                method_name="Popular Baseline",
                metric_type="popular",
                k=k,
            )
        
        user_evals: list[UserEvaluation] = []
        all_recommended_dishes: set[str] = set()
        popularity_scores: list[float] = []
        
        for user_id, relevant_dishes in self._user_test_dishes.items():
            if not relevant_dishes:
                continue
            
            try:
                recommendations = self._train_recommender.recommend_popular(
                    user_id=user_id,
                    k=k,
                )
                
                if not recommendations:
                    continue
                
                user_eval = self.evaluate_user(user_id, recommendations, relevant_dishes, k)
                user_evals.append(user_eval)
                
                for dish_id, score in recommendations[:k]:
                    all_recommended_dishes.add(dish_id)
                    popularity_scores.append(self._dish_popularity.get(dish_id, 0))
                    
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
        
        if not user_evals:
            logger.warning("No users were evaluated!")
            return EvaluationResult(
                method_name="Popular Baseline",
                metric_type="popular",
                k=k,
            )
        
        avg_ndcg = np.mean([e.ndcg for e in user_evals])
        hit_rate = np.mean([1 if e.hit else 0 for e in user_evals])
        mrr = np.mean([e.reciprocal_rank for e in user_evals])
        coverage = len(all_recommended_dishes) / len(self._all_dish_ids) if self._all_dish_ids else 0
        avg_popularity = np.mean(popularity_scores) if popularity_scores else 0
        
        return EvaluationResult(
            method_name="Popular Baseline",
            metric_type="popular",
            ndcg_at_k=float(avg_ndcg),
            hit_rate=float(hit_rate),
            mrr=float(mrr),
            coverage=float(coverage),
            avg_popularity=float(avg_popularity),
            num_users_evaluated=len(user_evals),
            k=k,
        )
    
    def run_full_evaluation(
        self,
        k: int = 10,
    ) -> list[EvaluationResult]:
        """Run full evaluation comparing all methods.
        
        Uses leave-one-out split method.
        
        Args:
            k: Number of recommendations to evaluate.
            
        Returns:
            List of EvaluationResult for each method.
        """
        logger.info("=" * 60)
        logger.info("Starting Full Recommendation Evaluation")
        logger.info("=" * 60)
        
        # Load and split data
        self.load_data()
        self.leave_one_out_split()
        
        results: list[EvaluationResult] = []
        
        # Evaluate Baselines first (for comparison)
        results.append(self.evaluate_popular_baseline(k=k))
        
        # Evaluate Collaborative Filtering (Cosine only)
        results.append(self.evaluate_collaborative_filtering(k=k, metric="cosine"))
        
        # Evaluate Content-Based methods
        results.append(self.evaluate_content_based(k=k, metric="jaccard"))
        results.append(self.evaluate_content_based(k=k, metric="ingredient_embedding"))
        
        return results
    
    def print_comparison_table(self, results: list[EvaluationResult]) -> None:
        """Print a comparison table of all evaluation results.
        
        Args:
            results: List of evaluation results to compare.
        """
        print("\n" + "=" * 95)
        print("RECOMMENDATION SYSTEM EVALUATION SUMMARY")
        print("=" * 95)
        print(f"{'Method':<35} {'Metric':<12} {'nDCG@k':<10} {'Hit Rate':<10} {'MRR':<10} {'Coverage':<10}")
        print("-" * 95)
        
        for r in results:
            print(
                f"{r.method_name:<35} {r.metric_type:<12} "
                f"{r.ndcg_at_k:<10.4f} {r.hit_rate:<10.4f} "
                f"{r.mrr:<10.4f} {r.coverage:<10.4f}"
            )
        
        print("=" * 95)
        
        # Find best method for each metric
        if results:
            best_ndcg = max(results, key=lambda x: x.ndcg_at_k)
            best_hit_rate = max(results, key=lambda x: x.hit_rate)
            best_mrr = max(results, key=lambda x: x.mrr)
            
            print("\nBest Methods:")
            print(f"  Best nDCG@k:      {best_ndcg.method_name} ({best_ndcg.metric_type}) = {best_ndcg.ndcg_at_k:.4f}")
            print(f"  Best Hit Rate:    {best_hit_rate.method_name} ({best_hit_rate.metric_type}) = {best_hit_rate.hit_rate:.4f}")
            print(f"  Best MRR:         {best_mrr.method_name} ({best_mrr.metric_type}) = {best_mrr.mrr:.4f}")


def main():
    """Main function to run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate recommendation system")
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="Number of recommendations to evaluate (default: 10)",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=4.0,
        help="Minimum rating to consider as relevant (default: 4.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator(
        min_rating_for_relevant=args.min_rating,
        random_seed=args.seed,
    )
    
    # Run evaluation (always uses leave_one_out split)
    results = evaluator.run_full_evaluation(k=args.top_k)
    
    # Print individual results
    for result in results:
        print(result)
    
    # Print comparison table
    evaluator.print_comparison_table(results)
    
    # Close connection
    evaluator.neo4j.close()


if __name__ == "__main__":
    main()
