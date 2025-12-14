"""Ingredient canonicalization service.

Normalizes ingredient names by matching them against canonical ingredients
using EmbeddingGemma embeddings and semantic similarity.

Pipeline:
1. Generate EmbeddingGemma embedding for new ingredient
2. Query Neo4j vector index for top-k similar canonical ingredients
3. Apply threshold logic:
   - score > high_threshold: Auto-map to existing canonical
    - low_threshold <= score < high_threshold: Create as pending ("medium" confidence)
    - score < low_threshold: Create as pending ("low" confidence)

All pending ingredients store suggested canonical merge candidates (top-k) to
assist human review.
"""

import logging
from typing import Any

from src.config import config
from src.services.ingredient_embedder import IngredientEmbedder, get_ingredient_embedder
from src.services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class IngredientCanonicalizer:
    """Service for canonicalizing ingredient names using semantic similarity."""

    def __init__(
        self,
        neo4j: Neo4jService | None = None,
        embedder: IngredientEmbedder | None = None,
        high_threshold: float | None = None,
        low_threshold: float | None = None,
        top_k: int | None = None,
    ):
        """Initialize the canonicalizer.

        Args:
            neo4j: Neo4j service instance.
            embedder: Ingredient embedder instance (EmbeddingGemma).
            high_threshold: Score above which to auto-merge (default from config).
            low_threshold: Score below which to create as new (default from config).
            top_k: Number of candidates to retrieve (default from config).
        """
        self._neo4j = neo4j
        self._embedder = embedder
        self._high_threshold = high_threshold or config.SIMILARITY_THRESHOLD_HIGH
        self._low_threshold = low_threshold or config.SIMILARITY_THRESHOLD_LOW
        self._top_k = top_k or config.CANONICAL_TOP_K

    @property
    def neo4j(self) -> Neo4jService:
        """Get or create Neo4j service."""
        if self._neo4j is None:
            self._neo4j = Neo4jService()
        return self._neo4j

    @property
    def embedder(self) -> IngredientEmbedder:
        """Get or create Ingredient embedder."""
        if self._embedder is None:
            self._embedder = get_ingredient_embedder()
        return self._embedder

    def canonicalize(self, ingredient_name: str) -> dict[str, Any]:
        """Canonicalize a single ingredient name.

        Processes the ingredient through the canonicalization pipeline:
        1. Check if exact match exists
        2. Generate embedding and find similar canonical ingredients
        3. Apply threshold logic with optional LLM decision

        Args:
            ingredient_name: The raw ingredient name to canonicalize.

        Returns:
            Dictionary with:
                - canonical_name: The final ingredient name to use
                - is_canonical: Whether it's a canonical ingredient
                - action: "exact_match", "auto_merge", "new_pending"
                - score: Similarity score (if matched)
                - reason: Explanation of the decision
                - pending_confidence: "medium" or "low" (only for action "new_pending")
                - suggested_merges: Top-k candidate canonical ingredients (only for action "new_pending")
        """
        name = ingredient_name.lower().strip()
        logger.info("Canonicalizing ingredient: %s", name)

        # Step 1: Check for exact match
        existing = self.neo4j.get_ingredient(name)
        if existing:
            logger.info("Exact match found for '%s'", name)
            return {
                "canonical_name": name,
                "is_canonical": existing.get("is_canonical", False),
                "action": "exact_match",
                "score": 1.0,
                "reason": "Exact match in database",
            }

        # Step 2: Generate embedding using EmbeddingGemma (query mode for search)
        embedding = self.embedder.embed_ingredient(name, mode="query")

        # Step 3: Find similar canonical ingredients
        candidates = self.neo4j.find_similar_ingredients(
            embedding=embedding,
            k=self._top_k,
            threshold=0.0,
            canonical_only=True,
        )

        # Handle cold start (no canonical ingredients yet)
        if not candidates:
            logger.info("No canonical ingredients found, creating '%s' as pending", name)
            # Use document mode embedding for storage
            storage_embedding = self.embedder.embed_ingredient(name, mode="document")
            self.neo4j.create_pending_ingredient(
                name,
                storage_embedding,
                confidence="low",
                suggested_merges=[],
                best_score=None,
            )
            return {
                "canonical_name": name,
                "is_canonical": False,
                "action": "new_pending",
                "score": 0.0,
                "reason": "No canonical ingredients to match against",
                "pending_confidence": "low",
                "suggested_merges": [],
            }

        best_match = candidates[0]
        best_score = best_match["score"]
        best_name = best_match["name"]

        suggested_merges = candidates[: self._top_k]

        logger.info(
            "Best match for '%s': '%s' (score=%.4f)",
            name,
            best_name,
            best_score,
        )

        # Step 4: Apply threshold logic
        if best_score >= self._high_threshold:
            # Auto-merge: high confidence match
            logger.info("Auto-merging '%s' → '%s' (score=%.4f >= %.4f)",
                       name, best_name, best_score, self._high_threshold)
            return {
                "canonical_name": best_name,
                "is_canonical": True,
                "action": "auto_merge",
                "score": best_score,
                "reason": f"High confidence match (score={best_score:.4f})",
            }

        # Not high enough to merge: always create as pending with confidence
        pending_confidence = "medium" if best_score >= self._low_threshold else "low"
        logger.info(
            "Creating pending ingredient '%s' (confidence=%s, best_score=%.4f)",
            name,
            pending_confidence,
            best_score,
        )

        storage_embedding = self.embedder.embed_ingredient(name, mode="document")
        self.neo4j.create_pending_ingredient(
            name,
            storage_embedding,
            confidence=pending_confidence,
            suggested_merges=suggested_merges,
            best_score=best_score,
        )

        return {
            "canonical_name": name,
            "is_canonical": False,
            "action": "new_pending",
            "score": best_score,
            "reason": (
                f"Medium similarity score (score={best_score:.4f})"
                if pending_confidence == "medium"
                else f"Low similarity score (score={best_score:.4f})"
            ),
            "pending_confidence": pending_confidence,
            "suggested_merges": suggested_merges,
        }

    def canonicalize_batch(
        self,
        ingredient_names: list[str],
    ) -> list[dict[str, Any]]:
        """Canonicalize a batch of ingredients.

        Processes each ingredient individually for accuracy.

        Args:
            ingredient_names: List of raw ingredient names.

        Returns:
            List of canonicalization results.
        """
        results = []
        for name in ingredient_names:
            result = self.canonicalize(name)
            results.append(result)
        return results


# Singleton instance for reuse
_canonicalizer: IngredientCanonicalizer | None = None


def get_canonicalizer() -> IngredientCanonicalizer:
    """Get or create a singleton IngredientCanonicalizer instance.

    Returns:
        IngredientCanonicalizer instance.
    """
    global _canonicalizer
    if _canonicalizer is None:
        _canonicalizer = IngredientCanonicalizer()
    return _canonicalizer
