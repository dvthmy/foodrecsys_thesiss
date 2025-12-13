"""Ingredient canonicalization service.

Normalizes ingredient names by matching them against canonical ingredients
using CLIP embeddings and semantic similarity. Uses Gemma LLM for ambiguous
cases where similarity scores fall between high and low thresholds.

Pipeline:
1. Generate CLIP embedding for new ingredient
2. Query Neo4j vector index for top-k similar canonical ingredients
3. Apply threshold logic:
   - score > high_threshold: Auto-map to existing canonical
   - low_threshold < score < high_threshold: LLM decides merge or new
   - score < low_threshold: Create as pending (requires admin approval)
"""

import logging
from typing import Any

from src.config import config
from src.services.clip_embedder import CLIPEmbedder, get_clip_embedder
from src.services.gemma_extractor import GemmaExtractor, get_gemma_extractor
from src.services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class IngredientCanonicalizer:
    """Service for canonicalizing ingredient names using semantic similarity."""

    # Prompt template for LLM decision on ingredient matching
    DECISION_PROMPT = """You are an expert chef and food scientist. Determine if a new ingredient should be merged with an existing canonical ingredient or kept as a new distinct ingredient.

New ingredient: "{new_ingredient}"

Top similar existing ingredients (with similarity scores):
{candidates}

**Strictly** follow decision criteria:
- MERGE if the new ingredient is essentially the same thing (just different wording, spelling, or minor variation). Examples: "fresh veggies" → "fresh vegetables", "beef meat" → "beef", "white onion" → "onion"
  - THEN: Respond with ONLY a JSON object:
    {{
        "decision": "merge",
        "merge_into": "canonical ingredient name" (only if decision is "merge", otherwise null),
        "reason": "brief explanation"
    }}
- KEEP SEPARATE if:
 - Different parts of the plant/animal (e.g., "almond" vs "almond milk", "bone" vs "beef").
 - Different preparations that change the ingredient significantly (e.g., "dried basil" vs "fresh basil", "fresh vegetables" vs "pickled vegetables")
 - THEN: {{
        "decision": "new",
        "merge_into": null,
        "reason": "criteria explanation" (e.g., "different preparation", "different part of plant/animal")
    }}
"""

    def __init__(
        self,
        neo4j: Neo4jService | None = None,
        clip: CLIPEmbedder | None = None,
        gemma: GemmaExtractor | None = None,
        high_threshold: float | None = None,
        low_threshold: float | None = None,
        top_k: int | None = None,
    ):
        """Initialize the canonicalizer.

        Args:
            neo4j: Neo4j service instance.
            clip: CLIP embedder instance.
            gemma: Gemma extractor instance for LLM decisions.
            high_threshold: Score above which to auto-merge (default from config).
            low_threshold: Score below which to create as new (default from config).
            top_k: Number of candidates to retrieve (default from config).
        """
        self._neo4j = neo4j
        self._clip = clip
        self._gemma = gemma
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
    def clip(self) -> CLIPEmbedder:
        """Get or create CLIP embedder."""
        if self._clip is None:
            self._clip = get_clip_embedder()
        return self._clip

    @property
    def gemma(self) -> GemmaExtractor:
        """Get or create Gemma extractor."""
        if self._gemma is None:
            self._gemma = get_gemma_extractor()
        return self._gemma

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
                - action: "exact_match", "auto_merge", "llm_merge", "new_pending"
                - score: Similarity score (if matched)
                - reason: Explanation of the decision
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

        # Step 2: Generate embedding
        embedding = self.clip.embed_text(name)

        # Step 3: Find similar canonical ingredients
        candidates = self.neo4j.find_similar_ingredients(
            embedding=embedding,
            k=self._top_k,
            threshold=0.9,  # Get all to see what's available
            canonical_only=True,
        )

        # Handle cold start (no canonical ingredients yet)
        if not candidates:
            logger.info("No canonical ingredients found, creating '%s' as pending", name)
            self.neo4j.create_pending_ingredient(name, embedding)
            return {
                "canonical_name": name,
                "is_canonical": False,
                "action": "new_pending",
                "score": 0.0,
                "reason": "No canonical ingredients to match against",
            }

        best_match = candidates[0]
        best_score = best_match["score"]
        best_name = best_match["name"]

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

        elif best_score >= self._low_threshold:
            # Ambiguous: use LLM to decide
            logger.info("Ambiguous match for '%s', consulting LLM...", name)
            decision = self._llm_decide(name, candidates)

            if decision.get("decision") == "merge":
                merge_into = decision.get("merge_into", best_name)
                logger.info("LLM decided to merge '%s' → '%s': %s",
                           name, merge_into, decision.get("reason"))
                return {
                    "canonical_name": merge_into,
                    "is_canonical": True,
                    "action": "llm_merge",
                    "score": best_score,
                    "reason": decision.get("reason", "LLM decided to merge"),
                }
            else:
                # LLM says keep separate, create as pending
                logger.info("LLM decided '%s' is new: %s", name, decision.get("reason"))
                self.neo4j.create_pending_ingredient(name, embedding)
                return {
                    "canonical_name": name,
                    "is_canonical": False,
                    "action": "new_pending",
                    "score": best_score,
                    "reason": decision.get("reason", "LLM decided ingredient is distinct"),
                }

        else:
            # Low score: create as pending
            logger.info("Low match score for '%s' (%.4f < %.4f), creating as pending",
                       name, best_score, self._low_threshold)
            self.neo4j.create_pending_ingredient(name, embedding)
            return {
                "canonical_name": name,
                "is_canonical": False,
                "action": "new_pending",
                "score": best_score,
                "reason": f"Low similarity score (score={best_score:.4f})",
            }

    def _llm_decide(
        self,
        new_ingredient: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Use Gemma LLM to decide if ingredient should be merged.

        Args:
            new_ingredient: The new ingredient name.
            candidates: List of candidate matches with 'name' and 'score'.

        Returns:
            Dictionary with 'decision', 'merge_into', and 'reason'.
        """
        # Format candidates for prompt
        candidates_text = "\n".join(
            f"- {c['name']} (similarity: {c['score']:.4f})"
            for c in candidates[:self._top_k]
        )

        prompt = self.DECISION_PROMPT.format(
            new_ingredient=new_ingredient,
            candidates=candidates_text,
        )

        try:
            # Use Gemma to make decision
            messages = [{"role": "user", "content": prompt}]

            # Ensure model is loaded
            self.gemma._load_model()

            # Access pipeline and tokenizer after ensuring model is loaded
            pipeline = self.gemma._pipeline
            tokenizer = self.gemma._tokenizer

            if pipeline is None or tokenizer is None:
                raise RuntimeError("Gemma model failed to load")

            outputs = pipeline(
                messages,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.3,  # Lower temperature for more deterministic decisions
                top_p=0.9,
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
            )

            # Extract response
            response_text = outputs[0]["generated_text"]
            if isinstance(response_text, list):
                for msg in response_text:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        response_text = msg.get("content", "")
                        break

            # Ensure response_text is a string
            if not isinstance(response_text, str):
                response_text = str(response_text)

            # Parse JSON response
            result = self.gemma._parse_json_response(response_text)

            return {
                "decision": result.get("decision", "new"),
                "merge_into": result.get("merge_into"),
                "reason": result.get("reason", "LLM decision"),
            }

        except Exception as e:
            logger.error("LLM decision failed: %s", e)
            # Default to new on failure
            return {
                "decision": "new",
                "merge_into": None,
                "reason": f"LLM decision failed: {e}",
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
