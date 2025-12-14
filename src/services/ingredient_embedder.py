"""Ingredient embedding service using Google's EmbeddingGemma model.

Uses the EmbeddingGemma model via SentenceTransformers to generate
semantic embeddings for ingredients, capturing their origins, groups,
and culinary relationships.

Design Principle:
- EmbeddingGemma: Specialized for semantic text understanding
- Better captures ingredient relationships (e.g., "basil" closer to "oregano")
- Supports both query and document embedding modes
"""

import logging
import threading
from typing import Literal

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config import config

try:
    from huggingface_hub import login
    login(token=config.HF_API_KEY)
except Exception:
    pass

logger = logging.getLogger(__name__)


class IngredientEmbedder:
    """Service for generating ingredient embeddings using EmbeddingGemma.
    
    This embedder is specifically designed for ingredient similarity search,
    capturing semantic relationships between ingredients including:
    - Origin (e.g., Italian herbs, Asian spices)
    - Category/Group (e.g., vegetables, proteins, dairy)
    - Culinary usage patterns
    """

    # Default model - EmbeddingGemma 300M for good balance of speed and quality
    DEFAULT_MODEL = "google/embeddinggemma-300m"
    
    # Output dimension - truncate from native 768 to 512 for consistency with Neo4j index
    OUTPUT_DIM = 512
    # Asymmetric retrieval prefixes (official EmbeddingGemma format)
    # Query prefix: for search queries (short, vague, or functional)
    QUERY_PREFIX = "task: search result | query: {text}"
    # Document prefix: for indexed items (with title and description)
    DOCUMENT_PREFIX = "title: {title} | text: {text}"
    DOC_TEMPLATE = "title: {title} | text: {text}"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        use_context: bool = True,
        output_dim: int | None = None,
    ):
        """Initialize the ingredient embedder.

        Args:
            model_name: Hugging Face model name. Defaults to EmbeddingGemma 300M.
            device: Device to run on ('cpu', 'cuda', 'mps'). Auto-detected if None.
            use_context: Whether to wrap ingredient names with context for better
                        semantic understanding.
            output_dim: Output embedding dimension. Defaults to 512 (truncated from native 768).
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._device = device or self._detect_device()
        self._use_context = use_context
        self._output_dim = output_dim or self.OUTPUT_DIM
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        
        logger.info(
            "Initializing IngredientEmbedder with model=%s, device=%s",
            self._model_name,
            self._device,
        )

    def _detect_device(self) -> str:
        """Detect the best available device.

        Returns:
            Device string ('cuda', 'mps', or 'cpu').
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        """Load the EmbeddingGemma model (lazy loading)."""
        if self._model is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._model is None:
                    logger.info("Loading EmbeddingGemma model: %s", self._model_name)
                    self._model = SentenceTransformer(
                        self._model_name,
                        device=self._device,
                    )
                    logger.info(
                        "EmbeddingGemma model loaded on device: %s", 
                        self._device
                    )

    @property
    def model(self) -> SentenceTransformer:
        """Get the EmbeddingGemma model (loads if not already loaded)."""
        self._load_model()
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the dimension of the output embedding vectors.

        Returns:
            Embedding dimension (512 by default, truncated from native 768).
        """
        return self._output_dim

    def _prepare_query_text(self, query: str) -> str:
        """Prepare query text for asymmetric retrieval.

        Args:
            query: The search query (ingredient name or description).

        Returns:
            Formatted query text with retrieval prefix.
        """
        return self.QUERY_PREFIX.format(text=query.lower().strip())

    def _prepare_document_text(
        self, 
        ingredient: str, 
        description: str | None = None,
        dish_name: str | None = None,
    ) -> str:
        """Prepare document text for asymmetric retrieval.

        Args:
            ingredient: The ingredient name (used as title).
            description: Optional description of the ingredient.
            dish_name: Optional dish name to include in description.

        Returns:
            Formatted document text with title and description.
        """
        title = ingredient.lower().strip()
        
        # Build description from available context
        if description:
            text = description.lower().strip()
        elif dish_name:
            text = f"food ingredient used in {dish_name.lower().strip()}"
        else:
            text = "food ingredient"
        
        return self.DOCUMENT_PREFIX.format(title=title, text=text)

    def embed_ingredient(
            self,
            ingredient: str,
            mode: Literal["query", "document"] = "document",
            dish_name: str | None = None,
            description: str | None = "food ingredient",
        ) -> list[float]:
            """Generate embedding with correct prompting and truncation."""
            
            # Ensure model is loaded
            _ = self.model 

            if mode == "query":
                # CASE A: QUERY
                # We use encode_query(). It automatically handles the "task: search..." prefix.
                # We pass the RAW text.
                query_text = ingredient.lower().strip()
                embedding = self.model.encode_query(
                    query_text, 
                    truncate_dim=self._output_dim
                )

            else:
                # CASE B: DOCUMENT
                # We use generic encode(). We manually format to inject the Dynamic Title.
                # If we used encode_document(), it might force "title: none".
                
                # 1. Construct Content
                if description:
                    content = description.lower().strip()
                else:
                    content = "food ingredient"
                print(content)
                # 2. Format with Official Template
                # Result: "title: basil | text: a sweet herb..."
                full_text = self.DOC_TEMPLATE.format(
                    title=ingredient.lower().strip(),
                    text=content
                )

                # 3. Encode as "raw" text (prompt_name=None prevents double prefixing)
                embedding = self.model.encode(
                    full_text,
                    prompt_name=None, 
                    truncate_dim=self._output_dim
                )
            embedding = np.asarray(embedding)
            
            return embedding.tolist()
    
    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score (-1 to 1, typically 0 to 1 for normalized).
        """
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

    def find_similar(
        self,
        query_ingredient: str,
        candidate_embeddings: list[list[float]],
        candidate_names: list[str],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Find similar ingredients from a list of candidates.

        Args:
            query_ingredient: The ingredient to search for.
            candidate_embeddings: Pre-computed embeddings of candidates.
            candidate_names: Names corresponding to the embeddings.
            top_k: Number of top results to return.
            threshold: Minimum similarity score to include.

        Returns:
            List of dicts with 'name' and 'score' keys, sorted by score descending.
        """
        if not candidate_embeddings:
            return []

        # Use query mode for the search ingredient
        query_embedding = np.array(self.embed_ingredient(query_ingredient, mode="query"))
        candidates = np.array(candidate_embeddings)
        
        # Compute similarities
        similarities = np.dot(candidates, query_embedding)
        
        # Sort by similarity
        indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in indices[:top_k]:
            score = float(similarities[idx])
            if score >= threshold:
                results.append({
                    "name": candidate_names[idx],
                    "score": score,
                })
        
        return results

    def embed_with_category_context(
        self,
        ingredient: str,
        category: str | None = None,
        origin: str | None = None,
    ) -> list[float]:
        """Generate embedding with additional category/origin context.

        This method provides richer embeddings by including category and
        origin information when available. Uses the asymmetric document format.

        Args:
            ingredient: The ingredient name.
            category: Optional category (e.g., "vegetable", "spice", "protein").
            origin: Optional origin (e.g., "Italian", "Asian", "Mexican").

        Returns:
            List of floats representing the contextual embedding.
        """
        # Build rich description from category/origin context
        context_parts = []
        if category:
            context_parts.append(f"{category}")
        if origin:
            context_parts.append(f"{origin} cuisine")
        context_parts.append("food ingredient")
        
        description = " ".join(context_parts)
        text = self._prepare_document_text(ingredient, description)
        
        embedding = np.asarray(self.model.encode_document(text, truncate_dim=512))
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()


# Global embedder instance (lazy loaded)
_embedder: IngredientEmbedder | None = None
_embedder_lock = threading.Lock()


def get_ingredient_embedder(
    model_name: str | None = None,
    device: str | None = None,
) -> IngredientEmbedder:
    """Get or create the global IngredientEmbedder instance.

    Args:
        model_name: Model name (only used on first call).
        device: Device (only used on first call).

    Returns:
        IngredientEmbedder singleton instance.
    """
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = IngredientEmbedder(model_name=model_name, device=device)
        return _embedder
