"""Services package for external integrations."""

from src.services.neo4j_service import Neo4jService
from src.services.gemini_extractor import GeminiExtractor
from src.services.clip_embedder import CLIPEmbedder, get_clip_embedder
from src.services.gemma_extractor import GemmaExtractor, get_gemma_extractor
from src.services.ingredient_embedder import IngredientEmbedder, get_ingredient_embedder
from src.services.ingredient_canonicalizer import IngredientCanonicalizer, get_canonicalizer

__all__ = [
    "Neo4jService",
    "GeminiExtractor",
    "GemmaExtractor",
    "get_gemma_extractor",
    "CLIPEmbedder",
    "get_clip_embedder",
    "IngredientEmbedder",
    "get_ingredient_embedder",
    "IngredientCanonicalizer",
    "get_canonicalizer",
]
