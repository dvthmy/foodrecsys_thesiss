"""CLIP service for generating image embeddings.

Uses OpenAI's CLIP model via Hugging Face Transformers to generate
embeddings for dish images, enabling visual similarity search.
"""

import threading
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.config import config
from huggingface_hub import login

login(token=config.HF_API_KEY)

class CLIPEmbedder:
    """Service for generating image embeddings using CLIP."""

    # Default model - ViT-B/32 is a good balance of speed and quality
    DEFAULT_MODEL = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        """Initialize the CLIP embedder.

        Args:
            model_name: Hugging Face model name. Defaults to ViT-B/32.
            device: Device to run on ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._device = device or self._detect_device()
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self._lock = threading.Lock()

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
        """Load the CLIP model and processor (lazy loading)."""
        if self._model is None:
            with self._lock:
                # Double-check after acquiring lock
                if self._model is None:
                    self._processor = CLIPProcessor.from_pretrained(self._model_name)
                    self._model = CLIPModel.from_pretrained(self._model_name)
                    self._model.to(self._device)
                    self._model.eval()

    @property
    def model(self) -> CLIPModel:
        """Get the CLIP model (loads if not already loaded)."""
        self._load_model()
        return self._model

    @property
    def processor(self) -> CLIPProcessor:
        """Get the CLIP processor (loads if not already loaded)."""
        self._load_model()
        return self._processor

    @property
    def embedding_dim(self) -> int:
        """Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension (512 for ViT-B/32).
        """
        return self.model.config.projection_dim

    def embed_image(self, image_path: str | Path) -> list[float]:
        """Generate embedding for a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of floats representing the image embedding.

        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be processed.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

                # Normalize the embedding
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to list of floats
            return image_features.squeeze().cpu().tolist()

        except Exception as e:
            raise ValueError(f"Failed to process image {image_path}: {e}")

    def embed_images(self, image_paths: list[str | Path]) -> list[list[float]]:
        """Generate embeddings for multiple images (batched).

        Args:
            image_paths: List of paths to image files.

        Returns:
            List of embeddings, one per image.
        """
        if not image_paths:
            return []

        # Load all images
        images = []
        valid_indices = []
        for idx, path in enumerate(image_paths):
            try:
                image = Image.open(path).convert("RGB")
                images.append(image)
                valid_indices.append(idx)
            except Exception:
                continue

        if not images:
            return []

        # Process batch
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text (for text-to-image similarity).

        Args:
            text: Text description to embed.

        Returns:
            List of floats representing the text embedding.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.squeeze().cpu().tolist()

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
            Cosine similarity score (-1 to 1).
        """
        t1 = torch.tensor(embedding1)
        t2 = torch.tensor(embedding2)
        return torch.nn.functional.cosine_similarity(t1, t2, dim=0).item()


# Global embedder instance (lazy loaded)
_embedder: CLIPEmbedder | None = None
_embedder_lock = threading.Lock()


def get_clip_embedder() -> CLIPEmbedder:
    """Get or create the global CLIP embedder instance.

    Returns:
        CLIPEmbedder singleton instance.
    """
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = CLIPEmbedder()
        return _embedder
