"""Gemma local model service for extracting ingredients from text descriptions.

Uses Google's Gemma 3 1B Instruct model via Hugging Face Transformers
to analyze dish descriptions and extract ingredient lists locally,
without requiring an external API key.

Design Principle:
- Gemma: Local text/description-based ingredient extraction
- No API key required - runs entirely on local hardware
"""

import json
import logging
import re
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configure logger for this module
logger = logging.getLogger(__name__)


class GemmaExtractor:
    """Service for extracting ingredients from text descriptions using local Gemma model.

    This class focuses solely on text-based ingredient extraction using
    the Gemma 3 1B Instruct model from Hugging Face Transformers.
    For image processing, use the CLIPEmbedder service.
    """

    MODEL_ID = "google/gemma-3-1b-it"

    # Prompt template for ingredient extraction from text descriptions
    TEXT_PROMPT = """Extract all ingredients mentioned in this food dish description.

Description: {description}

Return a JSON object with the following structure:
{{
    "dish_name": "Name of the dish if mentioned, otherwise 'Unknown Dish'",
    "ingredients": ["ingredient1", "ingredient2", ...],
    "cuisine": "Type of cuisine if identifiable, otherwise null",
    "confidence": "high" | "medium" | "low"
}}

Guidelines:
- Extract all explicitly mentioned ingredients
- Infer common ingredients if the dish type is mentioned (e.g., "pizza" implies dough, tomato sauce, cheese)
- Use lowercase for ingredient names
- Be specific with ingredient names

Respond with ONLY the JSON object, no additional text."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ):
        """Initialize the Gemma extractor.

        Args:
            model_id: Hugging Face model ID. Defaults to google/gemma-3-1b-it.
            device: Device to run the model on ('cuda', 'cpu', 'mps', or 'auto').
                   Defaults to 'auto' which selects the best available device.
            torch_dtype: Data type for model weights. Defaults to bfloat16 for GPU,
                        float32 for CPU.
        """
        self._model_id = model_id or self.MODEL_ID
        self._device = device or self._get_default_device()
        self._torch_dtype = torch_dtype or self._get_default_dtype()
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._pipeline = None
        
        logger.info(
            "Initializing GemmaExtractor with model=%s, device=%s, dtype=%s",
            self._model_id,
            self._device,
            self._torch_dtype,
        )

    def _get_default_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_default_dtype(self) -> torch.dtype:
        """Determine the best dtype based on device."""
        if self._device in ("cuda", "mps"):
            return torch.bfloat16
        return torch.float32

    def _load_model(self) -> None:
        """Load the model and tokenizer lazily."""
        if self._model is not None:
            return

        logger.info("Loading Gemma model: %s", self._model_id)
        
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=self._torch_dtype,
            device_map=self._device if self._device != "cpu" else None,
        )
        
        if self._device == "cpu":
            self._model = self._model.to("cpu")
        
        # Create text generation pipeline
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            device_map=self._device if self._device != "cpu" else None,
        )
        
        logger.info("Gemma model loaded successfully on device: %s", self._device)

    @property
    def model(self) -> AutoModelForCausalLM:
        """Get or create the Gemma model instance."""
        self._load_model()
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get or create the tokenizer instance."""
        self._load_model()
        return self._tokenizer

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from model response, handling markdown code blocks.

        Args:
            text: Raw response text from the model.

        Returns:
            Parsed JSON dictionary.

        Raises:
            ValueError: If JSON parsing fails.
        """
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```"):
            # Remove ```json or ``` at start and ``` at end
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()

        # Try to find JSON object if response has extra text
        # Look for pattern starting with { and ending with }
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        logger.debug("Cleaned text for JSON parsing:\n%s", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse model response as JSON: %s", e)
            logger.error("Full raw response:\n%s", text)
            raise ValueError(
                f"Failed to parse model response as JSON: {e}\nResponse text: {text[:200]}"
            )

    def extract_from_description(
        self,
        description: str,
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        """Extract ingredients from a text description.

        This is the primary method for ingredient extraction.
        Gemma analyzes the text description to identify ingredients.

        Args:
            description: Text description of the dish.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Dictionary with:
                - dish_name: Extracted or inferred dish name
                - ingredients: List of ingredient names
                - cuisine: Type of cuisine if identifiable
                - confidence: Extraction confidence level
                - source: Always "description"

        Raises:
            ValueError: If description is empty or model response cannot be parsed.
        """
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")

        # Ensure model is loaded
        self._load_model()

        prompt = self.TEXT_PROMPT.format(description=description)
        logger.info("Sending prompt to Gemma for description: %s...", description[:100])

        # Format as chat message for instruction-tuned model
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"}
        ]

        # Generate response
        outputs = self._pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        # Extract generated text
        response_text = outputs[0]["generated_text"]
        
        # If the response is a list of messages, get the assistant's response
        if isinstance(response_text, list):
            # Find the assistant's response
            for msg in response_text:
                if msg.get("role") == "assistant":
                    response_text = msg.get("content", "")
                    break
            else:
                # If no assistant message, use the last message content
                response_text = response_text[-1].get("content", "") if response_text else ""

        logger.info("Gemma raw response text:\n%s", response_text)

        # Parse and return the response
        result = self._parse_json_response(response_text)

        # Validate result is a dictionary
        if not isinstance(result, dict):
            raise ValueError(
                f"Expected dictionary from Gemma, got {type(result).__name__}"
            )

        # Ensure required fields exist with safe access
        dish_name = (
            result.get("dish_name")
            if isinstance(result.get("dish_name"), str)
            else "Unknown Dish"
        )
        ingredients = (
            result.get("ingredients")
            if isinstance(result.get("ingredients"), list)
            else []
        )
        cuisine = (
            result.get("cuisine") if isinstance(result.get("cuisine"), str) else None
        )
        confidence = (
            result.get("confidence")
            if isinstance(result.get("confidence"), str)
            else "medium"
        )

        return {
            "dish_name": dish_name or "Unknown Dish",
            "ingredients": ingredients,
            "cuisine": cuisine,
            "confidence": confidence,
            "source": "description",
        }


# Singleton instance for reuse
_gemma_extractor: GemmaExtractor | None = None


def get_gemma_extractor(
    model_id: str | None = None,
    device: str | None = None,
) -> GemmaExtractor:
    """Get or create a singleton GemmaExtractor instance.

    This function provides a convenient way to reuse the same extractor
    instance across multiple calls, avoiding repeated model loading.

    Args:
        model_id: Hugging Face model ID. Only used on first call.
        device: Device to run the model on. Only used on first call.

    Returns:
        GemmaExtractor instance.
    """
    global _gemma_extractor
    if _gemma_extractor is None:
        _gemma_extractor = GemmaExtractor(model_id=model_id, device=device)
    return _gemma_extractor
