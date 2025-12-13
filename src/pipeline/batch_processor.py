"""Batch processor for ingredient extraction pipeline.

Handles parallel processing of multiple dish images using ThreadPoolExecutor,
with progress tracking and automatic temp file cleanup on success.
"""

import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import config
from src.services.gemini_extractor import GeminiExtractor
from src.services.gemma_extractor import GemmaExtractor, get_gemma_extractor
from src.services.neo4j_service import Neo4jService
from src.services.clip_embedder import CLIPEmbedder, get_clip_embedder

class ProcessingStatus(Enum):
    """Status of a batch processing job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some items succeeded, some failed


@dataclass
class ProcessingResult:
    """Result of processing a single item."""

    item_id: str
    success: bool
    dish_id: str | None = None
    dish_name: str | None = None
    ingredients: list[str] = field(default_factory=list)
    image_embedding: list[float] | None = None
    error: str | None = None
    temp_path: str | None = None


@dataclass
class BatchJob:
    """Represents a batch processing job with progress tracking."""

    job_id: str
    total_items: int
    status: ProcessingStatus = ProcessingStatus.PENDING
    completed: int = 0
    failed: int = 0
    results: dict[str, ProcessingResult] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def progress(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.completed + self.failed) / self.total_items * 100

    @property
    def is_finished(self) -> bool:
        """Check if job has finished processing."""
        return self.status in (
            ProcessingStatus.COMPLETED,
            ProcessingStatus.FAILED,
            ProcessingStatus.PARTIAL,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for API response."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": round(self.progress, 2),
            "total": self.total_items,
            "completed": self.completed,
            "failed": self.failed,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "results": {
                k: {
                    "item_id": v.item_id,
                    "success": v.success,
                    "dish_id": v.dish_id,
                    "dish_name": v.dish_name,
                    "ingredients": v.ingredients,
                    "has_embedding": v.image_embedding is not None,
                    "error": v.error,
                }
                for k, v in self.results.items()
            },
        }


class BatchProcessor:
    """Processor for batch ingredient extraction and image embedding jobs."""

    def __init__(
        self,
        extractor: GemmaExtractor | None = None,
        clip_embedder: CLIPEmbedder | None = None,
        neo4j_service: Neo4jService | None = None,
        max_workers: int | None = None,
    ):
        """Initialize the batch processor.

        Args:
            extractor: Gemma extractor service instance.
            clip_embedder: CLIP embedder service instance.
            neo4j_service: Neo4j service instance.
            max_workers: Maximum concurrent workers. Defaults to config value.
        """
        self._gemini = extractor
        self._clip = clip_embedder
        self._neo4j = neo4j_service
        self._max_workers = max_workers or config.MAX_WORKERS

        # Job storage with thread-safe access
        self._jobs: dict[str, BatchJob] = {}
        self._jobs_lock = threading.Lock()

    @property
    def extractor(self) -> GemmaExtractor:
        """Get or create  Extractor instance."""
        if self._gemini is None:
            self._gemini = GemmaExtractor()
        return self._gemini

    @property
    def clip(self) -> CLIPEmbedder:
        """Get or create CLIP embedder instance."""
        if self._clip is None:
            self._clip = get_clip_embedder()
        return self._clip

    @property
    def neo4j(self) -> Neo4jService:
        """Get or create Neo4j service instance."""
        if self._neo4j is None:
            self._neo4j = Neo4jService()
        return self._neo4j

    def get_job(self, job_id: str) -> BatchJob | None:
        """Get a job by ID.

        Args:
            job_id: The job identifier.

        Returns:
            BatchJob instance or None if not found.
        """
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def _process_single_item(
        self,
        item_id: str,
        temp_path: str,
        description: str | None = None,
        dish_name: str | None = None,
    ) -> ProcessingResult:
        """Process a single dish item.

        Performs two AI operations:
        1. Extract ingredients from description using Gemini API (text-only)
        2. Generate image embedding using CLIP (for visual similarity)

        Design:
        - Gemini: Analyzes text descriptions to extract ingredients
        - CLIP: Generates image embeddings for visual similarity search

        Args:
            item_id: Unique identifier for the item.
            temp_path: Path to the temporary image file.
            description: Text description of the dish (required for ingredient extraction).
            dish_name: Optional dish name (will be extracted from description if not provided).

        Returns:
            ProcessingResult with extraction, embedding, and storage results.
        """
        try:
            # Step 1: Extract ingredients using Gemini (from description only)
            extraction = None
            if description:
                logging.info("Extracting ingredients for item %s using Gemma...", item_id)
                extraction = self.extractor.extract_from_description(description)
            
            # Use provided dish_name or extracted one, or default
            final_dish_name = dish_name or (extraction.get("dish_name") if extraction else None) or "Unknown Dish"
            ingredients = extraction.get("ingredients", []) if extraction else []
            cuisine = extraction.get("cuisine") if extraction else None

            # Step 2: Generate image embedding using CLIP
            image_embedding = self.clip.embed_image(temp_path)

            # Generate dish_id
            dish_id = str(uuid.uuid4())

            # Step 3: Store in Neo4j with embedding
            self.neo4j.merge_dish_with_ingredients(
                dish_id=dish_id,
                name=final_dish_name,
                ingredients=ingredients,
                description=description,
                image_url=temp_path,
                image_embedding=image_embedding,
                country=cuisine,
            )

            return ProcessingResult(
                item_id=item_id,
                success=True,
                dish_id=dish_id,
                dish_name=final_dish_name,
                ingredients=ingredients,
                image_embedding=image_embedding,
                temp_path=temp_path,
            )

        except Exception as e:
            return ProcessingResult(
                item_id=item_id,
                success=False,
                error=str(e),
                temp_path=temp_path,
            )

    def _cleanup_temp_file(self, path: str) -> None:
        """Delete a temporary file if it exists.

        Args:
            path: Path to the file to delete.
        """
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass  # Ignore cleanup errors

    def _process_batch(
        self,
        job_id: str,
        items: list[dict[str, Any]],
    ) -> None:
        """Process a batch of items in the background.

        Args:
            job_id: The job identifier.
            items: List of items to process, each with:
                - id: Unique item identifier
                - temp_path: Path to temporary image file
                - description: Text description (required for ingredient extraction)
                - name: Optional dish name
        """
        job = self._jobs[job_id]

        with self._jobs_lock:
            job.status = ProcessingStatus.PROCESSING
            job.started_at = datetime.now()

        # Process items in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_item = {
                executor.submit(
                    self._process_single_item,
                    item["id"],
                    item["temp_path"],
                    item.get("description"),
                    item.get("name"),
                ): item
                for item in items
            }

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                result = future.result()

                with self._jobs_lock:
                    job.results[item["id"]] = result

                    if result.success:
                        job.completed += 1
                        # Delete temp file on success
                        self._cleanup_temp_file(result.temp_path)
                    else:
                        job.failed += 1
                        # Keep temp file on failure for debugging/retry

        # Set final status
        with self._jobs_lock:
            job.finished_at = datetime.now()

            if job.failed == 0:
                job.status = ProcessingStatus.COMPLETED
            elif job.completed == 0:
                job.status = ProcessingStatus.FAILED
            else:
                job.status = ProcessingStatus.PARTIAL

    def start_batch(
        self,
        items: list[dict[str, Any]],
    ) -> str:
        """Start a batch processing job.

        Args:
            items: List of items to process, each with:
                - id: Unique item identifier
                - temp_path: Path to temporary image file
                - description: Text description (required for ingredient extraction)
                - name: Optional dish name

        Returns:
            The job ID for tracking progress.
        """
        job_id = str(uuid.uuid4())
        job = BatchJob(job_id=job_id, total_items=len(items))

        with self._jobs_lock:
            self._jobs[job_id] = job

        # Start processing in background thread
        thread = threading.Thread(
            target=self._process_batch,
            args=(job_id, items),
            daemon=True,
        )
        thread.start()

        return job_id

    def process_single(
        self,
        temp_path: str,
        description: str | None = None,
        dish_name: str | None = None,
    ) -> ProcessingResult:
        """Process a single item synchronously.

        Args:
            temp_path: Path to the image file.
            description: Text description (required for ingredient extraction).
            dish_name: Optional dish name.

        Returns:
            ProcessingResult with extraction and storage results.
        """
        item_id = str(uuid.uuid4())
        result = self._process_single_item(item_id, temp_path, description, dish_name)

        # Cleanup on success
        if result.success:
            self._cleanup_temp_file(temp_path)

        return result

    def close(self) -> None:
        """Close service connections."""
        if self._neo4j is not None:
            self._neo4j.close()


# Global processor instance
_processor: BatchProcessor | None = None
_processor_lock = threading.Lock()


def get_processor() -> BatchProcessor:
    """Get or create the global batch processor instance.

    Returns:
        BatchProcessor singleton instance.
    """
    global _processor
    with _processor_lock:
        if _processor is None:
            _processor = BatchProcessor()
        return _processor
