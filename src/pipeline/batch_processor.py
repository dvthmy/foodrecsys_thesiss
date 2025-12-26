"""Batch processor for ingredient extraction pipeline.

Handles parallel processing of multiple dish images using ThreadPoolExecutor,
with progress tracking and automatic temp file cleanup on success.
"""

import json
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
from src.services.gemma_extractor import GemmaExtractor, get_gemma_extractor
from src.services.neo4j_service import Neo4jService
from src.services.clip_embedder import CLIPEmbedder, get_clip_embedder
from src.services.ingredient_canonicalizer import IngredientCanonicalizer, get_canonicalizer

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
class UserIngestResult:
    """Result of ingesting a single user."""

    name: str
    success: bool
    user_id: str | None = None
    is_new: bool = False
    restriction_linked: bool = False
    restriction_skipped: str | None = None
    ratings_linked: int = 0
    ratings_skipped: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class UserIngestJob:
    """Represents a user ingestion job with progress tracking."""

    job_id: str
    total_users: int
    status: ProcessingStatus = ProcessingStatus.PENDING
    created: int = 0
    updated: int = 0
    failed: int = 0
    results: list[UserIngestResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def progress(self) -> float:
        """Calculate progress percentage."""
        if self.total_users == 0:
            return 100.0
        return len(self.results) / self.total_users * 100

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
            "total_users": self.total_users,
            "created": self.created,
            "updated": self.updated,
            "failed": self.failed,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "results": [
                {
                    "name": r.name,
                    "success": r.success,
                    "user_id": r.user_id,
                    "is_new": r.is_new,
                    "restriction_linked": r.restriction_linked,
                    "restriction_skipped": r.restriction_skipped,
                    "ratings_linked": r.ratings_linked,
                    "ratings_skipped": r.ratings_skipped,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


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
        canonicalizer: IngredientCanonicalizer | None = None,
        max_workers: int | None = None,
    ):
        """Initialize the batch processor.

        Args:
            extractor: Gemma extractor service instance.
            clip_embedder: CLIP embedder service instance.
            neo4j_service: Neo4j service instance.
            canonicalizer: Ingredient canonicalizer service instance.
            max_workers: Maximum concurrent workers. Defaults to config value.
        """
        self._gemma = extractor
        self._clip = clip_embedder
        self._neo4j = neo4j_service
        self._canonicalizer = canonicalizer
        self._max_workers = max_workers or config.MAX_WORKERS

        # Job storage with thread-safe access
        self._jobs: dict[str, BatchJob] = {}
        self._jobs_lock = threading.Lock()

        # User ingestion job storage
        self._user_jobs: dict[str, UserIngestJob] = {}
        self._user_jobs_lock = threading.Lock()

    @property
    def extractor(self) -> GemmaExtractor:
        """Get or create Extractor instance."""
        if self._gemma is None:
            # Reuse singleton so the model can be preloaded at setup.
            self._gemma = get_gemma_extractor()
        return self._gemma

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

    @property
    def canonicalizer(self) -> IngredientCanonicalizer:
        """Get or create Ingredient Canonicalizer instance."""
        if self._canonicalizer is None:
            self._canonicalizer = get_canonicalizer()
        return self._canonicalizer

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
        1. Extract ingredients from description using Gemma (text-only)
        2. Generate image embedding using CLIP (for visual similarity)

        Design:
        - Gemma: Analyzes text descriptions to extract ingredients
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
            # Step 1: Extract ingredients using Gemma (from description only)
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

            # Step 3: Canonicalize each ingredient
            # Process ingredients one-by-one for accuracy
            canonical_ingredients = []
            for ingredient in ingredients:
                logging.info("Canonicalizing ingredient: %s", ingredient)
                result = self.canonicalizer.canonicalize(ingredient)
                canonical_name = result["canonical_name"]
                logging.info(
                    "  -> %s (action=%s, score=%.4f)",
                    canonical_name,
                    result["action"],
                    result.get("score", 0.0),
                )
                canonical_ingredients.append(canonical_name)

            # Use canonical ingredients for storage
            ingredients = canonical_ingredients

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
                        if result.temp_path:
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

    # =========================================================================
    # User Ingestion Methods
    # =========================================================================

    def get_user_job(self, job_id: str) -> UserIngestJob | None:
        """Get a user ingestion job by ID.

        Args:
            job_id: The job identifier.

        Returns:
            UserIngestJob instance or None if not found.
        """
        with self._user_jobs_lock:
            return self._user_jobs.get(job_id)

    def _process_single_user(
        self,
        user_data: dict[str, Any],
    ) -> UserIngestResult:
        """Process a single user from JSON data.

        Args:
            user_data: Dictionary with user data from JSON.

        Returns:
            UserIngestResult with ingestion details.
        """
        name = user_data.get("name", "")
        if not name:
            return UserIngestResult(
                name="<unknown>",
                success=False,
                error="User name is required",
            )

        try:
            # Step 1: Merge user by name (idempotent)
            user_result = self.neo4j.merge_user_by_name(
                name=name,
                age=user_data.get("age"),
                gender=user_data.get("gender"),
                nationality=user_data.get("nationality"),
            )

            user_id = user_result["user_id"]
            is_new = user_result.get("is_new", False)

            # Step 2: Handle dietary restriction (exact match, skip if not found)
            restriction_linked = False
            restriction_skipped = None
            dietary_restriction = user_data.get("dietary_restriction")

            if dietary_restriction:
                # Check if restriction exists (exact match)
                restriction = self.neo4j.get_restriction_by_name(dietary_restriction)
                if restriction:
                    rel_result = self.neo4j.create_user_restriction_relationship(
                        user_id=user_id,
                        restriction_name=dietary_restriction,
                    )
                    restriction_linked = rel_result is not None
                else:
                    restriction_skipped = dietary_restriction
                    logging.warning(
                        "Skipping dietary restriction '%s' for user '%s': not found in database",
                        dietary_restriction,
                        name,
                    )

            # Step 3: Handle ratings (exact match dish names, skip if not found)
            ratings_linked = 0
            ratings_skipped = []
            ratings = user_data.get("ratings", {})

            for dish_name, score in ratings.items():
                # Check if dish exists (exact match)
                dish = self.neo4j.get_dish_by_name(dish_name)
                if dish:
                    rating_result = self.neo4j.create_user_rating(
                        user_id=user_id,
                        dish_name=dish_name,
                        score=score,
                    )
                    if rating_result:
                        ratings_linked += 1
                else:
                    ratings_skipped.append(dish_name)
                    logging.warning(
                        "Skipping rating for dish '%s' (user '%s'): dish not found in database",
                        dish_name,
                        name,
                    )

            return UserIngestResult(
                name=name,
                success=True,
                user_id=user_id,
                is_new=is_new,
                restriction_linked=restriction_linked,
                restriction_skipped=restriction_skipped,
                ratings_linked=ratings_linked,
                ratings_skipped=ratings_skipped,
            )

        except Exception as e:
            logging.exception("Error ingesting user '%s'", name)
            return UserIngestResult(
                name=name,
                success=False,
                error=str(e),
            )

    def _process_user_batch(
        self,
        job_id: str,
        users: list[dict[str, Any]],
    ) -> None:
        """Process a batch of users in the background.

        Args:
            job_id: The job identifier.
            users: List of user dictionaries from JSON.
        """
        job = self._user_jobs[job_id]

        with self._user_jobs_lock:
            job.status = ProcessingStatus.PROCESSING
            job.started_at = datetime.now()

        # Process users sequentially (to avoid concurrent Neo4j transaction issues)
        for user_data in users:
            result = self._process_single_user(user_data)

            with self._user_jobs_lock:
                job.results.append(result)

                if result.success:
                    if result.is_new:
                        job.created += 1
                    else:
                        job.updated += 1
                else:
                    job.failed += 1

        # Set final status
        with self._user_jobs_lock:
            job.finished_at = datetime.now()

            if job.failed == 0:
                job.status = ProcessingStatus.COMPLETED
            elif job.created + job.updated == 0:
                job.status = ProcessingStatus.FAILED
            else:
                job.status = ProcessingStatus.PARTIAL

    def ingest_users_from_json(
        self,
        json_path: str | Path,
    ) -> str:
        """Start user ingestion from a JSON file.

        Args:
            json_path: Path to the JSON file containing user data.

        Returns:
            The job ID for tracking progress.

        Raises:
            FileNotFoundError: If the JSON file doesn't exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # Load and parse JSON
        with open(json_path, "r", encoding="utf-8") as f:
            users = json.load(f)

        if not isinstance(users, list):
            raise ValueError("JSON file must contain an array of users")

        job_id = str(uuid.uuid4())
        job = UserIngestJob(job_id=job_id, total_users=len(users))

        with self._user_jobs_lock:
            self._user_jobs[job_id] = job

        # Start processing in background thread
        thread = threading.Thread(
            target=self._process_user_batch,
            args=(job_id, users),
            daemon=True,
        )
        thread.start()

        return job_id

    def ingest_users_sync(
        self,
        json_path: str | Path,
    ) -> UserIngestJob:
        """Synchronously ingest users from a JSON file.

        Args:
            json_path: Path to the JSON file containing user data.

        Returns:
            UserIngestJob with complete results.

        Raises:
            FileNotFoundError: If the JSON file doesn't exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # Load and parse JSON
        with open(json_path, "r", encoding="utf-8") as f:
            users = json.load(f)

        return self.process_users_sync(users)

    def process_users_sync(self, users: list[dict]) -> UserIngestJob:
        """Synchronously process a list of user data.

        Args:
            users: List of user dictionaries.

        Returns:
            UserIngestJob with complete results.
        """
        if not isinstance(users, list):
            raise ValueError("User data must be a list of users")

        job_id = str(uuid.uuid4())
        job = UserIngestJob(job_id=job_id, total_users=len(users))
        job.status = ProcessingStatus.PROCESSING
        job.started_at = datetime.now()

        # Process users synchronously
        for user_data in users:
            result = self._process_single_user(user_data)
            job.results.append(result)

            if result.success:
                if result.is_new:
                    job.created += 1
                else:
                    job.updated += 1
            else:
                job.failed += 1

        # Set final status
        job.finished_at = datetime.now()

        if job.failed == 0:
            job.status = ProcessingStatus.COMPLETED
        elif job.created + job.updated == 0:
            job.status = ProcessingStatus.FAILED
        else:
            job.status = ProcessingStatus.PARTIAL

        return job

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
            # Preload Gemma during setup so the first extraction call doesn't fail.
            extractor = get_gemma_extractor(preload=True)
            _processor = BatchProcessor(extractor=extractor)
        return _processor
