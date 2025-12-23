"""FastAPI routes for dish image upload and ingredient extraction.

Provides endpoints for:
- Batch image upload
- Processing uploaded images (Gemma 3 for description, CLIP for images)
- Checking job status

Design:
- Gemma 3: Extracts ingredients from text descriptions only
- CLIP: Generates image embeddings for visual similarity search
"""

import logging
import os
import json
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.config import config
from src.pipeline.batch_processor import get_processor
from src.services.recommendation_service import get_recommendation_service
from src.services.clip_embedder import get_clip_embedder

# Create API router
router = APIRouter(prefix="/api/v1", tags=["dishes"])


# Pydantic models for request/response
class ProcessItem(BaseModel):
    """Item to process from upload response."""

    id: str
    temp_path: str
    name: str | None = Field(default=None, description="Dish name")
    description: str | None = Field(
        default=None, 
        description="Text description - required for ingredient extraction via Gemma 3"
    )


class ProcessRequest(BaseModel):
    """Request body for process endpoint."""

    items: list[ProcessItem]


class UploadResult(BaseModel):
    """Result of a single file upload."""

    id: str
    original_name: str
    temp_path: str
    name: str | None = Field(default=None, description="Dish name")
    description: str | None = Field(
        default=None, 
        description="Text description - required for ingredient extraction"
    )


class UploadError(BaseModel):
    """Error from a failed upload."""

    filename: str
    error: str


class UploadResponse(BaseModel):
    """Response from upload endpoint."""

    uploaded: list[UploadResult]
    errors: list[UploadError]
    total: int
    failed: int


class ProcessResponse(BaseModel):
    """Response from process endpoint."""

    job_id: str
    status: str
    total_items: int
    status_url: str
    upload_errors: list[UploadError] | None = None


class DishSearchResult(BaseModel):
    """Result of a dish search."""

    dish_id: str
    name: str
    description: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    score: float
    image_url: str | None = None


class DishSearchResponse(BaseModel):
    """Response from dish search endpoint."""

    results: list[DishSearchResult]
    count: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    code: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str


class IngredientsResponse(BaseModel):
    """Response from ingredients endpoint."""

    ingredients: list[str]
    count: int


class DishSummary(BaseModel):
    """Summary of a dish."""

    dish_id: str
    name: str
    description: str | None = None
    image_url: str | None = None
    country: str | None = None


class DishesResponse(BaseModel):
    """Response for list of dishes."""

    dishes: list[DishSummary]
    count: int


class PendingIngredient(BaseModel):
    """A pending ingredient awaiting approval."""

    name: str
    created_at: str | None = None
    dish_count: int = 0
    pending_confidence: str | None = Field(
        default=None,
        description="Pending confidence label (e.g., 'medium', 'low')",
    )
    pending_best_score: float | None = Field(
        default=None,
        description="Best similarity score observed during canonicalization",
    )
    suggested_merges: list[dict] = Field(
        default_factory=list,
        description="Suggested canonical ingredients to merge into (top-k), each item typically contains 'name' and 'score'",
    )


class PendingIngredientsResponse(BaseModel):
    """Response from pending ingredients endpoint."""

    pending: list[PendingIngredient]
    count: int


class ApproveIngredientRequest(BaseModel):
    """Request to approve a pending ingredient."""

    suited_for: list[str] = Field(
        default_factory=list,
        description="Dietary restrictions this ingredient is suited for (creates SUITED_FOR relationships)",
    )
    not_suited_for: list[str] = Field(
        default_factory=list,
        description="Dietary restrictions this ingredient is NOT suited for (creates NOT_SUITED_FOR relationships)",
    )


class DietaryRestrictionSummary(BaseModel):
    """Summary model for dietary restrictions."""

    name: str
    description: str | None = None


class DietaryRestrictionsResponse(BaseModel):
    """Response from dietary restrictions endpoint."""

    restrictions: list[DietaryRestrictionSummary]
    count: int


class RejectIngredientRequest(BaseModel):
    """Request to reject a pending ingredient and merge into another."""

    merge_into: str = Field(description="Canonical ingredient to merge relationships into")


class IngredientActionResponse(BaseModel):
    """Response from ingredient approval/rejection."""

    success: bool
    name: str
    action: str
    message: str


# =========================================================================
# User Models
# =========================================================================


class UserRating(BaseModel):
    """A user's rating of a dish."""

    dish_name: str
    dish_id: str | None = None
    score: int = Field(ge=1, le=5, description="Rating score from 1 to 5")


class UserResponse(BaseModel):
    """Response model for a single user."""

    user_id: str
    name: str
    age: int | str | None = None
    gender: int | str | None = None
    nationality: str | None = None
    dietary_restrictions: list[str] = Field(default_factory=list)
    ratings: list[UserRating] = Field(default_factory=list)


class UserSummaryResponse(BaseModel):
    """Summary response model for a user in list view."""

    user_id: str
    name: str
    age: int | str | None = None
    gender: int | str | None = None
    nationality: str | None = None
    dietary_restrictions: list[str] = Field(default_factory=list)
    rating_count: int = 0


class UsersListResponse(BaseModel):
    """Response from list users endpoint."""

    users: list[UserSummaryResponse]
    count: int


class UserIngestResultResponse(BaseModel):
    """Result of ingesting a single user."""

    name: str
    success: bool
    user_id: str | None = None
    is_new: bool = False
    restriction_linked: bool = False
    restriction_skipped: str | None = None
    ratings_linked: int = 0
    ratings_skipped: list[str] = Field(default_factory=list)
    error: str | None = None


class UserIngestJobResponse(BaseModel):
    """Response from user ingestion endpoint."""

    job_id: str
    status: str
    progress: float
    total_users: int
    created: int
    updated: int
    failed: int
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    results: list[UserIngestResultResponse] = Field(default_factory=list)


# =========================================================================
# Recommendation Models
# =========================================================================


class DishRecommendationResponse(BaseModel):
    """A recommended dish with prediction details."""

    dish_id: str
    name: str
    predicted_score: float = Field(description="Predicted rating score (1-5)")
    description: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    recommender_count: int = Field(
        default=0,
        description="Number of similar users who rated this dish",
    )
    reason: str = Field(
        default="collaborative_filtering",
        description="Reason for recommendation: 'collaborative_filtering' or 'popular_dish'",
    )


class SimilarUserResponse(BaseModel):
    """A similar user with similarity score."""

    user_id: str
    name: str
    similarity: float = Field(description="Cosine similarity score (0-1)")
    shared_dishes: int = Field(
        default=0,
        description="Number of dishes both users have rated",
    )


class RecommendationsResponse(BaseModel):
    """Response from recommendations endpoint."""

    user_id: str
    recommendations: list[DishRecommendationResponse]
    count: int
    method: str = Field(
        description="Method used: 'collaborative_filtering' or 'popular_fallback'",
    )


class SimilarUsersResponse(BaseModel):
    """Response from similar users endpoint."""

    user_id: str
    similar_users: list[SimilarUserResponse]
    count: int
    similarity_threshold: float


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed.

    Args:
        filename: The filename to check.

    Returns:
        True if extension is allowed.
    """
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS


def secure_filename(filename: str) -> str:
    """Sanitize a filename to be safe for filesystem.

    Args:
        filename: The original filename.

    Returns:
        Sanitized filename.
    """
    # Remove path separators and null bytes
    filename = filename.replace("/", "_").replace("\\", "_").replace("\x00", "")
    # Keep only safe characters
    return "".join(c for c in filename if c.isalnum() or c in "._-")


async def save_uploaded_file(file: UploadFile) -> dict:
    """Save an uploaded file to temporary storage.

    Args:
        file: FastAPI UploadFile object.

    Returns:
        Dictionary with file metadata.
    """
    filename = secure_filename(file.filename or "unknown")
    file_id = str(uuid.uuid4())
    ext = Path(filename).suffix

    # Ensure temp directory exists
    config.ensure_temp_dir()

    # Create temp file
    temp_path = config.TEMP_UPLOAD_DIR / f"{file_id}{ext}"

    # Read and save file content
    content = await file.read()
    temp_path.write_bytes(content)

    return {
        "id": file_id,
        "original_name": filename,
        "temp_path": str(temp_path),
    }


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status response.
    """
    return HealthResponse(status="healthy", service="food-recsys")

@router.post(
    "/dishes/upload-and-process",
    response_model=ProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={400: {"model": ErrorResponse}},
)
async def upload_and_process(
    images: Annotated[list[UploadFile], File(description="Dish images to upload")],
    names: Annotated[
        list[str] | None, Form(description="Names for each dish (matching image order)")
    ] = None,
    descriptions: Annotated[
        list[str] | None, Form(description="Descriptions for each image (required for ingredient extraction)")
    ] = None,
) -> ProcessResponse:
    """Upload and immediately process dish images.

    Convenience endpoint that combines upload and process steps.
    
    Processing Flow:
    1. Gemma 3: Extracts ingredients from text descriptions
    2. CLIP: Generates image embeddings for visual similarity
    3. Neo4j: Stores dish with ingredients and embedding

    Args:
        images: One or more image files.
        names: Optional dish names matching image order.
        descriptions: Descriptions matching image order (required for Gemma 3 extraction).

    Returns:
        Job ID for tracking progress.
    """
    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "No images provided", "code": "NO_IMAGES"},
        )

    names = names or []
    descriptions = descriptions or []
    items: list[dict] = []
    errors: list[UploadError] = []

    for idx, file in enumerate(images):
        if not file.filename:
            continue

        if allowed_file(file.filename):
            try:
                item = await save_uploaded_file(file)

                # Add name if provided
                if idx < len(names):
                    item["name"] = names[idx]
                    
                if idx < len(descriptions):
                    item["description"] = descriptions[idx]

                items.append(item)
            except Exception as e:
                errors.append(UploadError(filename=file.filename, error=str(e)))
        else:
            errors.append(
                UploadError(
                    filename=file.filename,
                    error=f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}",
                )
            )

    if not items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "No valid files to process",
                "errors": [e.model_dump() for e in errors],
                "code": "NO_VALID_FILES",
            },
        )

    # Start batch processing
    processor = get_processor()
    logging.info("Starting batch processing for %d items...", len(items))
    job_id = processor.start_batch(items)

    return ProcessResponse(
        job_id=job_id,
        status="processing",
        total_items=len(items),
        upload_errors=errors if errors else None,
        status_url=f"/api/v1/jobs/{job_id}/status",
    )


@router.get(
    "/jobs/{job_id}/status",
    responses={404: {"model": ErrorResponse}},
)
async def get_job_status(job_id: str) -> dict:
    """Get the status of a processing job.

    Args:
        job_id: The job identifier.

    Returns:
        Job status and results.
    """
    processor = get_processor()
    job = processor.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Job not found", "code": "JOB_NOT_FOUND"},
        )

    return job.to_dict()


@router.get("/dishes/", response_model=DishesResponse)
async def get_dishes(
    limit: int = 100,
    skip: int = 0,
) -> DishesResponse:
    """Get a list of dishes.

    Args:
        limit: Maximum number of dishes to return.
        skip: Number of dishes to skip.

    Returns:
        List of dishes.
    """
    processor = get_processor()
    dishes = processor.neo4j.get_dishes(limit=limit, skip=skip)

    return DishesResponse(
        dishes=[DishSummary(**d) for d in dishes],
        count=len(dishes),
    )


@router.get(
    "/dishes/{dish_id}",
    responses={404: {"model": ErrorResponse}},
)
async def get_dish(dish_id: str) -> dict:
    """Get a dish by ID from the database.

    Args:
        dish_id: The dish identifier.

    Returns:
        Dish data with ingredients.
    """
    processor = get_processor()
    dish = processor.neo4j.get_dish_by_id(dish_id)
    if dish is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Dish not found", "code": "DISH_NOT_FOUND"},
        )

    return dish


@router.post(
    "/dishes/search-by-image",
    response_model=DishSearchResponse,
    responses={400: {"model": ErrorResponse}},
)
async def search_dishes_by_image(
    image: Annotated[UploadFile, File(description="Dish image to search for")],
    limit: int = 5,
    threshold: float = 0.7,
) -> DishSearchResponse:
    """Search for similar dishes using an image.

    Args:
        image: The image file to search with.
        limit: Maximum number of results.
        threshold: Minimum similarity score.

    Returns:
        List of similar dishes with ingredients.
    """
    if not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "No image provided", "code": "NO_IMAGE"},
        )

    if not allowed_file(image.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}",
                "code": "INVALID_FILE_TYPE",
            },
        )

    try:
        # Save temp file
        item = await save_uploaded_file(image)
        temp_path = item["temp_path"]

        # Generate embedding
        embedder = get_clip_embedder()
        embedding = embedder.embed_image(temp_path)

        # Search in Neo4j
        processor = get_processor()
        results = processor.neo4j.find_similar_dishes_by_embedding(
            embedding=embedding,
            k=limit,
            threshold=threshold,
        )

        # Cleanup temp file
        try:
            os.remove(temp_path)
        except OSError:
            pass

        return DishSearchResponse(
            results=[
                DishSearchResult(
                    dish_id=r["dish_id"],
                    name=r["name"],
                    description=r.get("description"),
                    ingredients=r.get("ingredients", []),
                    score=r["score"],
                    image_url=r.get("image_url"),
                )
                for r in results
            ],
            count=len(results),
        )

    except Exception as e:
        logging.error("Search failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e), "code": "SEARCH_FAILED"},
        )


@router.get("/ingredients", response_model=IngredientsResponse)
async def get_all_ingredients() -> IngredientsResponse:
    """Get all ingredients from the database.

    Returns:
        List of all ingredient names.
    """
    processor = get_processor()
    ingredients = processor.neo4j.get_all_ingredients()

    return IngredientsResponse(ingredients=ingredients, count=len(ingredients))


# =========================================================================
# Admin Endpoints for Ingredient Canonicalization
# =========================================================================


@router.get("/ingredients/pending", response_model=PendingIngredientsResponse)
async def get_pending_ingredients() -> PendingIngredientsResponse:
    """Get all pending (non-canonical) ingredients awaiting approval.

    Returns:
        List of pending ingredients with metadata.
    """
    processor = get_processor()
    pending = processor.neo4j.get_pending_ingredients()

    pending_list = [
        PendingIngredient(
            name=p["name"],
            created_at=str(p.get("created_at")) if p.get("created_at") else None,
            dish_count=p.get("dish_count", 0),
            pending_confidence=p.get("pending_confidence"),
            pending_best_score=p.get("pending_best_score"),
            suggested_merges=p.get("suggested_merges") or [],
        )
        for p in pending
    ]

    return PendingIngredientsResponse(pending=pending_list, count=len(pending_list))


@router.post(
    "/ingredients/{name}/approve",
    response_model=IngredientActionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def approve_ingredient(
    name: str,
    request: ApproveIngredientRequest | None = None,
) -> IngredientActionResponse:
    """Approve a pending ingredient as canonical.

    This marks the ingredient as canonical, making it available
    for matching in the canonicalization pipeline.

    Args:
        name: The ingredient name to approve.

    Returns:
        Success confirmation.
    """
    processor = get_processor()
    result = processor.neo4j.approve_ingredient(name)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Ingredient '{name}' not found", "code": "INGREDIENT_NOT_FOUND"},
        )

    suited_added = 0
    not_suited_added = 0
    missing_restrictions: list[str] = []

    if request is not None:
        for restriction in request.suited_for:
            rel = processor.neo4j.create_ingredient_restriction_relationship(
                ingredient_name=result["name"],
                restriction_name=restriction,
                relationship_type="SUITED_FOR",
            )
            if rel is None:
                missing_restrictions.append(restriction)
            else:
                suited_added += 1

        for restriction in request.not_suited_for:
            rel = processor.neo4j.create_ingredient_restriction_relationship(
                ingredient_name=result["name"],
                restriction_name=restriction,
                relationship_type="NOT_SUITED_FOR",
            )
            if rel is None:
                missing_restrictions.append(restriction)
            else:
                not_suited_added += 1

    msg = f"Ingredient '{result['name']}' is now canonical"
    if request is not None:
        msg += f"; restrictions linked (suited={suited_added}, not_suited={not_suited_added})"
        if missing_restrictions:
            msg += f"; missing restrictions skipped: {', '.join(sorted(set(missing_restrictions)))}"

    return IngredientActionResponse(
        success=True,
        name=result["name"],
        action="approved",
        message=msg,
    )


@router.get(
    "/dietary-restrictions",
    response_model=DietaryRestrictionsResponse,
    tags=["dietary_restrictions"],
)
async def get_dietary_restrictions() -> DietaryRestrictionsResponse:
    """Get all dietary restrictions.

    Returns:
        List of dietary restriction names and descriptions.
    """
    processor = get_processor()
    restrictions = processor.neo4j.get_all_dietary_restrictions()

    items = [
        DietaryRestrictionSummary(name=r["name"], description=r.get("description"))
        for r in restrictions
    ]

    return DietaryRestrictionsResponse(restrictions=items, count=len(items))


@router.post(
    "/ingredients/{name}/reject",
    response_model=IngredientActionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def reject_ingredient(
    name: str,
    request: RejectIngredientRequest,
) -> IngredientActionResponse:
    """Reject a pending ingredient and merge its relationships into another.

    All dishes that contain the rejected ingredient will be updated
    to contain the target canonical ingredient instead. The rejected
    ingredient is then deleted.

    Args:
        name: The ingredient name to reject.
        request: Contains the canonical ingredient to merge into.

    Returns:
        Success confirmation with merge details.
    """
    processor = get_processor()
    result = processor.neo4j.reject_ingredient(name, request.merge_into)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": f"Ingredient '{name}' or '{request.merge_into}' not found",
                "code": "INGREDIENT_NOT_FOUND",
            },
        )

    return IngredientActionResponse(
        success=True,
        name=name,
        action="rejected",
        message=f"Ingredient '{name}' merged into '{result['merged_into']}' ({result['merged_count']} dishes updated)",
    )


# =========================================================================
# User Endpoints
# =========================================================================


@router.post(
    "/users/ingest",
    response_model=UserIngestJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["users"],
)
async def ingest_users(
    file: UploadFile = File(..., description="JSON file containing user data"),
) -> UserIngestJobResponse:
    """Ingest users from a JSON file.

    Loads users from the uploaded JSON file, creates User nodes,
    links dietary restrictions (HAS_RESTRICTION), and creates
    ratings (RATED) relationships to dishes.

    - Uses name-based lookup for idempotency (updates existing users)
    - Requires exact match for dietary restrictions (skips if not found)
    - Requires exact match for dish names (skips unmatched ratings)

    Args:
        file: JSON file containing user data.

    Returns:
        Job status with ingestion results.
    """
    if file.content_type != "application/json" and not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "File must be a JSON file", "code": "INVALID_FILE_TYPE"},
        )

    try:
        content = await file.read()
        users = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid JSON format", "code": "INVALID_JSON"},
        )

    if not isinstance(users, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "JSON must contain a list of users", "code": "INVALID_DATA_FORMAT"},
        )

    processor = get_processor()

    try:
        # Use synchronous ingestion for immediate results
        job = processor.process_users_sync(users)

        return UserIngestJobResponse(
            job_id=job.job_id,
            status=job.status.value,
            progress=round(job.progress, 2),
            total_users=job.total_users,
            created=job.created,
            updated=job.updated,
            failed=job.failed,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            finished_at=job.finished_at.isoformat() if job.finished_at else None,
            results=[
                UserIngestResultResponse(
                    name=r.name,
                    success=r.success,
                    user_id=r.user_id,
                    is_new=r.is_new,
                    restriction_linked=r.restriction_linked,
                    restriction_skipped=r.restriction_skipped,
                    ratings_linked=r.ratings_linked,
                    ratings_skipped=r.ratings_skipped,
                    error=r.error,
                )
                for r in job.results
            ],
        )
    except Exception as e:
        logging.error(f"Error ingesting users: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e), "code": "INGESTION_FAILED"},
        )


@router.get(
    "/users",
    response_model=UsersListResponse,
    tags=["users"],
)
async def get_all_users(limit: int = 1000) -> UsersListResponse:
    """Get all users from the database.

    Args:
        limit: Maximum number of users to return (default 1000).

    Returns:
        List of users with their dietary restrictions and rating counts.
    """
    processor = get_processor()
    users = processor.neo4j.get_all_users(limit=limit)

    return UsersListResponse(
        users=[
            UserSummaryResponse(
                user_id=u["user_id"],
                name=u["name"],
                age=u.get("age"),
                gender=u.get("gender"),
                nationality=u.get("nationality"),
                dietary_restrictions=u.get("dietary_restrictions", []),
                rating_count=u.get("rating_count", 0),
            )
            for u in users
        ],
        count=len(users),
    )


@router.get(
    "/users/{user_id}",
    response_model=UserResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["users"],
)
async def get_user(user_id: str) -> UserResponse:
    """Get a user by ID with their dietary restrictions and ratings.

    Args:
        user_id: The user identifier.

    Returns:
        User data with restrictions and ratings.
    """
    processor = get_processor()
    user = processor.neo4j.get_user_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "User not found", "code": "USER_NOT_FOUND"},
        )

    return UserResponse(
        user_id=user["user_id"],
        name=user["name"],
        age=user.get("age"),
        gender=user.get("gender"),
        nationality=user.get("nationality"),
        dietary_restrictions=user.get("dietary_restrictions", []),
        ratings=[
            UserRating(
                dish_name=r["dish_name"],
                dish_id=r.get("dish_id"),
                score=r["score"],
            )
            for r in user.get("ratings", [])
        ],
    )


@router.get(
    "/users/{user_id}/recommendations",
    response_model=RecommendationsResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["recommendations"],
)
async def get_recommendations(
    user_id: str,
    limit: int = 10,
    method: str = "collaborative",
    metric: str = "cosine",
) -> RecommendationsResponse:
    """Get dish recommendations for a user.

    Args:
        user_id: The user identifier.
        limit: Maximum number of recommendations.
        method: Recommendation method:
            - 'collaborative': Collaborative filtering
            - 'content_based': Content-based filtering (ingredients)
            - 'image_based': Image-based recommendations (CLIP embeddings)
        metric: Similarity metric:
            - For 'collaborative': 'cosine' or 'jaccard'
            - For 'content_based': 'jaccard' or 'ingredient_embedding'

    Returns:
        List of recommended dishes.
    """
    processor = get_processor()
    user = processor.neo4j.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "User not found", "code": "USER_NOT_FOUND"},
        )

    rec_service = get_recommendation_service()
    
    if method == "content_based":
        # For content-based:
        # - 'jaccard': Jaccard similarity on ingredient sets
        # - 'ingredient_embedding': Semantic similarity using Gemma ingredient embeddings + TF-IDF aggregation
        if metric not in ["jaccard", "ingredient_embedding"]:
            metric = "jaccard" # Default for content-based
            
        recs = rec_service.recommend_content_based(user_id=user_id, k=limit, metric=metric)
        method_used = f"content_based_{metric}"
    elif method == "image_based":
        # Image-based recommendations using CLIP image embeddings
        recs = rec_service.recommend_image_by_image(user_id=user_id, k=limit)
        method_used = "image_based"
    else:
        # For collaborative, 'cosine' or 'jaccard' on user vectors
        if metric not in ["cosine", "jaccard"]:
            metric = "cosine" # Default for collaborative
            
        recs = rec_service.recommend_dishes(user_id=user_id, k=limit, metric=metric)
        # Check if it fell back to popular
        if recs and recs[0].reason == "popular_dish":
            method_used = "popular_fallback"
        else:
            method_used = f"collaborative_filtering_{metric}"

    return RecommendationsResponse(
        user_id=user_id,
        recommendations=[
            DishRecommendationResponse(
                dish_id=r.dish_id,
                name=r.name,
                predicted_score=r.predicted_score,
                description=r.description,
                ingredients=r.ingredients,
                recommender_count=r.recommender_count,
                reason=r.reason,
            )
            for r in recs
        ],
        count=len(recs),
        method=method_used,
    )


@router.get(
    "/users/{user_id}/similar",
    response_model=SimilarUsersResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["recommendations"],
)
async def get_similar_users(
    user_id: str,
    limit: int = 10,
) -> SimilarUsersResponse:
    """Get users similar to the target user.

    Args:
        user_id: The user identifier.
        limit: Maximum number of similar users.

    Returns:
        List of similar users.
    """
    processor = get_processor()
    user = processor.neo4j.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "User not found", "code": "USER_NOT_FOUND"},
        )

    rec_service = get_recommendation_service()
    similar = rec_service.find_similar_users(user_id=user_id, k=limit)

    return SimilarUsersResponse(
        user_id=user_id,
        similar_users=[
            SimilarUserResponse(
                user_id=u.user_id,
                name=u.name,
                similarity=u.similarity,
                shared_dishes=u.shared_dishes,
            )
            for u in similar
        ],
        count=len(similar),
        similarity_threshold=0.5,  # Hardcoded for now, matches service constant
    )
