"""FastAPI routes for dish image upload and ingredient extraction.

Provides endpoints for:
- Batch image upload
- Processing uploaded images (Gemini for description, CLIP for images)
- Checking job status

Design:
- Gemini API: Extracts ingredients from text descriptions only
- CLIP: Generates image embeddings for visual similarity search
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.config import config
from src.pipeline.batch_processor import get_processor

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
        description="Text description - required for ingredient extraction via Gemini"
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


class PendingIngredient(BaseModel):
    """A pending ingredient awaiting approval."""

    name: str
    created_at: str | None = None
    dish_count: int = 0


class PendingIngredientsResponse(BaseModel):
    """Response from pending ingredients endpoint."""

    pending: list[PendingIngredient]
    count: int


class ApproveIngredientRequest(BaseModel):
    """Request to approve a pending ingredient."""

    pass  # No body needed, name comes from path


class RejectIngredientRequest(BaseModel):
    """Request to reject a pending ingredient and merge into another."""

    merge_into: str = Field(description="Canonical ingredient to merge relationships into")


class IngredientActionResponse(BaseModel):
    """Response from ingredient approval/rejection."""

    success: bool
    name: str
    action: str
    message: str


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
    1. Gemini API: Extracts ingredients from text descriptions
    2. CLIP: Generates image embeddings for visual similarity
    3. Neo4j: Stores dish with ingredients and embedding

    Args:
        images: One or more image files.
        names: Optional dish names matching image order.
        descriptions: Descriptions matching image order (required for Gemini extraction).

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
        )
        for p in pending
    ]

    return PendingIngredientsResponse(pending=pending_list, count=len(pending_list))


@router.post(
    "/ingredients/{name}/approve",
    response_model=IngredientActionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def approve_ingredient(name: str) -> IngredientActionResponse:
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

    return IngredientActionResponse(
        success=True,
        name=result["name"],
        action="approved",
        message=f"Ingredient '{result['name']}' is now canonical",
    )


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
