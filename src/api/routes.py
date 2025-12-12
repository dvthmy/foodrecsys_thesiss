"""FastAPI routes for dish image upload and ingredient extraction.

Provides endpoints for:
- Batch image upload
- Processing uploaded images with Gemini
- Checking job status
"""

import os
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.config import config
from src.pipeline.batch_processor import get_processor

# Create API router
router = APIRouter(prefix="/api/v1", tags=["dishes"])


# Pydantic models for request/response
class ProcessItem(BaseModel):
    """Item to process from upload response."""

    id: str
    temp_path: str
    description: str | None = None


class ProcessRequest(BaseModel):
    """Request body for process endpoint."""

    items: list[ProcessItem]


class UploadResult(BaseModel):
    """Result of a single file upload."""

    id: str
    original_name: str
    temp_path: str
    description: str | None = None


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
    "/dishes/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
)
async def upload_dish_images(
    images: Annotated[list[UploadFile], File(description="Dish images to upload")],
    descriptions: Annotated[
        list[str] | None, Form(description="Descriptions for each image")
    ] = None,
) -> UploadResponse:
    """Upload dish images for ingredient extraction.

    Args:
        images: One or more image files.
        descriptions: Optional descriptions matching image order.

    Returns:
        Upload results with item IDs for processing.
    """
    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "No images provided", "code": "NO_IMAGES"},
        )

    descriptions = descriptions or []
    uploaded: list[UploadResult] = []
    errors: list[UploadError] = []

    for idx, file in enumerate(images):
        if not file.filename:
            continue

        if allowed_file(file.filename):
            try:
                item = await save_uploaded_file(file)

                # Add description if provided
                if idx < len(descriptions):
                    item["description"] = descriptions[idx]

                uploaded.append(UploadResult(**item))
            except Exception as e:
                errors.append(UploadError(filename=file.filename, error=str(e)))
        else:
            errors.append(
                UploadError(
                    filename=file.filename,
                    error=f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}",
                )
            )

    if not uploaded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "No valid files uploaded",
                "errors": [e.model_dump() for e in errors],
                "code": "NO_VALID_FILES",
            },
        )

    return UploadResponse(
        uploaded=uploaded,
        errors=errors,
        total=len(uploaded),
        failed=len(errors),
    )


@router.post(
    "/dishes/process",
    response_model=ProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={400: {"model": ErrorResponse}},
)
async def process_dishes(request: ProcessRequest) -> ProcessResponse:
    """Start processing uploaded dish images.

    Args:
        request: Processing request with items from upload response.

    Returns:
        Job ID for tracking progress.
    """
    if not request.items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Items array is empty", "code": "EMPTY_ITEMS"},
        )

    # Validate items
    valid_items = []
    for item in request.items:
        # Check if file exists
        if not os.path.exists(item.temp_path):
            continue
        valid_items.append(item.model_dump())

    if not valid_items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "No valid items to process", "code": "NO_VALID_ITEMS"},
        )

    # Start batch processing
    processor = get_processor()
    job_id = processor.start_batch(valid_items)

    return ProcessResponse(
        job_id=job_id,
        status="processing",
        total_items=len(valid_items),
        status_url=f"/api/v1/jobs/{job_id}/status",
    )


@router.post(
    "/dishes/upload-and-process",
    response_model=ProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={400: {"model": ErrorResponse}},
)
async def upload_and_process(
    images: Annotated[list[UploadFile], File(description="Dish images to upload")],
    descriptions: Annotated[
        list[str] | None, Form(description="Descriptions for each image")
    ] = None,
) -> ProcessResponse:
    """Upload and immediately process dish images.

    Convenience endpoint that combines upload and process steps.

    Args:
        images: One or more image files.
        descriptions: Optional descriptions matching image order.

    Returns:
        Job ID for tracking progress.
    """
    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "No images provided", "code": "NO_IMAGES"},
        )

    descriptions = descriptions or []
    items: list[dict] = []
    errors: list[UploadError] = []

    for idx, file in enumerate(images):
        if not file.filename:
            continue

        if allowed_file(file.filename):
            try:
                item = await save_uploaded_file(file)

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
