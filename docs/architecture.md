# Architecture Overview

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Server                             │
├─────────────────────────────────────────────────────────────────┤
│  POST /api/v1/dishes/upload          → Batch image upload       │
│  POST /api/v1/dishes/process         → Start extraction job     │
│  POST /api/v1/dishes/upload-and-process → Combined endpoint     │
│  GET  /api/v1/jobs/{id}/status       → Poll job progress        │
│  GET  /api/v1/dishes/{id}            → Get dish from database   │
│  GET  /api/v1/ingredients            → List all ingredients     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Batch Processor (ThreadPoolExecutor)               │
├─────────────────────────────────────────────────────────────────┤
│  • max_workers=5 for parallel processing                        │
│  • Progress tracking with job status API                        │
│  • Automatic temp file cleanup on success                       │
│  • Partial failure handling (continue on errors)                │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Gemma 3 API   │  │   CLIP Model    │  │     Neo4j       │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ gemma-3         │  │ ViT-B/32        │  │ Graph database  │
│ TEXT-ONLY       │  │ Image embedding │  │ MERGE to avoid  │
│ Ingredient      │  │ 512-dim vectors │  │ duplicates      │
│ extraction from │  │ Visual features │  │                 │
│ descriptions    │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## AI Responsibilities

| Model | Purpose | Input | Output |
|-------|---------|-------|--------|
| **Gemma 3** | Ingredient extraction | Text description | Dish name, ingredients list, cuisine |
| **CLIP ViT-B/32** | Visual embedding | Dish image | 512-dim vector for similarity search |

> **Design Principle**: GEMMA 3 handles **text/description analysis only**. CLIP handles **image embeddings only**. This separation ensures clear responsibilities and optimal use of each model's strengths.

## Directory Structure

```
food-recsys/
├── main.py                    # FastAPI app entry point
├── pyproject.toml             # Project dependencies
├── docker-compose.yml         # Neo4j database setup
├── .env.example               # Environment template
├── docs/                      # Documentation
└── src/
    ├── __init__.py
    ├── config.py              # Configuration from env vars
    ├── api/
    │   ├── __init__.py
    │   └── routes.py          # FastAPI endpoints
    ├── services/
    │   ├── __init__.py
    │   ├── gemma_extractor.py    # Gemma ingredient extraction
    │   ├── clip_embedder.py       # CLIP image embedding
    │   └── neo4j_service.py       # Neo4j database operations
    └── pipeline/
        ├── __init__.py
        └── batch_processor.py     # Background job processing
```

## Data Flow

1. **Image Upload**: Client uploads dish images with names/descriptions via multipart/form-data
2. **Temporary Storage**: Images saved to `/tmp/food-recsys/uploads/`
3. **Background Processing**: ThreadPoolExecutor processes images in parallel
4. **GEMMA 3 Extraction**: Text description analyzed by GEMMA 3 for ingredients
5. **CLIP Embedding**: Each image embedded using CLIP ViT-B/32 (512-dim vector)
6. **Neo4j Storage**: Dish (with embedding), Ingredient nodes, and relationships created
7. **Cleanup**: Temp files deleted on success, retained on failure for retry

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **FastAPI** | Async support, auto OpenAPI docs, Pydantic validation |
| **ThreadPoolExecutor** | Simple for MVP, easy to swap for Celery later |
| **MERGE queries** | Idempotent operations, no duplicate ingredients |
| **Job-based processing** | Non-blocking uploads, progress tracking |
| **Temp file cleanup** | Delete on success, keep on failure for debugging |
| **CLIP ViT-B/32** | Good balance of speed vs quality, 512-dim embeddings |
| **Lazy model loading** | CLIP model loaded on first use to reduce startup time |
| **GEMMA 3 text-only** | Uses GEMMA 3 for description parsing, CLIP for images |
| **Separated AI concerns** | Clear responsibility: GEMMA 3=text, CLIP=images |
