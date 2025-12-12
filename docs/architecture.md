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
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│      Gemini API         │     │        Neo4j            │
├─────────────────────────┤     ├─────────────────────────┤
│  gemini-1.5-flash       │     │  Graph database         │
│  Multimodal analysis    │     │  MERGE to avoid dupes   │
│  JSON structured output │     │  Unique constraints     │
└─────────────────────────┘     └─────────────────────────┘
```

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
    │   ├── gemini_extractor.py    # Gemini AI integration
    │   └── neo4j_service.py       # Neo4j database operations
    └── pipeline/
        ├── __init__.py
        └── batch_processor.py     # Background job processing
```

## Data Flow

1. **Image Upload**: Client uploads dish images via multipart/form-data
2. **Temporary Storage**: Images saved to `/tmp/food-recsys/uploads/`
3. **Background Processing**: ThreadPoolExecutor processes images in parallel
4. **Gemini Extraction**: Each image analyzed by Gemini 1.5 Flash for ingredients
5. **Neo4j Storage**: Dish and Ingredient nodes created with CONTAINS relationships
6. **Cleanup**: Temp files deleted on success, retained on failure for retry

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **FastAPI** | Async support, auto OpenAPI docs, Pydantic validation |
| **ThreadPoolExecutor** | Simple for MVP, easy to swap for Celery later |
| **MERGE queries** | Idempotent operations, no duplicate ingredients |
| **Job-based processing** | Non-blocking uploads, progress tracking |
| **Temp file cleanup** | Delete on success, keep on failure for debugging |
