# Development Guide

## Development Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- Docker and Docker Compose
- Git

### Initial Setup

```bash
# Clone repository
git clone <repository-url>
cd food-recsys

# Install dependencies
uv sync

# Start Neo4j
docker compose up -d

# Configure environment
cp .env.example .env
# Add your GEMINI_API_KEY

# Initialize database
uv run python main.py --init-db

# Run with auto-reload
uv run python main.py --reload
```

## Project Structure

```
food-recsys/
├── main.py                 # Application entry point
├── pyproject.toml          # Dependencies and project config
├── docker-compose.yml      # Neo4j database setup
├── .env.example            # Environment template
├── docs/                   # Documentation
└── src/
    ├── config.py           # Configuration management
    ├── api/
    │   └── routes.py       # FastAPI endpoints
    ├── services/
    │   ├── gemini_extractor.py   # Gemini AI integration
    │   └── neo4j_service.py      # Database operations
    └── pipeline/
        └── batch_processor.py    # Background processing
```

## Adding Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Sync after manual pyproject.toml edit
uv sync
```

## Code Style

The project follows Python best practices:

- Type hints for all function signatures
- Docstrings for modules, classes, and public functions
- Async/await for I/O-bound operations in FastAPI
- Pydantic models for request/response validation

## Testing

### Manual Testing with cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Upload and process
curl -X POST http://localhost:8000/api/v1/dishes/upload-and-process \
  -F "images=@test-image.jpg"

# Check job status
curl http://localhost:8000/api/v1/jobs/{job_id}/status

# Get all ingredients
curl http://localhost:8000/api/v1/ingredients
```

### Interactive Testing

Use the Swagger UI at http://localhost:8000/docs for interactive API testing.

### Neo4j Queries

Access Neo4j Browser at http://localhost:7474 to run Cypher queries:

```cypher
-- View all dishes with ingredients
MATCH (d:Dish)-[:CONTAINS]->(i:Ingredient)
RETURN d.name, collect(i.name) as ingredients

-- Count nodes by label
MATCH (n) RETURN labels(n)[0] as label, count(*) as count
```

## Extending the System

### Adding a New Endpoint

1. Define Pydantic models in `src/api/routes.py`:
   ```python
   class NewRequest(BaseModel):
       field: str
   
   class NewResponse(BaseModel):
       result: str
   ```

2. Add the route:
   ```python
   @router.post("/new-endpoint", response_model=NewResponse)
   async def new_endpoint(request: NewRequest) -> NewResponse:
       # Implementation
       return NewResponse(result="success")
   ```

### Adding a New Service

1. Create `src/services/new_service.py`:
   ```python
   class NewService:
       def __init__(self):
           pass
       
       def do_something(self) -> str:
           return "result"
   ```

2. Export in `src/services/__init__.py`:
   ```python
   from src.services.new_service import NewService
   __all__ = [..., "NewService"]
   ```

### Adding Database Relationships

1. Add the relationship query in `neo4j_service.py`:
   ```python
   def create_new_relationship(self, ...):
       query = """
       MATCH (a:NodeA {id: $a_id})
       MATCH (b:NodeB {id: $b_id})
       MERGE (a)-[r:NEW_REL]->(b)
       RETURN a, r, b
       """
       # Execute query
   ```

2. If new node types, add constraints in `create_constraints()`.

## Debugging

### View Application Logs

```bash
# Run with debug mode
APP_DEBUG=1 uv run python main.py
```

### View Neo4j Logs

```bash
docker compose logs -f neo4j
```

### Check Failed Jobs

Failed processing jobs retain their temp files for debugging:
- Location: `/tmp/food-recsys/uploads/`
- Check job status for error messages

### Common Issues

**"GEMINI_API_KEY is required"**
- Ensure `.env` file exists with valid API key

**"ServiceUnavailable: Unable to retrieve routing information"**
- Neo4j not running: `docker compose up -d`
- Check Neo4j health: `docker compose ps`

**"No valid files uploaded"**
- Check file extensions (allowed: png, jpg, jpeg, webp, gif)
- Check file size (max 16MB)

## Future Development

### Planned Features

1. **User Management**
   - User registration and authentication
   - User preferences and dietary restrictions

2. **Recommendation Engine**
   - Collaborative filtering based on user ratings
   - Content-based filtering using ingredients

3. **Web Interface**
   - Streamlit dashboard for recommendations
   - Image upload and preference management

### Scaling Considerations

- Replace ThreadPoolExecutor with Celery for distributed processing
- Add Redis for job queue and caching
- Consider Neo4j Aura for managed database
- Add rate limiting for Gemini API calls
