# Getting Started

## Prerequisites

- Python 3.13+
- Docker and Docker Compose
- [uv](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd food-recsys
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Start Neo4j Database

```bash
docker compose up -d
```

Wait for Neo4j to be healthy:

```bash
docker compose ps
# Should show "healthy" status
```

Access Neo4j Browser at http://localhost:7474 (credentials: `neo4j` / `food-recsys-password`)

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

Get a Gemini API key from: https://aistudio.google.com/apikey

### 5. Initialize Database

Create required constraints in Neo4j:

```bash
uv run python main.py --init-db
```

Expected output:
```
Initializing Neo4j database...
Connected to Neo4j successfully!
Created 5 constraints:
  - dish_id_unique
  - ingredient_name_unique
  - country_name_unique
  - user_id_unique
  - restriction_name_unique
Database initialization complete!
```

### 6. Run the Server

```bash
uv run python main.py
```

Or with auto-reload for development:

```bash
uv run python main.py --reload
```

The API is now available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Quick Test

Upload and process a dish image:

```bash
curl -X POST "http://localhost:8000/api/v1/dishes/upload-and-process" \
  -F "images=@your-dish-image.jpg"
```

Check job status:

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}/status"
```

## Stopping Services

```bash
# Stop API server
Ctrl+C

# Stop Neo4j
docker compose down

# Stop and remove data
docker compose down -v
```
