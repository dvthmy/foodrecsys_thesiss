# Food Recommendation System - Documentation

A Food Recommendation System that extracts ingredients from dish images using Gemini AI and stores them in a Neo4j graph database.

## Table of Contents

- [Architecture Overview](./architecture.md)
- [Getting Started](./getting-started.md)
- [API Reference](./api-reference.md)
- [Database Schema](./database-schema.md)
- [Configuration](./configuration.md)
- [Development Guide](./development.md)

## Quick Start

```bash
# 1. Clone and install dependencies
git clone <repository-url>
cd food-recsys
uv sync

# 2. Start Neo4j database
docker compose up -d

# 3. Configure environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# 4. Initialize database constraints
uv run python main.py --init-db

# 5. Run the API server
uv run python main.py
```

API available at http://localhost:8000 with interactive docs at http://localhost:8000/docs

## Project Status

- ✅ Gemini AI ingredient extraction pipeline
- ✅ Neo4j graph database integration
- ✅ FastAPI REST API with batch processing
- 🔲 User management and preferences
- 🔲 Recommendation engine
- 🔲 Streamlit web interface
