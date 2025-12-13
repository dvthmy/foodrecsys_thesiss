"""Food Recommendation System - Main Entry Point.

Provides FastAPI application factory and CLI commands for:
- Running the API server with uvicorn
- Initializing database constraints
"""

import argparse
import logging
import os
import sys

import uvicorn
from fastapi import FastAPI

from src.config import config
from src.api.routes import router
from src.services.neo4j_service import Neo4jService
from src.pipeline.batch_processor import get_processor

# Configure logging to file for background job debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("food-recsys.log"),
    ],
)

# Suppress noisy watchfiles logger (triggers on every log write causing spam)
logging.getLogger("watchfiles").setLevel(logging.WARNING)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Food Recommendation System",
        description="API for extracting ingredients from dish images using Gemini AI",
        version="0.1.0",
    )

    # Include API router
    app.include_router(router)

    # Add root route
    @app.get("/")
    async def index():
        return {
            "name": "Food Recommendation System",
            "version": "0.1.0",
            "description": "API for extracting ingredients from dish images using Gemini AI",
            "docs": "/docs",
            "endpoints": {
                "health": "/api/v1/health",
                "upload": "POST /api/v1/dishes/upload",
                "process": "POST /api/v1/dishes/process",
                "upload_and_process": "POST /api/v1/dishes/upload-and-process",
                "job_status": "GET /api/v1/jobs/{job_id}/status",
                "get_dish": "GET /api/v1/dishes/{dish_id}",
                "get_ingredients": "GET /api/v1/ingredients",
            },
        }

    return app


def init_database() -> None:
    """Initialize Neo4j database with required constraints."""
    print("Initializing Neo4j database...")

    # Validate config
    missing = config.validate()
    if missing:
        print(f"Error: Missing configuration: {', '.join(missing)}")
        sys.exit(1)

    try:
        neo4j = Neo4jService()
        neo4j.verify_connectivity()
        print("Connected to Neo4j successfully!")

        constraints = neo4j.create_constraints()
        print(f"Created {len(constraints)} constraints/indexes:")
        for name in constraints:
            print(f"  - {name}")

        neo4j.close()
        print("Database initialization complete!")

    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)


def seed_ingredients() -> None:
    """Seed the database with canonical ingredients."""
    from src.data.canonical_ingredients import CANONICAL_INGREDIENTS
    from src.services.clip_embedder import get_clip_embedder

    print("Seeding canonical ingredients...")
    print(f"Found {len(CANONICAL_INGREDIENTS)} ingredients to seed")

    try:
        neo4j = Neo4jService()
        neo4j.verify_connectivity()
        print("Connected to Neo4j successfully!")

        clip = get_clip_embedder()
        print("CLIP embedder loaded")

        # Generate embeddings for all ingredients
        ingredients_with_embeddings = []
        for i, name in enumerate(CANONICAL_INGREDIENTS):
            print(f"  [{i+1}/{len(CANONICAL_INGREDIENTS)}] Embedding: {name}")
            embedding = clip.embed_text(name)
            ingredients_with_embeddings.append({
                "name": name,
                "embedding": embedding,
            })

        # Batch insert into Neo4j
        print("\nInserting into Neo4j...")
        count = neo4j.batch_create_canonical_ingredients(ingredients_with_embeddings)
        print(f"Created/updated {count} canonical ingredients")

        neo4j.close()
        print("Seeding complete!")

    except Exception as e:
        print(f"Error seeding ingredients: {e}")
        sys.exit(1)


def seed_dietary_restrictions() -> None:
    """Seed the database with dietary restrictions and their ingredient relationships."""
    from src.data.dietary_restrictions import DIETARY_RESTRICTIONS

    print("Seeding dietary restrictions...")
    print(f"Found {len(DIETARY_RESTRICTIONS)} restrictions to seed")

    try:
        neo4j = Neo4jService()
        neo4j.verify_connectivity()
        print("Connected to Neo4j successfully!")

        # Display what will be seeded
        for restriction in DIETARY_RESTRICTIONS:
            not_suited_count = len(restriction.get("not_suited_ingredients", []))
            suited_count = len(restriction.get("suited_ingredients", []))
            print(f"  - {restriction['name']}: {not_suited_count} not-suited, {suited_count} suited ingredients")

        # Batch create restrictions and relationships
        print("\nCreating restrictions and relationships...")
        result = neo4j.batch_create_dietary_restrictions(DIETARY_RESTRICTIONS)

        print(f"\nResults:")
        print(f"  - Restrictions created/updated: {result['restrictions_created']}")
        print(f"  - NOT_SUITED_FOR relationships: {result['not_suited_relationships']}")
        print(f"  - SUITED_FOR relationships: {result['suited_relationships']}")

        # Warn about missing ingredients
        total_expected = sum(
            len(r.get("not_suited_ingredients", [])) + len(r.get("suited_ingredients", []))
            for r in DIETARY_RESTRICTIONS
        )
        total_created = result["not_suited_relationships"] + result["suited_relationships"]

        if total_created < total_expected:
            missing = total_expected - total_created
            print(f"\n⚠️  Warning: {missing} ingredient(s) not found in database.")
            print("   Run --seed-ingredients first to ensure all ingredients exist.")

        neo4j.close()
        print("\nSeeding complete!")

    except Exception as e:
        print(f"Error seeding dietary restrictions: {e}")
        sys.exit(1)


def run_server() -> None:
    """Run the FastAPI server with uvicorn."""
    # Validate config
    missing = config.validate()
    if missing:
        print(f"Warning: Missing configuration: {', '.join(missing)}")
        print("Some features may not work correctly.")

    # Ensure temp directory exists
    config.ensure_temp_dir()

    uvicorn.run(
        "main:create_app",
        factory=True,
        host=config.APP_HOST,
        port=config.APP_PORT,
        reload=config.APP_DEBUG,
    )


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Food Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run the API server
  python main.py --init-db          # Initialize database constraints
  python main.py --seed-ingredients # Seed canonical ingredients
  python main.py --seed-restrictions # Seed dietary restrictions
  python main.py --host 0.0.0.0     # Run server on specific host
  python main.py --port 8080        # Run server on specific port
        """,
    )

    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize Neo4j database with constraints",
    )
    parser.add_argument(
        "--seed-ingredients",
        action="store_true",
        help="Seed database with canonical ingredients (run --init-db first)",
    )
    parser.add_argument(
        "--seed-restrictions",
        action="store_true",
        help="Seed database with dietary restrictions (run --seed-ingredients first)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Host to bind the server (default: {config.APP_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port to bind the server (default: {config.APP_PORT})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload mode",
    )

    args = parser.parse_args()

    if args.init_db:
        init_database()
    elif args.seed_ingredients:
        seed_ingredients()
    elif args.seed_restrictions:
        seed_dietary_restrictions()
    else:
        # Override config with CLI args
        if args.host:
            config.APP_HOST = args.host
        if args.port:
            config.APP_PORT = args.port
        if args.reload:
            config.APP_DEBUG = True

        run_server()


if __name__ == "__main__":
    main()
