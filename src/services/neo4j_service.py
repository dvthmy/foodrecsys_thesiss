"""Neo4j database service for storing dishes and ingredients.

Implements MERGE pattern to avoid duplicate nodes and handles
concurrent writes with retry logic.
"""

import time
from contextlib import contextmanager
from typing import Any, Generator

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import TransientError, ServiceUnavailable

from src.config import config


class Neo4jService:
    """Service for interacting with Neo4j graph database."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        """Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI. Defaults to config value.
            user: Neo4j username. Defaults to config value.
            password: Neo4j password. Defaults to config value.
        """
        self._uri = uri or config.NEO4J_URI
        self._user = user or config.NEO4J_USER
        self._password = password or config.NEO4J_PASSWORD
        self._driver: Driver | None = None

    @property
    def driver(self) -> Driver:
        """Get or create the Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
        return self._driver

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def verify_connectivity(self) -> bool:
        """Verify that the database is reachable.

        Returns:
            True if connection is successful.

        Raises:
            ServiceUnavailable: If the database is not reachable.
        """
        self.driver.verify_connectivity()
        return True

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Create a session context manager.

        Yields:
            Neo4j session instance.
        """
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def create_constraints(self) -> list[str]:
        """Create unique constraints and vector index for all entity types.

        This should be called once during database initialization.

        Returns:
            List of created constraint names.
        """
        constraints = [
            ("dish_id_unique", "CREATE CONSTRAINT dish_id_unique IF NOT EXISTS FOR (d:Dish) REQUIRE d.dish_id IS UNIQUE"),
            ("ingredient_name_unique", "CREATE CONSTRAINT ingredient_name_unique IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE"),
            ("country_name_unique", "CREATE CONSTRAINT country_name_unique IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE"),
            ("user_id_unique", "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE"),
            ("restriction_name_unique", "CREATE CONSTRAINT restriction_name_unique IF NOT EXISTS FOR (r:DietaryRestriction) REQUIRE r.name IS UNIQUE"),
        ]

        created = []
        with self.session() as session:
            for name, query in constraints:
                session.run(query)
                created.append(name)

            # Create vector index for ingredient embeddings (512 dimensions for CLIP)
            vector_index_query = """
            CREATE VECTOR INDEX ingredient_embeddings IF NOT EXISTS
            FOR (i:Ingredient)
            ON i.embedding
            OPTIONS {indexConfig: {
                `vector.dimensions`: 512,
                `vector.similarity_function`: 'cosine'
            }}
            """
            session.run(vector_index_query)
            created.append("ingredient_embeddings_vector_index")

        return created

    def merge_dish_with_ingredients(
        self,
        dish_id: str,
        name: str,
        ingredients: list[str],
        description: str | None = None,
        image_url: str | None = None,
        image_embedding: list[float] | None = None,
        country: str | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Merge a dish with its ingredients into the database.

        Uses MERGE to avoid duplicates. Creates Dish and Ingredient nodes,
        and CONTAINS relationships between them.

        Args:
            dish_id: Unique identifier for the dish.
            name: Name of the dish.
            ingredients: List of ingredient names.
            description: Optional description of the dish.
            image_url: Optional URL/path to dish image.
            image_embedding: Optional CLIP image embedding vector.
            country: Optional country of origin.
            max_retries: Maximum retry attempts for transient errors.

        Returns:
            Dictionary with dish_id and list of ingredients.

        Raises:
            TransientError: If retries are exhausted.
        """
        query = """
        // MERGE the dish using unique identifier
        MERGE (d:Dish {dish_id: $dish_id})
        ON CREATE SET
            d.name = $name,
            d.description = $description,
            d.image_url = $image_url,
            d.image_embedding = $image_embedding,
            d.created_at = datetime()
        ON MATCH SET
            d.updated_at = datetime(),
            d.name = COALESCE($name, d.name),
            d.description = COALESCE($description, d.description),
            d.image_url = COALESCE($image_url, d.image_url),
            d.image_embedding = COALESCE($image_embedding, d.image_embedding)

        // Process each ingredient
        WITH d
        UNWIND $ingredients AS ingredient_name
        MERGE (i:Ingredient {name: toLower(trim(ingredient_name))})
        ON CREATE SET i.created_at = datetime()

        // Create CONTAINS relationship
        MERGE (d)-[r:CONTAINS]->(i)
        ON CREATE SET r.created_at = datetime()

        RETURN d.dish_id AS dish_id, collect(i.name) AS ingredients
        """

        params = {
            "dish_id": dish_id,
            "name": name,
            "description": description,
            "image_url": image_url,
            "image_embedding": image_embedding,
            "ingredients": ingredients,
        }

        for attempt in range(max_retries):
            try:
                with self.session() as session:
                    result = session.execute_write(
                        lambda tx: tx.run(query, **params).single()
                    )

                    # Handle country relationship if provided
                    if country and result:
                        self._merge_country_relationship(session, dish_id, country)

                    return {
                        "dish_id": result["dish_id"] if result else dish_id,
                        "ingredients": list(result["ingredients"]) if result else ingredients,
                    }
            except TransientError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    time.sleep(0.1 * (2**attempt))
                    continue
                raise e

        return {"dish_id": dish_id, "ingredients": ingredients}

    def _merge_country_relationship(
        self,
        session: Session,
        dish_id: str,
        country_name: str,
    ) -> None:
        """Create ORIGINATES_FROM relationship between dish and country.

        Args:
            session: Active Neo4j session.
            dish_id: The dish identifier.
            country_name: Name of the country.
        """
        query = """
        MATCH (d:Dish {dish_id: $dish_id})
        MERGE (c:Country {name: $country_name})
        ON CREATE SET c.created_at = datetime()
        MERGE (d)-[r:ORIGINATES_FROM]->(c)
        ON CREATE SET r.created_at = datetime()
        """
        session.run(query, dish_id=dish_id, country_name=country_name)

    def batch_merge_dishes(
        self,
        dishes: list[dict[str, Any]],
    ) -> list[str]:
        """Batch merge multiple dishes with their ingredients.

        More efficient than individual merges for bulk operations.

        Args:
            dishes: List of dish dictionaries with keys:
                - dish_id: Unique identifier
                - name: Dish name
                - ingredients: List of ingredient names
                - description: Optional description
                - image_url: Optional image URL
                - image_embedding: Optional CLIP embedding vector

        Returns:
            List of successfully merged dish IDs.
        """
        query = """
        UNWIND $dishes AS dish

        // MERGE each dish
        MERGE (d:Dish {dish_id: dish.dish_id})
        ON CREATE SET
            d.name = dish.name,
            d.description = dish.description,
            d.image_url = dish.image_url,
            d.image_embedding = dish.image_embedding,
            d.created_at = datetime()
        ON MATCH SET
            d.updated_at = datetime()

        // Handle ingredients for each dish
        WITH d, dish
        UNWIND dish.ingredients AS ingredient_name
        MERGE (i:Ingredient {name: toLower(trim(ingredient_name))})
        MERGE (d)-[:CONTAINS]->(i)

        RETURN d.dish_id AS dish_id
        """

        with self.session() as session:
            result = session.execute_write(
                lambda tx: tx.run(query, dishes=dishes)
            )
            return [record["dish_id"] for record in result]

    def get_dish_by_id(self, dish_id: str) -> dict[str, Any] | None:
        """Retrieve a dish and its ingredients by ID.

        Args:
            dish_id: The dish identifier.

        Returns:
            Dictionary with dish data and ingredients, or None if not found.
        """
        query = """
        MATCH (d:Dish {dish_id: $dish_id})
        OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
        OPTIONAL MATCH (d)-[:ORIGINATES_FROM]->(c:Country)
        RETURN d.dish_id AS dish_id,
               d.name AS name,
               d.description AS description,
               d.image_url AS image_url,
               collect(DISTINCT i.name) AS ingredients,
               c.name AS country
        """

        with self.session() as session:
            result = session.run(query, dish_id=dish_id).single()
            if result:
                return dict(result)
            return None

    def get_all_ingredients(self) -> list[str]:
        """Retrieve all unique ingredient names.

        Returns:
            List of ingredient names.
        """
        query = "MATCH (i:Ingredient) RETURN i.name AS name ORDER BY i.name"

        with self.session() as session:
            result = session.run(query)
            return [record["name"] for record in result]

    # =========================================================================
    # Ingredient Canonicalization Methods
    # =========================================================================

    def find_similar_ingredients(
        self,
        embedding: list[float],
        k: int = 3,
        threshold: float = 0.0,
        canonical_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Find similar ingredients using vector similarity search.

        Args:
            embedding: The query embedding vector (512 dimensions for CLIP).
            k: Number of nearest neighbors to return.
            threshold: Minimum similarity score (0-1). Results below this are filtered.
            canonical_only: If True, only search canonical ingredients.

        Returns:
            List of dictionaries with 'name' and 'score' keys, ordered by similarity.
        """
        if canonical_only:
            query = """
            CALL db.index.vector.queryNodes('ingredient_embeddings', $k, $embedding)
            YIELD node AS ingredient, score
            WHERE ingredient.is_canonical = true AND score >= $threshold
            RETURN ingredient.name AS name, score
            ORDER BY score DESC
            """
        else:
            query = """
            CALL db.index.vector.queryNodes('ingredient_embeddings', $k, $embedding)
            YIELD node AS ingredient, score
            WHERE score >= $threshold
            RETURN ingredient.name AS name, score
            ORDER BY score DESC
            """

        with self.session() as session:
            result = session.run(query, embedding=embedding, k=k, threshold=threshold)
            return [{"name": record["name"], "score": record["score"]} for record in result]

    def create_canonical_ingredient(
        self,
        name: str,
        embedding: list[float],
    ) -> dict[str, Any]:
        """Create a canonical ingredient with embedding.

        Args:
            name: Ingredient name (will be lowercased and trimmed).
            embedding: CLIP embedding vector (512 dimensions).

        Returns:
            Dictionary with ingredient data.
        """
        query = """
        MERGE (i:Ingredient {name: toLower(trim($name))})
        ON CREATE SET
            i.embedding = $embedding,
            i.is_canonical = true,
            i.created_at = datetime()
        ON MATCH SET
            i.embedding = $embedding,
            i.is_canonical = true,
            i.updated_at = datetime()
        RETURN i.name AS name, i.is_canonical AS is_canonical
        """

        with self.session() as session:
            result = session.run(query, name=name, embedding=embedding).single()
            return dict(result) if result else {"name": name.lower().strip(), "is_canonical": True}

    def create_pending_ingredient(
        self,
        name: str,
        embedding: list[float],
    ) -> dict[str, Any]:
        """Create a pending (non-canonical) ingredient with embedding.

        Args:
            name: Ingredient name (will be lowercased and trimmed).
            embedding: CLIP embedding vector (512 dimensions).

        Returns:
            Dictionary with ingredient data.
        """
        query = """
        MERGE (i:Ingredient {name: toLower(trim($name))})
        ON CREATE SET
            i.embedding = $embedding,
            i.is_canonical = false,
            i.created_at = datetime()
        ON MATCH SET
            i.embedding = COALESCE(i.embedding, $embedding),
            i.updated_at = datetime()
        RETURN i.name AS name, i.is_canonical AS is_canonical
        """

        with self.session() as session:
            result = session.run(query, name=name, embedding=embedding).single()
            return dict(result) if result else {"name": name.lower().strip(), "is_canonical": False}

    def approve_ingredient(self, name: str) -> dict[str, Any] | None:
        """Approve a pending ingredient as canonical.

        Args:
            name: Ingredient name to approve.

        Returns:
            Dictionary with ingredient data, or None if not found.
        """
        query = """
        MATCH (i:Ingredient {name: toLower(trim($name))})
        SET i.is_canonical = true, i.approved_at = datetime()
        RETURN i.name AS name, i.is_canonical AS is_canonical
        """

        with self.session() as session:
            result = session.run(query, name=name).single()
            return dict(result) if result else None

    def reject_ingredient(self, name: str, merge_into: str) -> dict[str, Any] | None:
        """Reject a pending ingredient and merge its relationships into another.

        All dishes that CONTAIN the rejected ingredient will be updated
        to CONTAIN the target ingredient instead. The rejected ingredient
        is then deleted.

        Args:
            name: Ingredient name to reject and delete.
            merge_into: Canonical ingredient to merge relationships into.

        Returns:
            Dictionary with merge result, or None if ingredients not found.
        """
        query = """
        MATCH (rejected:Ingredient {name: toLower(trim($name))})
        MATCH (target:Ingredient {name: toLower(trim($merge_into))})
        
        // Move all CONTAINS relationships from rejected to target
        OPTIONAL MATCH (d:Dish)-[r:CONTAINS]->(rejected)
        WITH rejected, target, collect(d) AS dishes
        
        // Create new relationships to target
        UNWIND dishes AS dish
        MERGE (dish)-[:CONTAINS]->(target)
        
        // Delete old relationships and rejected node
        WITH rejected, target, count(dishes) AS merged_count
        DETACH DELETE rejected
        
        RETURN target.name AS merged_into, merged_count
        """

        with self.session() as session:
            result = session.run(query, name=name, merge_into=merge_into).single()
            return dict(result) if result else None

    def get_pending_ingredients(self) -> list[dict[str, Any]]:
        """Get all pending (non-canonical) ingredients.

        Returns:
            List of pending ingredient dictionaries with name and creation time.
        """
        query = """
        MATCH (i:Ingredient)
        WHERE i.is_canonical = false OR i.is_canonical IS NULL
        OPTIONAL MATCH (d:Dish)-[:CONTAINS]->(i)
        RETURN i.name AS name, 
               i.created_at AS created_at,
               count(d) AS dish_count
        ORDER BY i.created_at DESC
        """

        with self.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def get_ingredient(self, name: str) -> dict[str, Any] | None:
        """Get an ingredient by name.

        Args:
            name: Ingredient name.

        Returns:
            Dictionary with ingredient data, or None if not found.
        """
        query = """
        MATCH (i:Ingredient {name: toLower(trim($name))})
        OPTIONAL MATCH (d:Dish)-[:CONTAINS]->(i)
        RETURN i.name AS name,
               i.is_canonical AS is_canonical,
               i.created_at AS created_at,
               count(d) AS dish_count
        """

        with self.session() as session:
            result = session.run(query, name=name).single()
            return dict(result) if result else None

    def batch_create_canonical_ingredients(
        self,
        ingredients: list[dict[str, Any]],
    ) -> int:
        """Batch create canonical ingredients with embeddings.

        More efficient than individual creates for seeding.

        Args:
            ingredients: List of dicts with 'name' and 'embedding' keys.

        Returns:
            Number of ingredients created/updated.
        """
        query = """
        UNWIND $ingredients AS ing
        MERGE (i:Ingredient {name: toLower(trim(ing.name))})
        ON CREATE SET
            i.embedding = ing.embedding,
            i.is_canonical = true,
            i.created_at = datetime()
        ON MATCH SET
            i.embedding = ing.embedding,
            i.is_canonical = true,
            i.updated_at = datetime()
        RETURN count(i) AS count
        """

        with self.session() as session:
            result = session.run(query, ingredients=ingredients).single()
            return result["count"] if result else 0
