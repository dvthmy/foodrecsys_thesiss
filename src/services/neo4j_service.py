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

    # =========================================================================
    # Dietary Restriction Methods
    # =========================================================================

    def create_dietary_restriction(
        self,
        name: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create a dietary restriction node.

        Args:
            name: Restriction name (e.g., "vegetarian", "dairy-allergy").
            description: Optional description of the restriction.

        Returns:
            Dictionary with restriction data.
        """
        query = """
        MERGE (r:DietaryRestriction {name: toLower(trim($name))})
        ON CREATE SET
            r.description = $description,
            r.created_at = datetime()
        ON MATCH SET
            r.description = COALESCE($description, r.description),
            r.updated_at = datetime()
        RETURN r.name AS name, r.description AS description
        """

        with self.session() as session:
            result = session.run(query, name=name, description=description).single()
            return dict(result) if result else {"name": name.lower().strip(), "description": description}

    def create_ingredient_restriction_relationship(
        self,
        ingredient_name: str,
        restriction_name: str,
        relationship_type: str,
    ) -> dict[str, Any] | None:
        """Create a relationship between an ingredient and dietary restriction.

        Args:
            ingredient_name: Name of the ingredient.
            restriction_name: Name of the dietary restriction.
            relationship_type: Either "SUITED_FOR" or "NOT_SUITED_FOR".

        Returns:
            Dictionary with relationship info, or None if nodes not found.

        Raises:
            ValueError: If relationship_type is invalid.
        """
        if relationship_type not in ("SUITED_FOR", "NOT_SUITED_FOR"):
            raise ValueError(f"Invalid relationship_type: {relationship_type}. Must be 'SUITED_FOR' or 'NOT_SUITED_FOR'")

        # Use APOC or dynamic relationship - here we use conditional queries
        if relationship_type == "SUITED_FOR":
            query = """
            MATCH (i:Ingredient {name: toLower(trim($ingredient_name))})
            MATCH (r:DietaryRestriction {name: toLower(trim($restriction_name))})
            MERGE (i)-[rel:SUITED_FOR]->(r)
            ON CREATE SET rel.created_at = datetime()
            RETURN i.name AS ingredient, r.name AS restriction, type(rel) AS relationship
            """
        else:
            query = """
            MATCH (i:Ingredient {name: toLower(trim($ingredient_name))})
            MATCH (r:DietaryRestriction {name: toLower(trim($restriction_name))})
            MERGE (i)-[rel:NOT_SUITED_FOR]->(r)
            ON CREATE SET rel.created_at = datetime()
            RETURN i.name AS ingredient, r.name AS restriction, type(rel) AS relationship
            """

        with self.session() as session:
            result = session.run(
                query,
                ingredient_name=ingredient_name,
                restriction_name=restriction_name,
            ).single()
            return dict(result) if result else None

    def batch_create_dietary_restrictions(
        self,
        restrictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Batch create dietary restrictions with ingredient relationships.

        Creates DietaryRestriction nodes and their SUITED_FOR/NOT_SUITED_FOR
        relationships to existing ingredients.

        Args:
            restrictions: List of dicts with keys:
                - name: Restriction name
                - description: Optional description
                - suited_ingredients: List of ingredient names suited for this restriction
                - not_suited_ingredients: List of ingredient names not suited

        Returns:
            Dictionary with counts of created restrictions and relationships.
        """
        # First, create all restriction nodes
        create_restrictions_query = """
        UNWIND $restrictions AS r
        MERGE (dr:DietaryRestriction {name: toLower(trim(r.name))})
        ON CREATE SET
            dr.description = r.description,
            dr.created_at = datetime()
        ON MATCH SET
            dr.description = COALESCE(r.description, dr.description),
            dr.updated_at = datetime()
        RETURN count(dr) AS count
        """

        # Create NOT_SUITED_FOR relationships
        create_not_suited_query = """
        UNWIND $mappings AS m
        MATCH (i:Ingredient {name: toLower(trim(m.ingredient))})
        MATCH (r:DietaryRestriction {name: toLower(trim(m.restriction))})
        MERGE (i)-[rel:NOT_SUITED_FOR]->(r)
        ON CREATE SET rel.created_at = datetime()
        RETURN count(rel) AS count
        """

        # Create SUITED_FOR relationships
        create_suited_query = """
        UNWIND $mappings AS m
        MATCH (i:Ingredient {name: toLower(trim(m.ingredient))})
        MATCH (r:DietaryRestriction {name: toLower(trim(m.restriction))})
        MERGE (i)-[rel:SUITED_FOR]->(r)
        ON CREATE SET rel.created_at = datetime()
        RETURN count(rel) AS count
        """

        # Build mapping lists
        not_suited_mappings = []
        suited_mappings = []

        for restriction in restrictions:
            restriction_name = restriction["name"]
            for ing in restriction.get("not_suited_ingredients", []):
                not_suited_mappings.append({
                    "ingredient": ing,
                    "restriction": restriction_name,
                })
            for ing in restriction.get("suited_ingredients", []):
                suited_mappings.append({
                    "ingredient": ing,
                    "restriction": restriction_name,
                })

        with self.session() as session:
            # Create restrictions
            result = session.run(create_restrictions_query, restrictions=restrictions).single()
            restrictions_count = result["count"] if result else 0

            # Create NOT_SUITED_FOR relationships
            result = session.run(create_not_suited_query, mappings=not_suited_mappings).single()
            not_suited_count = result["count"] if result else 0

            # Create SUITED_FOR relationships
            result = session.run(create_suited_query, mappings=suited_mappings).single()
            suited_count = result["count"] if result else 0

        return {
            "restrictions_created": restrictions_count,
            "not_suited_relationships": not_suited_count,
            "suited_relationships": suited_count,
        }

    def get_restrictions_for_ingredient(self, ingredient_name: str) -> dict[str, list[str]]:
        """Get dietary restrictions associated with an ingredient.

        Args:
            ingredient_name: Name of the ingredient.

        Returns:
            Dictionary with 'suited_for' and 'not_suited_for' lists of restriction names.
        """
        query = """
        MATCH (i:Ingredient {name: toLower(trim($name))})
        OPTIONAL MATCH (i)-[:SUITED_FOR]->(suited:DietaryRestriction)
        OPTIONAL MATCH (i)-[:NOT_SUITED_FOR]->(not_suited:DietaryRestriction)
        RETURN collect(DISTINCT suited.name) AS suited_for,
               collect(DISTINCT not_suited.name) AS not_suited_for
        """

        with self.session() as session:
            result = session.run(query, name=ingredient_name).single()
            if result:
                return {
                    "suited_for": [r for r in result["suited_for"] if r],
                    "not_suited_for": [r for r in result["not_suited_for"] if r],
                }
            return {"suited_for": [], "not_suited_for": []}

    def get_ingredients_for_restriction(self, restriction_name: str) -> dict[str, list[str]]:
        """Get ingredients associated with a dietary restriction.

        Args:
            restriction_name: Name of the dietary restriction.

        Returns:
            Dictionary with 'suited' and 'not_suited' lists of ingredient names.
        """
        query = """
        MATCH (r:DietaryRestriction {name: toLower(trim($name))})
        OPTIONAL MATCH (suited:Ingredient)-[:SUITED_FOR]->(r)
        OPTIONAL MATCH (not_suited:Ingredient)-[:NOT_SUITED_FOR]->(r)
        RETURN collect(DISTINCT suited.name) AS suited,
               collect(DISTINCT not_suited.name) AS not_suited
        """

        with self.session() as session:
            result = session.run(query, name=restriction_name).single()
            if result:
                return {
                    "suited": [i for i in result["suited"] if i],
                    "not_suited": [i for i in result["not_suited"] if i],
                }
            return {"suited": [], "not_suited": []}

    def get_all_dietary_restrictions(self) -> list[dict[str, Any]]:
        """Retrieve all dietary restrictions with their ingredient counts.

        Returns:
            List of restriction dictionaries with name, description, and counts.
        """
        query = """
        MATCH (r:DietaryRestriction)
        OPTIONAL MATCH (suited:Ingredient)-[:SUITED_FOR]->(r)
        OPTIONAL MATCH (not_suited:Ingredient)-[:NOT_SUITED_FOR]->(r)
        RETURN r.name AS name,
               r.description AS description,
               count(DISTINCT suited) AS suited_count,
               count(DISTINCT not_suited) AS not_suited_count
        ORDER BY r.name
        """

        with self.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def get_dietary_restriction(self, name: str) -> dict[str, Any] | None:
        """Get a dietary restriction by name with full details.

        Args:
            name: Restriction name.

        Returns:
            Dictionary with restriction data, or None if not found.
        """
        query = """
        MATCH (r:DietaryRestriction {name: toLower(trim($name))})
        OPTIONAL MATCH (suited:Ingredient)-[:SUITED_FOR]->(r)
        OPTIONAL MATCH (not_suited:Ingredient)-[:NOT_SUITED_FOR]->(r)
        RETURN r.name AS name,
               r.description AS description,
               r.created_at AS created_at,
               collect(DISTINCT suited.name) AS suited_ingredients,
               collect(DISTINCT not_suited.name) AS not_suited_ingredients
        """

        with self.session() as session:
            result = session.run(query, name=name).single()
            if result:
                data = dict(result)
                # Filter out None values from collections
                data["suited_ingredients"] = [i for i in data["suited_ingredients"] if i]
                data["not_suited_ingredients"] = [i for i in data["not_suited_ingredients"] if i]
                return data
            return None

    # =========================================================================
    # Dish Similarity Visualization Methods
    # =========================================================================

    def get_dishes_summary(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get a summary of all dishes for selection.

        Args:
            limit: Maximum number of dishes to return.

        Returns:
            List of dictionaries with dish_id and name.
        """
        query = """
        MATCH (d:Dish)
        WHERE d.image_embedding IS NOT NULL
        RETURN d.dish_id AS dish_id, d.name AS name
        ORDER BY d.name
        LIMIT $limit
        """

        with self.session() as session:
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]

    def get_dishes_with_embeddings_and_ingredients(
        self,
        dish_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Get dishes with their image embeddings, ingredients, and ingredient embeddings.

        Filters out dishes where image_embedding is null.

        Args:
            dish_ids: List of dish IDs to retrieve.

        Returns:
            List of dictionaries with dish data, embeddings, and ingredients.
        """
        query = """
        MATCH (d:Dish)
        WHERE d.dish_id IN $dish_ids AND d.image_embedding IS NOT NULL
        OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
        WITH d, 
             collect(DISTINCT i.name) AS ingredient_names,
             collect(DISTINCT i.embedding) AS ingredient_embeddings
        RETURN d.dish_id AS dish_id,
               d.name AS name,
               d.image_embedding AS image_embedding,
               ingredient_names AS ingredients,
               [emb IN ingredient_embeddings WHERE emb IS NOT NULL] AS ingredient_embeddings
        ORDER BY d.name
        """

        with self.session() as session:
            result = session.run(query, dish_ids=dish_ids)
            return [dict(record) for record in result]

    def get_all_dishes_with_embeddings(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get all dishes with their embeddings and ingredients.

        Args:
            limit: Maximum number of dishes to return.

        Returns:
            List of dictionaries with dish data, embeddings, and ingredients.
        """
        query = """
        MATCH (d:Dish)
        WHERE d.image_embedding IS NOT NULL
        OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
        WITH d, 
             collect(DISTINCT i.name) AS ingredient_names,
             collect(DISTINCT i.embedding) AS ingredient_embeddings
        RETURN d.dish_id AS dish_id,
               d.name AS name,
               d.image_embedding AS image_embedding,
               ingredient_names AS ingredients,
               [emb IN ingredient_embeddings WHERE emb IS NOT NULL] AS ingredient_embeddings
        ORDER BY d.name
        LIMIT $limit
        """

        with self.session() as session:
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]

    # =========================================================================
    # User Methods
    # =========================================================================

    def merge_user_by_name(
        self,
        name: str,
        age: int | None = None,
        gender: int | None = None,
        nationality: str | None = None,
    ) -> dict[str, Any]:
        """Merge a user by name, generating UUID only on create.

        Uses normalized name for matching to ensure idempotency.

        Args:
            name: User's full name (used as unique identifier).
            age: User's age.
            gender: User's gender (0 = female, 1 = male).
            nationality: User's nationality/country.

        Returns:
            Dictionary with user data including user_id.
        """
        import uuid as uuid_module

        query = """
        MERGE (u:User {normalized_name: toLower(trim($name))})
        ON CREATE SET
            u.user_id = $user_id,
            u.name = $name,
            u.age = $age,
            u.gender = $gender,
            u.nationality = $nationality,
            u.created_at = datetime()
        ON MATCH SET
            u.name = COALESCE($name, u.name),
            u.age = COALESCE($age, u.age),
            u.gender = COALESCE($gender, u.gender),
            u.nationality = COALESCE($nationality, u.nationality),
            u.updated_at = datetime()
        RETURN u.user_id AS user_id,
               u.name AS name,
               u.age AS age,
               u.gender AS gender,
               u.nationality AS nationality,
               u.created_at IS NOT NULL AND u.updated_at IS NULL AS is_new
        """

        params = {
            "name": name,
            "user_id": str(uuid_module.uuid4()),
            "age": age,
            "gender": gender,
            "nationality": nationality,
        }

        with self.session() as session:
            result = session.run(query, **params).single()
            if result:
                return dict(result)
            return {"user_id": params["user_id"], "name": name, "is_new": True}

    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Retrieve a user by ID with their dietary restrictions and ratings.

        Args:
            user_id: The user identifier.

        Returns:
            Dictionary with user data, restrictions, and ratings, or None if not found.
        """
        query = """
        MATCH (u:User {user_id: $user_id})
        OPTIONAL MATCH (u)-[:HAS_RESTRICTION]->(r:DietaryRestriction)
        OPTIONAL MATCH (u)-[rated:RATED]->(d:Dish)
        RETURN u.user_id AS user_id,
               u.name AS name,
               u.age AS age,
               u.gender AS gender,
               u.nationality AS nationality,
               collect(DISTINCT r.name) AS dietary_restrictions,
               collect(DISTINCT {dish_name: d.name, dish_id: d.dish_id, score: rated.score}) AS ratings
        """

        with self.session() as session:
            result = session.run(query, user_id=user_id).single()
            if result:
                data = dict(result)
                # Filter out None values from collections
                data["dietary_restrictions"] = [r for r in data["dietary_restrictions"] if r]
                data["ratings"] = [r for r in data["ratings"] if r.get("dish_name")]
                return data
            return None

    def get_user_by_name(self, name: str) -> dict[str, Any] | None:
        """Retrieve a user by name.

        Args:
            name: The user's name.

        Returns:
            Dictionary with user data, or None if not found.
        """
        query = """
        MATCH (u:User {normalized_name: toLower(trim($name))})
        OPTIONAL MATCH (u)-[:HAS_RESTRICTION]->(r:DietaryRestriction)
        OPTIONAL MATCH (u)-[rated:RATED]->(d:Dish)
        RETURN u.user_id AS user_id,
               u.name AS name,
               u.age AS age,
               u.gender AS gender,
               u.nationality AS nationality,
               collect(DISTINCT r.name) AS dietary_restrictions,
               collect(DISTINCT {dish_name: d.name, dish_id: d.dish_id, score: rated.score}) AS ratings
        """

        with self.session() as session:
            result = session.run(query, name=name).single()
            if result:
                data = dict(result)
                data["dietary_restrictions"] = [r for r in data["dietary_restrictions"] if r]
                data["ratings"] = [r for r in data["ratings"] if r.get("dish_name")]
                return data
            return None

    def get_all_users(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve all users with their dietary restrictions.

        Args:
            limit: Maximum number of users to return.

        Returns:
            List of user dictionaries.
        """
        query = """
        MATCH (u:User)
        OPTIONAL MATCH (u)-[:HAS_RESTRICTION]->(r:DietaryRestriction)
        OPTIONAL MATCH (u)-[rated:RATED]->(d:Dish)
        WITH u, 
             collect(DISTINCT r.name) AS restrictions,
             count(DISTINCT d) AS rating_count
        RETURN u.user_id AS user_id,
               u.name AS name,
               u.age AS age,
               u.gender AS gender,
               u.nationality AS nationality,
               restrictions AS dietary_restrictions,
               rating_count
        ORDER BY u.name
        LIMIT $limit
        """

        with self.session() as session:
            result = session.run(query, limit=limit)
            users = []
            for record in result:
                data = dict(record)
                data["dietary_restrictions"] = [r for r in data["dietary_restrictions"] if r]
                users.append(data)
            return users

    def create_user_restriction_relationship(
        self,
        user_id: str,
        restriction_name: str,
    ) -> dict[str, Any] | None:
        """Create HAS_RESTRICTION relationship between user and dietary restriction.

        Args:
            user_id: The user identifier.
            restriction_name: Name of the dietary restriction.

        Returns:
            Dictionary with relationship info, or None if nodes not found.
        """
        query = """
        MATCH (u:User {user_id: $user_id})
        MATCH (r:DietaryRestriction {name: toLower(trim($restriction_name))})
        MERGE (u)-[rel:HAS_RESTRICTION]->(r)
        ON CREATE SET rel.created_at = datetime()
        RETURN u.user_id AS user_id, r.name AS restriction, type(rel) AS relationship
        """

        with self.session() as session:
            result = session.run(
                query,
                user_id=user_id,
                restriction_name=restriction_name,
            ).single()
            return dict(result) if result else None

    def create_user_rating(
        self,
        user_id: str,
        dish_name: str,
        score: int,
    ) -> dict[str, Any] | None:
        """Create RATED relationship between user and dish.

        Args:
            user_id: The user identifier.
            dish_name: Name of the dish (exact match).
            score: Rating score (1-5).

        Returns:
            Dictionary with relationship info, or None if nodes not found.
        """
        query = """
        MATCH (u:User {user_id: $user_id})
        MATCH (d:Dish {name: $dish_name})
        MERGE (u)-[rated:RATED]->(d)
        ON CREATE SET rated.score = $score, rated.created_at = datetime()
        ON MATCH SET rated.score = $score, rated.updated_at = datetime()
        RETURN u.user_id AS user_id, d.name AS dish_name, d.dish_id AS dish_id, rated.score AS score
        """

        with self.session() as session:
            result = session.run(
                query,
                user_id=user_id,
                dish_name=dish_name,
                score=score,
            ).single()
            return dict(result) if result else None

    def get_dish_by_name(self, name: str) -> dict[str, Any] | None:
        """Get a dish by exact name match.

        Args:
            name: Dish name (exact match).

        Returns:
            Dictionary with dish data, or None if not found.
        """
        query = """
        MATCH (d:Dish {name: $name})
        RETURN d.dish_id AS dish_id, d.name AS name
        """

        with self.session() as session:
            result = session.run(query, name=name).single()
            return dict(result) if result else None

    def get_restriction_by_name(self, name: str) -> dict[str, Any] | None:
        """Get a dietary restriction by exact name match.

        Args:
            name: Restriction name (will be normalized).

        Returns:
            Dictionary with restriction data, or None if not found.
        """
        query = """
        MATCH (r:DietaryRestriction {name: toLower(trim($name))})
        RETURN r.name AS name, r.description AS description
        """

        with self.session() as session:
            result = session.run(query, name=name).single()
            return dict(result) if result else None

    # =========================================================================
    # Collaborative Filtering Methods
    # =========================================================================

    def get_all_ratings(self) -> list[dict[str, Any]]:
        """Get all user-dish ratings for building rating matrix.

        Returns:
            List of dictionaries with user_id, dish_id, dish_name, and score.
        """
        query = """
        MATCH (u:User)-[r:RATED]->(d:Dish)
        RETURN u.user_id AS user_id,
               d.dish_id AS dish_id,
               d.name AS dish_name,
               r.score AS score
        """

        with self.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def get_user_ratings(self, user_id: str) -> list[dict[str, Any]]:
        """Get all ratings for a specific user.

        Args:
            user_id: The user identifier.

        Returns:
            List of dictionaries with dish_id, dish_name, and score.
        """
        query = """
        MATCH (u:User {user_id: $user_id})-[r:RATED]->(d:Dish)
        RETURN d.dish_id AS dish_id,
               d.name AS dish_name,
               r.score AS score
        """

        with self.session() as session:
            result = session.run(query, user_id=user_id)
            return [dict(record) for record in result]

    def get_unrated_dishes_for_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get dishes that a user hasn't rated yet.

        Args:
            user_id: The user identifier.
            limit: Maximum number of dishes to return.

        Returns:
            List of dish dictionaries.
        """
        query = """
        MATCH (d:Dish)
        WHERE NOT EXISTS((u:User {user_id: $user_id})-[:RATED]->(d))
        OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
        WITH d, collect(DISTINCT i.name) AS ingredients
        RETURN d.dish_id AS dish_id,
               d.name AS name,
               d.description AS description,
               ingredients
        LIMIT $limit
        """

        with self.session() as session:
            result = session.run(query, user_id=user_id, limit=limit)
            return [dict(record) for record in result]

    def get_popular_dishes(
        self,
        limit: int = 10,
        min_ratings: int = 1,
    ) -> list[dict[str, Any]]:
        """Get most popular dishes based on average rating and rating count.

        Args:
            limit: Maximum number of dishes to return.
            min_ratings: Minimum number of ratings required.

        Returns:
            List of dish dictionaries with avg_rating and rating_count.
        """
        query = """
        MATCH (d:Dish)<-[r:RATED]-(:User)
        WITH d, avg(r.score) AS avg_rating, count(r) AS rating_count
        WHERE rating_count >= $min_ratings
        OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
        WITH d, avg_rating, rating_count, collect(DISTINCT i.name) AS ingredients
        RETURN d.dish_id AS dish_id,
               d.name AS name,
               d.description AS description,
               ingredients,
               avg_rating,
               rating_count
        ORDER BY avg_rating DESC, rating_count DESC
        LIMIT $limit
        """

        with self.session() as session:
            result = session.run(query, limit=limit, min_ratings=min_ratings)
            return [dict(record) for record in result]

    def get_popular_dishes_with_restriction_filter(
        self,
        restriction_names: list[str],
        limit: int = 10,
        min_ratings: int = 1,
    ) -> list[dict[str, Any]]:
        """Get popular dishes excluding those with ingredients not suited for restrictions.

        Args:
            restriction_names: List of dietary restriction names to filter by.
            limit: Maximum number of dishes to return.
            min_ratings: Minimum number of ratings required.

        Returns:
            List of dish dictionaries safe for the given restrictions.
        """
        if not restriction_names:
            return self.get_popular_dishes(limit=limit, min_ratings=min_ratings)

        query = """
        // Get ingredients that are NOT suited for the user's restrictions
        MATCH (bad_ing:Ingredient)-[:NOT_SUITED_FOR]->(r:DietaryRestriction)
        WHERE r.name IN $restriction_names
        WITH collect(DISTINCT bad_ing.name) AS bad_ingredients

        // Find popular dishes that don't contain bad ingredients
        MATCH (d:Dish)<-[rated:RATED]-(:User)
        WITH d, avg(rated.score) AS avg_rating, count(rated) AS rating_count, bad_ingredients
        WHERE rating_count >= $min_ratings

        // Check dish ingredients
        OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
        WITH d, avg_rating, rating_count, 
             collect(DISTINCT i.name) AS ingredients,
             bad_ingredients

        // Filter out dishes with bad ingredients
        WHERE NONE(ing IN ingredients WHERE ing IN bad_ingredients)

        RETURN d.dish_id AS dish_id,
               d.name AS name,
               d.description AS description,
               ingredients,
               avg_rating,
               rating_count
        ORDER BY avg_rating DESC, rating_count DESC
        LIMIT $limit
        """

        with self.session() as session:
            result = session.run(
                query,
                restriction_names=[r.lower().strip() for r in restriction_names],
                limit=limit,
                min_ratings=min_ratings,
            )
            return [dict(record) for record in result]

    def get_dishes_rated_by_users(
        self,
        user_ids: list[str],
        exclude_user_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get dishes rated by specific users, excluding dishes already rated by target user.

        Args:
            user_ids: List of user IDs whose ratings to consider.
            exclude_user_id: User ID to exclude (target user).
            limit: Maximum number of dishes to return.

        Returns:
            List of dish dictionaries with aggregated ratings from similar users.
        """
        query = """
        // Get dishes rated by target user to exclude
        MATCH (target:User {user_id: $exclude_user_id})
        OPTIONAL MATCH (target)-[:RATED]->(rated_dish:Dish)
        WITH collect(rated_dish.dish_id) AS already_rated

        // Get dishes rated by similar users
        MATCH (u:User)-[r:RATED]->(d:Dish)
        WHERE u.user_id IN $user_ids 
          AND NOT d.dish_id IN already_rated
        WITH d, 
             avg(r.score) AS avg_score,
             count(r) AS rater_count,
             collect({user_id: u.user_id, score: r.score}) AS ratings

        OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
        WITH d, avg_score, rater_count, ratings, collect(DISTINCT i.name) AS ingredients

        RETURN d.dish_id AS dish_id,
               d.name AS name,
               d.description AS description,
               ingredients,
               avg_score,
               rater_count,
               ratings
        ORDER BY avg_score DESC, rater_count DESC
        LIMIT $limit
        """

        with self.session() as session:
            result = session.run(
                query,
                user_ids=user_ids,
                exclude_user_id=exclude_user_id,
                limit=limit,
            )
            return [dict(record) for record in result]

    def get_user_dietary_restrictions(self, user_id: str) -> list[str]:
        """Get dietary restrictions for a user.

        Args:
            user_id: The user identifier.

        Returns:
            List of restriction names.
        """
        query = """
        MATCH (u:User {user_id: $user_id})-[:HAS_RESTRICTION]->(r:DietaryRestriction)
        RETURN collect(r.name) AS restrictions
        """

        with self.session() as session:
            result = session.run(query, user_id=user_id).single()
            if result:
                return result["restrictions"] or []
            return []

    def filter_dishes_by_restrictions(
        self,
        dish_ids: list[str],
        restriction_names: list[str],
    ) -> list[str]:
        """Filter dish IDs to exclude those not suited for dietary restrictions.

        Args:
            dish_ids: List of dish IDs to filter.
            restriction_names: List of dietary restriction names.

        Returns:
            Filtered list of dish IDs that are safe for the restrictions.
        """
        if not restriction_names or not dish_ids:
            return dish_ids

        query = """
        // Get ingredients that are NOT suited for the restrictions
        MATCH (bad_ing:Ingredient)-[:NOT_SUITED_FOR]->(r:DietaryRestriction)
        WHERE r.name IN $restriction_names
        WITH collect(DISTINCT bad_ing.name) AS bad_ingredients

        // Check each dish
        UNWIND $dish_ids AS did
        MATCH (d:Dish {dish_id: did})
        OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
        WITH d, collect(DISTINCT i.name) AS ingredients, bad_ingredients
        WHERE NONE(ing IN ingredients WHERE ing IN bad_ingredients)
        RETURN d.dish_id AS dish_id
        """

        with self.session() as session:
            result = session.run(
                query,
                dish_ids=dish_ids,
                restriction_names=[r.lower().strip() for r in restriction_names],
            )
            return [record["dish_id"] for record in result]
