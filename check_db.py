from src.services.neo4j_service import Neo4jService

n = Neo4jService()

# Check dish_id types
query = """
MATCH (d:Dish)
RETURN d.dish_id AS dish_id, d.name AS name, 
       size(keys(d)) AS prop_count
LIMIT 5
"""

with n.driver.session() as s:
    result = s.run(query)
    records = [dict(r) for r in result]
    print(f"Sample dishes:")
    for r in records:
        print(f"  dish_id type: {type(r['dish_id'])}, value: {r['dish_id']}")
        print(f"  name: {r['name']}")
        print()

# Also check popular dishes query result directly
print("\n--- Popular dishes from service ---")
popular = n.get_popular_dishes(limit=3, min_ratings=1)
for p in popular:
    print(f"  dish_id type: {type(p['dish_id'])}, value: {p['dish_id'][:50] if isinstance(p['dish_id'], list) else p['dish_id'][:50]}...")
    print(f"  name: {p['name']}")
