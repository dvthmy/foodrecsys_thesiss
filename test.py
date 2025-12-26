
from src.services.ingredient_embedder import IngredientEmbedder

embedder = IngredientEmbedder(device='cuda')

query = "peanut"
candidates = [
    # 1. Exact Match (The Baseline)
    ("peanuts", "food ingredient"),
    ("peanut butter", "food ingredient"),
    #2. Strong Semantic Neighbor (Should be close, but lower)
    ("cashew", "food ingredient"),
    # 3. Weak Neighbor (Same category, different specific)
    ("sunflower seeds", "food ingredient"),
    # 4. Unrelated (Control)
    ("steak", "food ingredient"),
    # 5. Orthogonal (Sanity check)
    ("toyota camry", "vehicle") 
]

print(f"Query: '{query}' (Mode: QUERY)")
print("-" * 50)
q_emb = embedder.embed_ingredient(query, mode='query')

for name, desc in candidates:
    # Embed as DOCUMENT (Simulating your database)
    doc_emb = embedder.embed_ingredient(name, mode='document')
    score = embedder.compute_similarity(q_emb, doc_emb)
    
    # Calculate Delta from Baseline
    status = "✅" if score > 0.6 else "❌"
    print(f"{status} Match: {name:<15} | Score: {score:.4f}")