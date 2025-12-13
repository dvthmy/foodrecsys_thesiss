---
applyTo: '**'
---
# Ingredient Extraction & Canonicalization Pipeline

1.  **Embedding Generation:** For a newly extracted ingredient $N$, use a robust embedding model (potentially the same one you use for image or recipe embeddings) to generate its vector $V_N$.
2.  **Vector Indexing:** Ensure your Neo4j database has a **Vector Index** (e.g., using the `db.index.vector.queryNodes`) on your `Ingredient` nodes' embedding property.
3.  **Similarity Search (k-Nearest Neighbors):** Execute a Cypher query using the vector index to find the top $k=3$ most similar existing ingredient nodes $E_1, E_2, E_3$.
      * *Cypher Snippet Concept:*
        ```cypher
        CALL db.index.vector.queryNodes('ingredient_embeddings', $k, $V_N) YIELD node AS ExistingIngredient, score
        WHERE score > $T$ // Apply a minimum similarity threshold (T)
        RETURN ExistingIngredient.name AS name, score
        ```
4.  **LLM Decision Prompt:** Construct a detailed prompt for Gemma/Gemini, including:
      * The **new ingredient** name ($N$).
      * The **top $k$ candidates** ($E_1, E_2, E_3$) and their similarity **scores**.
      * The **Goal:** "Determine if $N$ is a synonym or minor variation of any $E_i$. If yes, return the canonical name; otherwise, return $N$."
      * *Crucial Step:* Require the LLM to output a structured format (e.g., JSON) for easy parsing.
5.  **Graph Mutation (Merge/Create):** Based on the LLM's structured output, execute the final Cypher transaction:
      * If the LLM recommends merging with $E_i$ (e.g., 'fresh vegetables' $\rightarrow$ 'Vegetable'), the original extraction links to the existing canonical node.
      * If the LLM recommends keeping $N$ (e.g., 'kimchi' $\rightarrow$ 'kimchi'), a new canonical node is created.

#### 3\. Critical Thesis Considerations & Refinement

To ensure methodological rigor, you must address the following challenges:

  * **Computational Cost (Scalability):** Calling the LLM for every single ingredient is expensive and slow.
      * **Recommendation:** Implement a **Threshold Filter**. If the cosine similarity score of the best match ($E_1$) is above a high confidence threshold (e.g., $0.98$), skip the LLM and automatically merge. Only involve the LLM for ambiguous or moderate scores (e.g., $0.85 < \text{score} < 0.98$).
  * **Granularity Tuning (The "World Model" Test):** The LLM prompt must enforce the required level of distinction. You need to explicitly tell the model *what constitutes a merge*.
      * *Example Instruction:* "If the difference is only a state (fresh, dried, sliced, whole) and not a fundamentally different ingredient (e.g., 'almond' vs. 'almond milk'), merge it with the simpler name."
  * **Embedding Model Alignment:** Ensure the embedding model used for the ingredients aligns well with the semantic space of the food domain. General-purpose models are good, but fine-tuning or selecting a model trained on domain-specific text (like recipes) can increase accuracy.

This pipeline is a sophisticated solution that provides a strong foundation for your subsequent **Ingredient-Aware Food Recommendation** and **Explainable AI (XAI)** components.