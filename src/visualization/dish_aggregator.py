import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import math

class DishAggregator:
    def __init__(self, method='tfidf', embedding_dim=512):
        self.method = method
        self.idf_map = {}
        
        # For Attention Method (Requires PyTorch training later)
        if method == 'attention':
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=embedding_dim, 
                num_heads=4, 
                batch_first=True
            )
            self.query_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def fit_idf(self, all_recipes: list[list[str]]):
        """
        Pre-compute IDF scores for all ingredients in your dataset.
        Call this ONCE before processing dishes.
        """
        doc_count = len(all_recipes)
        df_counter = Counter()
        
        for ingredients in all_recipes:
            # unique ingredients per document for IDF
            unique_ings = set(ing.lower().strip() for ing in ingredients)
            df_counter.update(unique_ings)
            
        # Calculate IDF: log(N / (df + 1)) + 1
        self.idf_map = {
            ing: math.log((doc_count + 1) / (count + 1)) + 1
            for ing, count in df_counter.items()
        }
        print(f"IDF fitted on {doc_count} recipes. Top-3 rarest: {list(df_counter.most_common()[:-4:-1])}")

    def aggregate(self, ingredient_embeddings: list[list[float]], ingredient_names: list[str]) -> list[float]:
        """
        Aggregates a list of ingredient vectors into a single Dish Vector.
        """
        if not ingredient_embeddings:
            return np.zeros(len(ingredient_embeddings[0])).tolist()
            
        vectors = np.array(ingredient_embeddings)
        
        if self.method == 'mean':
            # Simple average
            return np.mean(vectors, axis=0).tolist()
            
        elif self.method == 'tfidf':
            # Weighted average
            weights = []
            for name in ingredient_names:
                clean_name = name.lower().strip()
                # Default to median IDF if unseen
                weight = self.idf_map.get(clean_name, 1.0) 
                weights.append(weight)
            
            weights = np.array(weights).reshape(-1, 1)
            
            # Weighted Sum
            weighted_sum = np.sum(vectors * weights, axis=0)
            
            # Normalize sum of weights to avoid magnitude explosion
            final_vector = weighted_sum / np.sum(weights)
            
            # L2 Normalize final vector
            return (final_vector / np.linalg.norm(final_vector)).tolist()

        elif self.method == 'attention':
            # Requires PyTorch context
            with torch.no_grad():
                x = torch.tensor(vectors).unsqueeze(0).float() # (1, Seq, Dim)
                
                # Self-Attention to let ingredients talk to each other
                attn_output, _ = self.attention_layer(x, x, x)
                
                # Simple Max Pooling over the sequence
                # (Or use the query_token approach if training)
                dish_vector = torch.max(attn_output, dim=1)[0]
                
                return dish_vector.squeeze().numpy().tolist()

        else:
            raise ValueError(f"Unknown method: {self.method}")


# Singleton instance for reuse
_aggregator: DishAggregator | None = None


def get_aggregator(method: str = 'tfidf', embedding_dim: int = 512) -> DishAggregator:
    """Get or create a singleton DishAggregator instance.

    Args:
        method: Aggregation method ('mean', 'tfidf', or 'attention').
        embedding_dim: Embedding dimension (only used for 'attention' method).

    Returns:
        DishAggregator instance.
    """
    global _aggregator
    if _aggregator is None:
        _aggregator = DishAggregator(method=method, embedding_dim=embedding_dim)
    return _aggregator