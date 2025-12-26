# Cách Tính Similarity Score để So Sánh với Threshold

## Tóm tắt Flow

```
Input: "tomato" (ingredient name)
  ↓
1. Tạo Query Embedding (512 dimensions vector)
   embed_ingredient("tomato", mode="query")
   → [0.123, -0.456, 0.789, ...] (512 số)
  ↓
2. Tìm Similar Ingredients trong Neo4j
   find_similar_ingredients(embedding, k=3, canonical_only=True)
   → Neo4j tính cosine similarity với tất cả canonical ingredients
  ↓
3. Neo4j Trả Về Scores
   [
     {"name": "tomatoes", "score": 0.95},
     {"name": "cherry tomato", "score": 0.87},
     {"name": "tomato paste", "score": 0.72}
   ]
  ↓
4. So Sánh với Threshold
   best_score = 0.95
   - best_score >= 0.8 (high_threshold) → Auto-merge
   - 0.6 <= best_score < 0.8 → Pending (medium)
   - best_score < 0.6 → Pending (low)
```

## Chi Tiết Cách Tính Score

### 1. Embedding Generation
- Model: EmbeddingGemma (512 dimensions)
- Query mode: `encode_query()` - tối ưu cho search
- Document mode: `encode()` với template - tối ưu cho storage

### 2. Cosine Similarity Calculation (Neo4j tự động tính)

**Công thức:**
```python
cosine_similarity(a, b) = dot_product(a, b) / (norm(a) * norm(b))

# Ví dụ:
query_embedding = [0.5, 0.3, 0.8]
candidate_embedding = [0.4, 0.2, 0.9]

# Dot product
dot = (0.5 * 0.4) + (0.3 * 0.2) + (0.8 * 0.9)
     = 0.2 + 0.06 + 0.72 = 0.98

# Norms
norm_query = sqrt(0.5² + 0.3² + 0.8²) = sqrt(0.98) ≈ 0.99
norm_candidate = sqrt(0.4² + 0.2² + 0.9²) = sqrt(1.01) ≈ 1.00

# Cosine similarity
score = 0.98 / (0.99 * 1.00) ≈ 0.99
```

**Đặc điểm:**
- Score nằm trong khoảng **-1 đến 1**
- Với embeddings đã normalize (như EmbeddingGemma): **0 đến 1**
- **1.0** = hoàn toàn giống nhau
- **0.0** = không liên quan
- **> 0.8** = rất giống (high confidence)
- **0.6-0.8** = khá giống (medium confidence)
- **< 0.6** = ít giống (low confidence)

### 3. Neo4j Vector Index

Neo4j sử dụng `db.index.vector.queryNodes()` để:
- Tự động tính cosine similarity với tất cả embeddings trong index
- Trả về top-k kết quả đã sắp xếp theo score giảm dần
- Filter theo threshold nếu cần

### 4. So Sánh với Threshold

```python
# Trong ingredient_canonicalizer.py

best_score = candidates[0]["score"]  # Ví dụ: 0.95

if best_score >= 0.8:  # high_threshold
    # Auto-merge: Tự động merge vào canonical ingredient
    action = "auto_merge"
    
elif best_score >= 0.6:  # low_threshold
    # Pending với confidence "medium"
    action = "new_pending"
    confidence = "medium"
    
else:
    # Pending với confidence "low"
    action = "new_pending"
    confidence = "low"
```

## Ví Dụ Cụ Thể

### Ví dụ 1: High Score (Auto-merge)
```
Input: "tomato"
Query embedding: [0.5, 0.3, ...]
Best match: "tomatoes" với score = 0.95

0.95 >= 0.8 → Auto-merge
Result: canonical_name = "tomatoes", is_canonical = True
```

### Ví dụ 2: Medium Score (Pending - Medium)
```
Input: "roma tomato"
Query embedding: [0.4, 0.25, ...]
Best match: "tomatoes" với score = 0.72

0.6 <= 0.72 < 0.8 → Pending (medium)
Result: canonical_name = "roma tomato", is_canonical = False, confidence = "medium"
```

### Ví dụ 3: Low Score (Pending - Low)
```
Input: "xyz ingredient"
Query embedding: [0.1, 0.05, ...]
Best match: "some ingredient" với score = 0.45

0.45 < 0.6 → Pending (low)
Result: canonical_name = "xyz ingredient", is_canonical = False, confidence = "low"
```

## Lưu Ý

1. **Embeddings phải được normalize** để dot product = cosine similarity
2. **Query mode vs Document mode**: 
   - Query mode cho search (asymmetric encoding)
   - Document mode cho storage (symmetric encoding)
3. **Neo4j tự động tính**: Không cần tính thủ công, Neo4j vector index đã optimize
4. **Threshold là giá trị cố định**: Có thể điều chỉnh qua config/env variables

