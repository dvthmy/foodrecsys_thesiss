# Giải Thích Chi Tiết: Ingredient Embedder - Toàn Bộ Code

## Tổng Quan

File `ingredient_embedder.py` chứa class `IngredientEmbedder` - service để tạo embedding vector cho nguyên liệu sử dụng Google's EmbeddingGemma model.

**Mục đích chính**: Ingredient name → 512-dim vector để tính semantic similarity

---

## Phần 1: Imports và Setup

```python
import logging
import threading
from typing import Literal

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config import config
```

**Giải thích**:
- `logging`: Logging cho debugging
- `threading`: Thread-safe operations (Lock)
- `typing.Literal`: Type hints cho mode ("query" | "document")
- `numpy`: Array operations, similarity computation
- `torch`: PyTorch (để detect device)
- `sentence_transformers`: High-level API cho embedding models
- `config`: Config cho HF_API_KEY

### Hugging Face Login

```python
try:
    from huggingface_hub import login
    login(token=config.HF_API_KEY)
except Exception:
    pass
```

**Mục đích**: Login vào Hugging Face để download model (nếu cần authentication)

**Error handling**: Nếu fail → bỏ qua (có thể model đã public)

---

## Phần 2: Class Definition và Constants

### Class: IngredientEmbedder

```python
class IngredientEmbedder:
    """Service for generating ingredient embeddings using EmbeddingGemma."""
```

### Constants

#### 1. DEFAULT_MODEL
```python
DEFAULT_MODEL = "google/embeddinggemma-300m"
```

**Giải thích**:
- EmbeddingGemma 300M parameters
- Cân bằng tốt giữa speed và quality
- Chuyên biệt cho semantic text understanding

#### 2. OUTPUT_DIM
```python
OUTPUT_DIM = 512
```

**Giải thích**:
- Native dimension: 768
- Truncate về 512 để consistency với Neo4j index
- Neo4j vector index dùng 512-dim

#### 3. Asymmetric Retrieval Prefixes

```python
QUERY_PREFIX = "task: search result | query: {text}"
DOCUMENT_PREFIX = "title: {title} | text: {text}"
DOC_TEMPLATE = "title: {title} | text: {text}"
```

**Giải thích**:
- **Asymmetric Retrieval**: Query và Document dùng format khác nhau
- **Query**: Ngắn, vague, functional
  - Ví dụ: `"task: search result | query: basil"`
- **Document**: Có title và description
  - Ví dụ: `"title: basil | text: a sweet herb used in Italian cuisine"`
- **Tại sao**: EmbeddingGemma được train với format này → tốt hơn

**DOC_TEMPLATE vs DOCUMENT_PREFIX**:
- Cùng format, nhưng DOC_TEMPLATE dùng trong code
- DOCUMENT_PREFIX có thể deprecated

---

## Phần 3: Initialization (`__init__`)

```python
def __init__(
    self,
    model_name: str | None = None,
    device: str | None = None,
    use_context: bool = True,
    output_dim: int | None = None,
):
```

### Parameters

1. **`model_name`** (str | None):
   - Hugging Face model name
   - Default: `None` → dùng `DEFAULT_MODEL`

2. **`device`** (str | None):
   - Device: `'cpu'`, `'cuda'`, `'mps'`
   - Default: `None` → auto-detect

3. **`use_context`** (bool):
   - Có wrap ingredient với context không?
   - Default: `True`
   - **Note**: Hiện tại không được dùng trong code

4. **`output_dim`** (int | None):
   - Output dimension
   - Default: `None` → dùng `OUTPUT_DIM` (512)

### Code Flow

```python
self._model_name = model_name or self.DEFAULT_MODEL
self._device = device or self._detect_device()
self._use_context = use_context  # Not used currently
self._output_dim = output_dim or self.OUTPUT_DIM
self._model: SentenceTransformer | None = None  # Lazy loading
self._lock = threading.Lock()  # Thread-safe
```

**Lưu ý**:
- Model chưa load (lazy loading)
- Thread-safe với Lock

---

## Phần 4: Helper Methods

### 1. `_detect_device(self) -> str`

```python
def _detect_device(self) -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

**Tương tự như CLIP và Gemma**: Auto-detect device tốt nhất

---

### 2. `_load_model(self) -> None`

```python
def _load_model(self) -> None:
    if self._model is None:
        with self._lock:  # Thread-safe
            if self._model is None:  # Double-check
                logger.info("Loading EmbeddingGemma model: %s", self._model_name)
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                )
                logger.info("EmbeddingGemma model loaded on device: %s", self._device)
```

**Giải thích**:
- **Double-check locking**: Thread-safe pattern
- **SentenceTransformer**: High-level API từ sentence-transformers
  - Tự động handle:
    - Model loading
    - Tokenization
    - Encoding
    - Pooling

**SentenceTransformer vs Raw Model**:
- SentenceTransformer: Dễ dùng, tự động pooling
- Raw model: Phải tự handle pooling

---

## Phần 5: Properties

### 1. `@property model(self) -> SentenceTransformer`

```python
@property
def model(self) -> SentenceTransformer:
    self._load_model()  # Load nếu chưa có
    return self._model
```

**Lazy loading getter**

---

### 2. `@property embedding_dim(self) -> int`

```python
@property
def embedding_dim(self) -> int:
    return self._output_dim  # 512
```

**Trả về output dimension**

---

## Phần 6: Text Preparation Methods

### 1. `_prepare_query_text(self, query) -> str`

```python
def _prepare_query_text(self, query: str) -> str:
    return self.QUERY_PREFIX.format(text=query.lower().strip())
```

**Mục đích**: Format query text với prefix

**Ví dụ**:
```python
query = "basil"
formatted = "task: search result | query: basil"
```

**Tại sao lowercase**:
- Consistency
- EmbeddingGemma case-insensitive

---

### 2. `_prepare_document_text(self, ingredient, description, dish_name) -> str`

```python
def _prepare_document_text(
    self, 
    ingredient: str, 
    description: str | None = None,
    dish_name: str | None = None,
) -> str:
    title = ingredient.lower().strip()
    
    # Build description từ context
    if description:
        text = description.lower().strip()
    elif dish_name:
        text = f"food ingredient used in {dish_name.lower().strip()}"
    else:
        text = "food ingredient"
    
    return self.DOCUMENT_PREFIX.format(title=title, text=text)
```

**Mục đích**: Format document text với title và description

**Priority**:
1. `description` (nếu có) → dùng trực tiếp
2. `dish_name` (nếu có) → format: "food ingredient used in {dish_name}"
3. Default → "food ingredient"

**Ví dụ**:
```python
# Case 1: Có description
_prepare_document_text("basil", description="a sweet herb")
# → "title: basil | text: a sweet herb"

# Case 2: Có dish_name
_prepare_document_text("basil", dish_name="Italian Pasta")
# → "title: basil | text: food ingredient used in italian pasta"

# Case 3: Không có gì
_prepare_document_text("basil")
# → "title: basil | text: food ingredient"
```

---

## Phần 7: Main Embedding Method (`embed_ingredient`)

```python
def embed_ingredient(
    self,
    ingredient: str,
    mode: Literal["query", "document"] = "document",
    dish_name: str | None = None,
    description: str | None = "food ingredient",
) -> list[float]:
```

**Mục đích**: Generate embedding cho ingredient

### Parameters

1. **`ingredient`** (str): Ingredient name
2. **`mode`** ("query" | "document"): Embedding mode
3. **`dish_name`** (str | None): Optional dish name
4. **`description`** (str | None): Optional description

### Step 1: Load Model

```python
_ = self.model  # Load nếu chưa có
```

**Trick**: Dùng `_` để indicate không dùng return value

### Step 2: Query Mode

```python
if mode == "query":
    # CASE A: QUERY
    query_text = ingredient.lower().strip()
    embedding = self.model.encode_query(
        query_text, 
        truncate_dim=self._output_dim
    )
```

**Giải thích**:
- **`encode_query()`**: Method từ SentenceTransformer
- Tự động thêm query prefix: `"task: search result | query: {text}"`
- **`truncate_dim`**: Truncate từ 768 → 512

**Tại sao dùng `encode_query()`**:
- Tự động handle prefix
- Optimized cho search queries

### Step 3: Document Mode

```python
else:
    # CASE B: DOCUMENT
    # 1. Construct Content
    if description:
        content = description.lower().strip()
    else:
        content = "food ingredient"
    
    # 2. Format with Official Template
    full_text = self.DOC_TEMPLATE.format(
        title=ingredient.lower().strip(),
        text=content
    )
    # Result: "title: basil | text: a sweet herb..."
    
    # 3. Encode as "raw" text
    embedding = self.model.encode(
        full_text,
        prompt_name=None,  # Prevent double prefixing
        truncate_dim=self._output_dim
    )
```

**Giải thích**:
- **Manual formatting**: Tự format với DOC_TEMPLATE
- **`encode()`**: Generic encode method
- **`prompt_name=None`**: Không tự động thêm prefix (vì đã format thủ công)

**Tại sao không dùng `encode_document()`**:
- Comment nói: "might force 'title: none'"
- Manual format cho control tốt hơn

**Note**: Có `print(content)` ở line 197 - có thể là debug code, nên remove

### Step 4: Convert to List

```python
embedding = np.asarray(embedding)
return embedding.tolist()
```

**Convert**: numpy array → Python list

**Tại sao**:
- Neo4j cần list[float]
- Dễ serialize

---

## Phần 8: Similarity Computation

### 1. `compute_similarity(self, embedding1, embedding2) -> float`

```python
def compute_similarity(
    self,
    embedding1: list[float],
    embedding2: list[float],
) -> float:
    e1 = np.array(embedding1)
    e2 = np.array(embedding2)
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
```

**Mục đích**: Tính cosine similarity

**Công thức**:
```
cosine_similarity = (A · B) / (||A|| * ||B||)
```

**Giải thích**:
- `np.dot(e1, e2)`: Dot product
- `np.linalg.norm(e1)`: L2 norm (độ dài vector)
- Kết quả: -1 đến 1 (thường 0 đến 1 cho normalized vectors)

**Tại sao không normalize trước**:
- Code này không normalize trước
- Nếu vectors đã normalize → `cosine = dot product`

---

### 2. `find_similar(self, query_ingredient, candidate_embeddings, candidate_names, top_k, threshold) -> list[dict]`

```python
def find_similar(
    self,
    query_ingredient: str,
    candidate_embeddings: list[list[float]],
    candidate_names: list[str],
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[dict]:
```

**Mục đích**: Tìm ingredients tương tự từ list candidates

### Parameters

1. **`query_ingredient`**: Ingredient cần tìm
2. **`candidate_embeddings`**: Pre-computed embeddings của candidates
3. **`candidate_names`**: Tên tương ứng với embeddings
4. **`top_k`**: Số kết quả trả về
5. **`threshold`**: Minimum similarity score

### Step 1: Validate

```python
if not candidate_embeddings:
    return []
```

**Early return**: Nếu không có candidates

### Step 2: Generate Query Embedding

```python
query_embedding = np.array(self.embed_ingredient(query_ingredient, mode="query"))
candidates = np.array(candidate_embeddings)
```

**Query mode**: Dùng query mode cho search

### Step 3: Compute Similarities

```python
similarities = np.dot(candidates, query_embedding)
```

**Vectorized computation**:
- `candidates`: Shape `[n_candidates, 512]`
- `query_embedding`: Shape `[512]`
- `np.dot()`: Matrix-vector multiplication
- Result: `[n_candidates]` - similarity scores

**Tại sao nhanh**:
- Vectorized (numpy)
- Không loop qua từng candidate

### Step 4: Sort và Filter

```python
indices = np.argsort(similarities)[::-1]  # Sort descending

results = []
for idx in indices[:top_k]:  # Top k
    score = float(similarities[idx])
    if score >= threshold:  # Filter by threshold
        results.append({
            "name": candidate_names[idx],
            "score": score,
        })
```

**Giải thích**:
- `np.argsort()`: Get sorted indices
- `[::-1]`: Reverse (descending order)
- Filter: Top k + threshold

**Return format**:
```python
[
    {"name": "oregano", "score": 0.85},
    {"name": "thyme", "score": 0.78},
    ...
]
```

---

## Phần 9: Contextual Embedding (`embed_with_category_context`)

```python
def embed_with_category_context(
    self,
    ingredient: str,
    category: str | None = None,
    origin: str | None = None,
) -> list[float]:
```

**Mục đích**: Generate embedding với category/origin context

### Step 1: Build Description

```python
context_parts = []
if category:
    context_parts.append(f"{category}")
if origin:
    context_parts.append(f"{origin} cuisine")
context_parts.append("food ingredient")

description = " ".join(context_parts)
```

**Ví dụ**:
```python
category = "herb"
origin = "Italian"
# → "herb italian cuisine food ingredient"
```

### Step 2: Format Document Text

```python
text = self._prepare_document_text(ingredient, description)
```

**Format**: "title: basil | text: herb italian cuisine food ingredient"

### Step 3: Encode và Normalize

```python
embedding = np.asarray(self.model.encode_document(text, truncate_dim=512))
embedding = embedding / np.linalg.norm(embedding)  # Normalize
```

**Giải thích**:
- **`encode_document()`**: Dùng method này (khác với `embed_ingredient`)
- **Normalize**: L2 normalization (length = 1)

**Tại sao normalize**:
- Cosine similarity = dot product khi normalized
- Consistent với các embeddings khác

### Step 4: Return

```python
return embedding.tolist()
```

---

## Phần 10: Singleton Pattern

### Global Variables

```python
_embedder: IngredientEmbedder | None = None
_embedder_lock = threading.Lock()
```

**Singleton instance**: Chỉ 1 instance trong app

### Factory Function

```python
def get_ingredient_embedder(
    model_name: str | None = None,
    device: str | None = None,
) -> IngredientEmbedder:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = IngredientEmbedder(model_name=model_name, device=device)
        return _embedder
```

**Thread-safe singleton**: Dùng lock để tránh race condition

**Usage**:
```python
embedder1 = get_ingredient_embedder()
embedder2 = get_ingredient_embedder()
assert embedder1 is embedder2  # True
```

---

## Workflow Tổng Quan

### Khi embed 1 ingredient (document mode):

```
1. User gọi: embedder.embed_ingredient("basil", mode="document")
   ↓
2. Load model (nếu chưa load)
   ↓
3. Format text: "title: basil | text: food ingredient"
   ↓
4. Encode với model.encode()
   ├── Tokenize
   ├── Forward pass qua EmbeddingGemma
   ├── Pooling (mean pooling)
   └── Truncate 768 → 512
   ↓
5. Convert to list
   ↓
6. Return: [0.123, -0.456, ..., 0.234] (512 numbers)
```

### Khi tìm similar ingredients:

```
1. User gọi: find_similar("basil", candidates, names, top_k=5)
   ↓
2. Generate query embedding cho "basil" (query mode)
   ↓
3. Compute dot product với tất cả candidates (vectorized)
   ↓
4. Sort by similarity (descending)
   ↓
5. Filter top k + threshold
   ↓
6. Return: [{"name": "oregano", "score": 0.85}, ...]
```

---

## So Sánh Query vs Document Mode

| Đặc điểm | Query Mode | Document Mode |
|----------|------------|---------------|
| **Prefix** | `"task: search result \| query: {text}"` | `"title: {title} \| text: {text}"` |
| **Method** | `encode_query()` | `encode()` với manual format |
| **Use case** | Search queries | Indexed documents |
| **Example** | `"basil"` → search | `"basil"` với description → store |

**Tại sao khác nhau**:
- EmbeddingGemma được train với asymmetric format
- Query: Ngắn, vague
- Document: Có context (title + description)
- → Embeddings khác nhau → Better retrieval

---

## Best Practices

### 1. **Query Mode cho Search**
```python
# ✅ Tốt: Query mode cho search
query_emb = embedder.embed_ingredient("basil", mode="query")
```

### 2. **Document Mode cho Indexing**
```python
# ✅ Tốt: Document mode với description
doc_emb = embedder.embed_ingredient(
    "basil", 
    mode="document",
    description="a sweet herb used in Italian cuisine"
)
```

### 3. **Batch Similarity Search**
```python
# ✅ Tốt: Pre-compute embeddings, dùng find_similar
candidate_embs = [embedder.embed_ingredient(name) for name in names]
results = embedder.find_similar("basil", candidate_embs, names, top_k=5)
```

### 4. **Contextual Embeddings**
```python
# ✅ Tốt: Dùng category/origin khi có
emb = embedder.embed_with_category_context(
    "basil",
    category="herb",
    origin="Italian"
)
```

---

## Issues và Improvements

### 1. **Debug Print Statement**
```python
# Line 197: Có print(content) - nên remove
print(content)  # ❌ Debug code
```

### 2. **Unused Parameter**
```python
# use_context parameter không được dùng
def __init__(self, ..., use_context: bool = True, ...):
    self._use_context = use_context  # Không dùng
```

### 3. **Inconsistent Normalization**
```python
# embed_ingredient: Không normalize
# embed_with_category_context: Có normalize
# → Nên consistent
```

---

## Kết Luận

`IngredientEmbedder` là một service hoàn chỉnh để:
- ✅ Generate embeddings cho ingredients
- ✅ Support query và document modes (asymmetric retrieval)
- ✅ Compute similarity giữa embeddings
- ✅ Find similar ingredients từ candidates
- ✅ Contextual embeddings với category/origin
- ✅ Thread-safe singleton pattern
- ✅ Lazy loading cho memory efficiency

Mỗi method có vai trò riêng và được thiết kế để tối ưu cho semantic similarity search!

