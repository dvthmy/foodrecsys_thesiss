# Giải Thích Chi Tiết: CLIP, GEMMA 3, và EmbeddingGemma

## Tổng Quan

Hệ thống sử dụng **3 model AI khác nhau**, mỗi model có vai trò riêng:

1. **Gemma 3** - Trích xuất (Extraction)
2. **EmbeddingGemma** - Embedding cho nguyên liệu (Ingredient Embedding)
3. **CLIP** - Embedding cho hình ảnh (Image Embedding)

---

## 1. GEMMA 3 (GemmaExtractor)

### Mục đích
**Trích xuất nguyên liệu từ mô tả văn bản** (Text → Structured Data)

### Model Details
- **Model ID**: `google/gemma-3-1b-it`
- **Loại**: Causal Language Model (Text Generation)
- **Kiến trúc**: Transformer-based decoder-only model
- **Kích thước**: 1B parameters (1 tỷ tham số)
- **Nhà phát triển**: Google

### Cấu trúc

```
Input: Text Description
  ↓
[Tokenization] → Token IDs
  ↓
[Gemma 3 Model]
  ├── Embedding Layer (token → vectors)
  ├── Transformer Blocks (attention layers)
  │   ├── Self-Attention
  │   ├── Feed-Forward Networks
  │   └── Layer Normalization
  └── Output Head (generation)
  ↓
Output: Generated JSON Text
  ↓
[Parsing] → Structured Data
```

### Quy trình hoạt động

1. **Input**: Mô tả món ăn dạng text
   ```
   "A delicious Italian pasta dish with tomato sauce, basil, and parmesan cheese"
   ```

2. **Prompt Engineering**: Format prompt với template
   ```python
   TEXT_PROMPT = """Extract all ingredients mentioned in this food dish description.
   
   Description: {description}
   
   Return a JSON object with the following structure:
   {
       "dish_name": "...",
       "ingredients": ["...", "..."],
       "cuisine": "...",
       "confidence": "high|medium|low"
   }
   """
   ```

3. **Model Processing**:
   - Tokenize input text
   - Pass qua các transformer layers
   - Generate text response (JSON format)

4. **Output**: Structured data
   ```json
   {
       "dish_name": "Italian Pasta",
       "ingredients": ["pasta", "tomato sauce", "basil", "parmesan cheese"],
       "cuisine": "Italian",
       "confidence": "high"
   }
   ```

### Tại sao dùng cho Extraction?

✅ **Ưu điểm**:
- **Hiểu ngữ cảnh**: Model được train trên lượng lớn text, hiểu được mối quan hệ giữa từ ngữ
- **Suy luận**: Có thể infer nguyên liệu từ tên món (ví dụ: "pizza" → dough, tomato sauce, cheese)
- **Structured Output**: Có thể generate JSON format trực tiếp
- **Local Processing**: Chạy local, không cần API key

❌ **Không phù hợp cho Embedding**:
- Model này là **generative** (tạo text), không phải **embedding model**
- Output là text, không phải vector số
- Không được optimize cho similarity search

---

## 2. EmbeddingGemma (IngredientEmbedder)

### Mục đích
**Tạo embedding vector cho nguyên liệu** (Text → Vector) để tính similarity

### Model Details
- **Model ID**: `google/embeddinggemma-300m`
- **Loại**: Embedding Model (Dual Encoder)
- **Kiến trúc**: Transformer-based encoder
- **Kích thước**: 300M parameters
- **Output Dimension**: 768 (native) → 512 (truncated for Neo4j)
- **Nhà phát triển**: Google

### Cấu trúc

```
Input: Ingredient Name + Context
  ↓
[Tokenization] → Token IDs
  ↓
[EmbeddingGemma Encoder]
  ├── Embedding Layer
  ├── Transformer Encoder Blocks
  │   ├── Self-Attention
  │   ├── Feed-Forward
  │   └── Layer Norm
  └── Pooling Layer (mean pooling)
  ↓
Output: 512-dim Vector
```

### Quy trình hoạt động

1. **Asymmetric Retrieval Format**:
   - **Query mode**: `"task: search result | query: {ingredient}"`
   - **Document mode**: `"title: {ingredient} | text: {description}"`

2. **Example**:
   ```python
   # Query mode (for search)
   query_text = "task: search result | query: basil"
   query_embedding = model.encode_query(query_text, truncate_dim=512)
   
   # Document mode (for indexing)
   doc_text = "title: basil | text: a sweet herb used in Italian cuisine"
   doc_embedding = model.encode(doc_text, truncate_dim=512)
   ```

3. **Output**: Vector 512 chiều
   ```python
   embedding = [0.123, -0.456, 0.789, ..., 0.234]  # 512 numbers
   ```

### Tại sao dùng cho Embedding?

✅ **Ưu điểm**:
- **Semantic Understanding**: Hiểu được mối quan hệ ngữ nghĩa
  - "basil" gần với "oregano" (cùng là herbs)
  - "tomato" gần với "tomato sauce" (cùng category)
- **Asymmetric Retrieval**: Hỗ trợ query/document mode khác nhau
- **Normalized Output**: Embeddings đã được normalize (0-1 range)
- **Optimized for Similarity**: Được train để tối ưu cosine similarity

❌ **Không phù hợp cho Extraction**:
- Model này chỉ tạo vector, không generate text
- Không thể extract structured data từ description

---

## 3. CLIP (CLIPEmbedder)

### Mục đích
**Tạo embedding vector cho hình ảnh** (Image → Vector) để tìm món ăn tương tự về mặt hình ảnh

### Model Details
- **Model ID**: `openai/clip-vit-base-patch32`
- **Loại**: Vision-Language Model (Dual Encoder)
- **Kiến trúc**: 
  - **Image Encoder**: Vision Transformer (ViT-B/32)
  - **Text Encoder**: Transformer
- **Output Dimension**: 512
- **Nhà phát triển**: OpenAI

### Cấu trúc

```
Input: Image
  ↓
[Image Preprocessing]
  ├── Resize to 224x224
  ├── Normalize
  └── Patch Extraction (32x32 patches)
  ↓
[ViT Image Encoder]
  ├── Patch Embeddings
  ├── Position Embeddings
  ├── Transformer Encoder Blocks
  │   ├── Multi-Head Self-Attention
  │   ├── Feed-Forward
  │   └── Layer Norm
  └── CLS Token Pooling
  ↓
[Projection Layer] → 512-dim
  ↓
Output: Normalized 512-dim Vector
```

### Quy trình hoạt động

1. **Image Processing**:
   ```python
   image = Image.open(path).convert("RGB")  # Load image
   inputs = processor(images=image)  # Preprocess
   ```

2. **Feature Extraction**:
   ```python
   image_features = model.get_image_features(**inputs)
   # Shape: [1, 512]
   ```

3. **Normalization**:
   ```python
   image_features = image_features / image_features.norm(dim=-1, keepdim=True)
   ```

4. **Output**: Vector 512 chiều
   ```python
   embedding = [0.234, -0.123, 0.567, ..., 0.890]  # 512 numbers
   ```

### Tại sao dùng cho Image Embedding?

✅ **Ưu điểm**:
- **Visual Understanding**: Hiểu được đặc trưng hình ảnh
  - Màu sắc, texture, hình dạng
  - Bố cục món ăn
- **Cross-Modal**: Có thể so sánh image với text
- **Pre-trained**: Được train trên 400M image-text pairs
- **Efficient**: ViT-B/32 là balance tốt giữa speed và quality

❌ **Không phù hợp cho Text Extraction**:
- CLIP không thể extract ingredients từ text
- CLIP chỉ tạo embedding, không generate text

---

## So Sánh 3 Model

| Đặc điểm | Gemma 3 | EmbeddingGemma | CLIP |
|----------|---------|----------------|------|
| **Mục đích** | Extraction | Ingredient Embedding | Image Embedding |
| **Input** | Text description | Ingredient name | Image |
| **Output** | JSON (structured) | 512-dim vector | 512-dim vector |
| **Loại Model** | Generative LM | Embedding Model | Vision-Language |
| **Parameters** | 1B | 300M | ~150M (ViT) |
| **Use Case** | Parse text → ingredients | Find similar ingredients | Find similar dishes (visual) |
| **Example** | "pasta with tomato" → `["pasta", "tomato"]` | "basil" → `[0.1, 0.2, ...]` | `image.jpg` → `[0.3, 0.4, ...]` |

---

## Tại sao tách riêng 3 model?

### 1. **Separation of Concerns**
- Mỗi model làm một việc tốt nhất
- Gemma 3: Text understanding → extraction
- EmbeddingGemma: Semantic similarity → ingredient matching
- CLIP: Visual understanding → image similarity

### 2. **Optimization**
- Mỗi model được optimize cho task riêng
- Không thể dùng 1 model cho tất cả tasks

### 3. **Flexibility**
- Có thể thay đổi model độc lập
- Ví dụ: Upgrade CLIP mà không ảnh hưởng Gemma 3

### 4. **Performance**
- EmbeddingGemma nhẹ hơn Gemma 3 (300M vs 1B)
- CLIP chuyên biệt cho vision tasks

---

## Workflow trong Hệ Thống

```
1. User uploads image + description
   ↓
2. Gemma 3 extracts ingredients from description
   Input: "Italian pasta with tomato sauce"
   Output: ["pasta", "tomato sauce", "basil"]
   ↓
3. EmbeddingGemma creates embeddings for each ingredient
   "pasta" → [0.1, 0.2, ...]
   "tomato sauce" → [0.3, 0.4, ...]
   ↓
4. CLIP creates image embedding
   image.jpg → [0.5, 0.6, ...]
   ↓
5. Store in Neo4j:
   - Ingredients (with EmbeddingGemma vectors)
   - Dish (with CLIP image vector)
   ↓
6. Recommendation:
   - Content-based: Use EmbeddingGemma similarity
   - Image-based: Use CLIP similarity
```

---

## Kết Luận

- **Gemma 3**: Dùng để **extract** (trích xuất) thông tin từ text
- **EmbeddingGemma**: Dùng để **embed** (vector hóa) nguyên liệu cho similarity search
- **CLIP**: Dùng để **embed** (vector hóa) hình ảnh cho visual similarity

Mỗi model có vai trò riêng và không thể thay thế cho nhau!

