# Giải Thích Chi Tiết: CLIP Embedder - Từng Hàm

## Tổng Quan

File `clip_embedder.py` chứa class `CLIPEmbedder` - service để tạo embedding vector cho hình ảnh món ăn, cho phép tìm kiếm món ăn tương tự dựa trên hình ảnh.

---

## Class: CLIPEmbedder

### 1. Constants (Hằng số)

```python
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
```

**Giải thích**:
- Model mặc định sử dụng CLIP ViT-B/32
- ViT-B/32 = Vision Transformer Base với patch size 32x32
- Cân bằng tốt giữa tốc độ và chất lượng
- Output: 512-dimension vector

---

### 2. `__init__(self, model_name, device)`

**Mục đích**: Khởi tạo CLIP embedder

**Parameters**:
- `model_name` (str | None): Tên model từ Hugging Face. Nếu None → dùng DEFAULT_MODEL
- `device` (str | None): Device để chạy model ('cpu', 'cuda', 'mps'). Nếu None → auto-detect

**Hoạt động**:
```python
def __init__(self, model_name: str | None = None, device: str | None = None):
    self._model_name = model_name or self.DEFAULT_MODEL  # Dùng model mặc định nếu không chỉ định
    self._device = device or self._detect_device()        # Auto-detect device
    self._model: CLIPModel | None = None                 # Model chưa load (lazy loading)
    self._processor: CLIPProcessor | None = None         # Processor chưa load
    self._lock = threading.Lock()                         # Thread-safe lock
```

**Lưu ý**:
- **Lazy Loading**: Model không được load ngay khi khởi tạo
- Model chỉ load khi gọi lần đầu (tiết kiệm memory)
- Thread-safe: Dùng lock để tránh race condition khi load model

---

### 3. `_detect_device(self) -> str`

**Mục đích**: Tự động phát hiện device tốt nhất có sẵn

**Returns**: 
- `"cuda"` nếu có GPU NVIDIA
- `"mps"` nếu có Apple Silicon GPU
- `"cpu"` nếu không có GPU

**Code**:
```python
def _detect_device(self) -> str:
    if torch.cuda.is_available():
        return "cuda"           # NVIDIA GPU (nhanh nhất)
    elif torch.backends.mps.is_available():
        return "mps"            # Apple Silicon (M1/M2/M3)
    return "cpu"                # Fallback
```

**Tại sao cần**:
- GPU nhanh hơn CPU rất nhiều (10-100x)
- Tự động chọn device tốt nhất mà không cần config thủ công

---

### 4. `_load_model(self) -> None`

**Mục đích**: Load CLIP model và processor (lazy loading với thread safety)

**Hoạt động**:
```python
def _load_model(self) -> None:
    if self._model is None:                    # Chưa load?
        with self._lock:                       # Acquire lock (thread-safe)
            if self._model is None:             # Double-check (tránh race condition)
                # Load processor (preprocessing)
                self._processor = CLIPProcessor.from_pretrained(self._model_name)
                
                # Load model
                self._model = CLIPModel.from_pretrained(self._model_name)
                
                # Move to device (GPU/CPU)
                self._model.to(self._device)
                
                # Set to evaluation mode (không cần gradient)
                self._model.eval()
```

**Chi tiết**:
- **Double-check locking pattern**: Đảm bảo chỉ load 1 lần dù nhiều thread gọi cùng lúc
- **CLIPProcessor**: Xử lý image/text (resize, normalize, tokenize)
- **CLIPModel**: Model chính để extract features
- **eval()**: Tắt gradient computation (tiết kiệm memory, tăng tốc)

**Khi nào được gọi**:
- Tự động gọi khi access `self.model` hoặc `self.processor` lần đầu

---

### 5. `@property model(self) -> CLIPModel`

**Mục đích**: Getter cho CLIP model (lazy load nếu chưa có)

**Usage**:
```python
embedder = CLIPEmbedder()
model = embedder.model  # Model tự động load ở đây
```

**Code**:
```python
@property
def model(self) -> CLIPModel:
    self._load_model()      # Load nếu chưa có
    return self._model      # Return model (đã đảm bảo không None)
```

**Lợi ích**:
- Không cần gọi `_load_model()` thủ công
- Model chỉ load khi thực sự cần dùng

---

### 6. `@property processor(self) -> CLIPProcessor`

**Mục đích**: Getter cho CLIP processor (lazy load nếu chưa có)

**Usage**:
```python
processor = embedder.processor  # Processor tự động load
```

**Code**:
```python
@property
def processor(self) -> CLIPProcessor:
    self._load_model()          # Load nếu chưa có
    return self._processor      # Return processor
```

**Processor làm gì**:
- Resize image về 224x224
- Normalize pixel values
- Convert to tensor format
- Tokenize text (nếu có)

---

### 7. `@property embedding_dim(self) -> int`

**Mục đích**: Lấy dimension của embedding vector

**Returns**: `512` (cho ViT-B/32)

**Code**:
```python
@property
def embedding_dim(self) -> int:
    return self.model.config.projection_dim  # 512
```

**Tại sao cần**:
- Biết kích thước vector để allocate memory
- Đảm bảo consistency với Neo4j index (cũng dùng 512-dim)

---

### 8. `embed_image(self, image_path) -> list[float]`

**Mục đích**: Tạo embedding cho 1 hình ảnh

**Parameters**:
- `image_path` (str | Path): Đường dẫn đến file ảnh

**Returns**: 
- `list[float]`: Vector 512 chiều (normalized)

**Quy trình chi tiết**:

```python
def embed_image(self, image_path: str | Path) -> list[float]:
    # 1. Validate file exists
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        # 2. Load và convert image
        image = Image.open(image_path).convert("RGB")
        # → Đảm bảo 3 channels (RGB), loại bỏ alpha channel
        
        # 3. Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        # → Resize 224x224, normalize, convert to tensor
        
        # 4. Move to device (GPU/CPU)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        # → {"pixel_values": tensor on GPU}
        
        # 5. Generate embedding (no gradient)
        with torch.no_grad():  # Tắt gradient để tiết kiệm memory
            image_features = self.model.get_image_features(**inputs)
            # → Shape: [1, 512]
            
            # 6. Normalize vector (L2 normalization)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # → Vector có độ dài = 1 (cho cosine similarity)
        
        # 7. Convert to Python list
        return image_features.squeeze().cpu().tolist()
        # → [0.123, -0.456, 0.789, ..., 0.234] (512 numbers)
        
    except Exception as e:
        raise ValueError(f"Failed to process image {image_path}: {e}")
```

**Ví dụ sử dụng**:
```python
embedder = CLIPEmbedder()
embedding = embedder.embed_image("pasta.jpg")
# → [0.123, -0.456, 0.789, ..., 0.234]
```

**Tại sao normalize**:
- Cosine similarity = dot product của 2 normalized vectors
- Normalize giúp so sánh dễ dàng hơn
- Range: -1 đến 1 (thường 0 đến 1 cho similar images)

---

### 9. `embed_images(self, image_paths) -> list[list[float]]`

**Mục đích**: Tạo embedding cho nhiều hình ảnh cùng lúc (batch processing)

**Parameters**:
- `image_paths` (list[str | Path]): Danh sách đường dẫn ảnh

**Returns**:
- `list[list[float]]`: List các embedding vectors

**Quy trình**:

```python
def embed_images(self, image_paths: list[str | Path]) -> list[list[float]]:
    if not image_paths:
        return []  # Empty list → empty result
    
    # 1. Load tất cả images (skip invalid files)
    images = []
    valid_indices = []
    for idx, path in enumerate(image_paths):
        try:
            image = Image.open(path).convert("RGB")
            images.append(image)
            valid_indices.append(idx)  # Track valid indices
        except Exception:
            continue  # Skip invalid files
    
    if not images:
        return []  # No valid images
    
    # 2. Batch preprocessing
    inputs = self.processor(images=images, return_tensors="pt", padding=True)
    # → padding=True: Pad images khác size về cùng size
    inputs = {k: v.to(self._device) for k, v in inputs.items()}
    
    # 3. Batch inference
    with torch.no_grad():
        image_features = self.model.get_image_features(**inputs)
        # → Shape: [batch_size, 512]
        
        # 4. Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 5. Convert to list
    return image_features.cpu().tolist()
    # → [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
```

**Ưu điểm batch processing**:
- **Nhanh hơn**: Process nhiều ảnh cùng lúc (GPU parallel)
- **Hiệu quả**: Giảm overhead của việc load model nhiều lần
- **Tự động skip**: Bỏ qua file lỗi, không crash toàn bộ

**Ví dụ**:
```python
paths = ["pasta.jpg", "pizza.jpg", "sushi.jpg"]
embeddings = embedder.embed_images(paths)
# → [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
```

---

### 10. `embed_text(self, text) -> list[float]`

**Mục đích**: Tạo embedding cho text (để so sánh text với image)

**Parameters**:
- `text` (str): Text description

**Returns**:
- `list[float]`: Vector 512 chiều (normalized)

**Code**:
```python
def embed_text(self, text: str) -> list[float]:
    # 1. Preprocess text
    inputs = self.processor(text=[text], return_tensors="pt", padding=True)
    # → Tokenize text, convert to token IDs
    inputs = {k: v.to(self._device) for k, v in inputs.items()}
    
    # 2. Get text features
    with torch.no_grad():
        text_features = self.model.get_text_features(**inputs)
        # → Shape: [1, 512]
        
        # 3. Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 4. Convert to list
    return text_features.squeeze().cpu().tolist()
```

**Use case**:
- Tìm ảnh món ăn từ text query: "Italian pasta with tomato sauce"
- Cross-modal search: Text → Image similarity

**Ví dụ**:
```python
text_emb = embedder.embed_text("Italian pasta")
image_emb = embedder.embed_image("pasta.jpg")
similarity = embedder.compute_similarity(text_emb, image_emb)
# → 0.85 (rất giống nhau!)
```

---

### 11. `compute_similarity(self, embedding1, embedding2) -> float`

**Mục đích**: Tính cosine similarity giữa 2 embeddings

**Parameters**:
- `embedding1` (list[float]): Embedding vector thứ nhất
- `embedding2` (list[float]): Embedding vector thứ hai

**Returns**:
- `float`: Cosine similarity score (-1 đến 1)

**Code**:
```python
def compute_similarity(
    self,
    embedding1: list[float],
    embedding2: list[float],
) -> float:
    # Convert to tensors
    t1 = torch.tensor(embedding1)
    t2 = torch.tensor(embedding2)
    
    # Cosine similarity = dot product / (norm1 * norm2)
    # Vì đã normalize → cosine = dot product
    return torch.nn.functional.cosine_similarity(t1, t2, dim=0).item()
```

**Công thức**:
```
cosine_similarity = (A · B) / (||A|| * ||B||)
```

Vì embeddings đã được normalize (||A|| = ||B|| = 1):
```
cosine_similarity = A · B  (dot product)
```

**Ý nghĩa score**:
- `1.0`: Giống hệt nhau
- `0.8-0.9`: Rất giống
- `0.5-0.7`: Tương đối giống
- `0.0`: Không liên quan
- `-1.0`: Đối lập hoàn toàn

**Ví dụ**:
```python
emb1 = embedder.embed_image("pasta1.jpg")
emb2 = embedder.embed_image("pasta2.jpg")
score = embedder.compute_similarity(emb1, emb2)
# → 0.87 (2 món pasta rất giống nhau)
```

---

## Global Functions (Hàm toàn cục)

### 12. `get_clip_embedder() -> CLIPEmbedder`

**Mục đích**: Singleton pattern - trả về 1 instance duy nhất của CLIPEmbedder

**Code**:
```python
# Global variables
_embedder: CLIPEmbedder | None = None
_embedder_lock = threading.Lock()

def get_clip_embedder() -> CLIPEmbedder:
    global _embedder
    with _embedder_lock:                    # Thread-safe
        if _embedder is None:                # Chưa tạo?
            _embedder = CLIPEmbedder()       # Tạo mới
        return _embedder                     # Return instance
```

**Lợi ích**:
- **Memory efficient**: Chỉ load model 1 lần
- **Fast**: Reuse model đã load
- **Thread-safe**: Dùng lock để tránh race condition

**Usage**:
```python
# Lần 1: Tạo mới
embedder1 = get_clip_embedder()

# Lần 2: Reuse cùng instance
embedder2 = get_clip_embedder()

# embedder1 và embedder2 là cùng 1 object!
assert embedder1 is embedder2  # True
```

---

## Tóm Tắt Workflow

### Khi embed 1 image:

```
1. User gọi: embedder.embed_image("pasta.jpg")
   ↓
2. Check model loaded? → Nếu chưa → load model
   ↓
3. Load image: Image.open("pasta.jpg").convert("RGB")
   ↓
4. Preprocess: processor(images=image)
   → Resize 224x224, normalize, tensor
   ↓
5. Forward pass: model.get_image_features(**inputs)
   → ViT encoder → 512-dim vector
   ↓
6. Normalize: vector / norm(vector)
   → Unit vector (length = 1)
   ↓
7. Return: [0.123, -0.456, ..., 0.234] (512 numbers)
```

### Khi embed nhiều images:

```
1. Load tất cả images vào memory
   ↓
2. Batch preprocessing (padding nếu cần)
   ↓
3. Batch forward pass (parallel trên GPU)
   ↓
4. Normalize tất cả vectors
   ↓
5. Return list of embeddings
```

---

## Best Practices

### 1. **Lazy Loading**
```python
# ✅ Tốt: Model chỉ load khi cần
embedder = CLIPEmbedder()
embedding = embedder.embed_image("pasta.jpg")  # Model load ở đây

# ❌ Không cần: Load ngay khi init
embedder = CLIPEmbedder()
embedder._load_model()  # Không cần thiết
```

### 2. **Batch Processing**
```python
# ✅ Tốt: Process nhiều ảnh cùng lúc
embeddings = embedder.embed_images(["pasta.jpg", "pizza.jpg"])

# ❌ Chậm: Process từng ảnh
embeddings = [embedder.embed_image(path) for path in paths]
```

### 3. **Error Handling**
```python
# ✅ Tốt: Handle exceptions
try:
    embedding = embedder.embed_image("pasta.jpg")
except FileNotFoundError:
    print("Image not found")
except ValueError as e:
    print(f"Processing error: {e}")
```

### 4. **Singleton Pattern**
```python
# ✅ Tốt: Dùng get_clip_embedder()
embedder = get_clip_embedder()

# ❌ Không tốt: Tạo nhiều instances
embedder1 = CLIPEmbedder()
embedder2 = CLIPEmbedder()  # Load model 2 lần!
```

---

## Kết Luận

`CLIPEmbedder` là một service hoàn chỉnh để:
- ✅ Embed images thành vectors
- ✅ Embed text thành vectors (cross-modal)
- ✅ Compute similarity giữa embeddings
- ✅ Batch processing cho hiệu quả
- ✅ Thread-safe và memory efficient
- ✅ Auto-detect device (GPU/CPU)

Mỗi hàm có vai trò riêng và được thiết kế để tối ưu performance và dễ sử dụng!

