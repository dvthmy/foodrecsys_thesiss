# Cosine Similarity - Giải Thích Chi Tiết

## 1. Cosine Similarity Là Gì?

Cosine Similarity đo độ **tương tự về HƯỚNG** giữa 2 vectors, không phải độ lớn.

**Công thức:**
```
cosine_similarity = dot_product(e1, e2) / (norm(e1) * norm(e2))
```

## 2. Tại Sao Lại Có Công Thức Này?

### A. Ý Nghĩa Hình Học (Visual)

Hãy tưởng tượng mỗi embedding vector là một **mũi tên** trong không gian nhiều chiều:

```
Vector 1: →→→  (hướng lên phải)
Vector 2: →→→  (hướng lên phải) → GIỐNG NHAU
Vector 3: ←←←  (hướng lên trái) → KHÁC NHAU
```

**Cosine similarity = cos(θ)** - góc giữa 2 vectors:

```
Vector A: →
Vector B: →  
Góc θ = 0° → cos(0°) = 1.0 → RẤT GIỐNG NHAU

Vector A: →
Vector B: ↑
Góc θ = 90° → cos(90°) = 0.0 → KHÔNG LIÊN QUAN

Vector A: →
Vector B: ←
Góc θ = 180° → cos(180°) = -1.0 → ĐỐI NGHỊCH NHAU
```

### B. Tại Sao Chia Cho Norm?

**Norm (||vector||) = độ dài của vector**

```
norm = sqrt(x₁² + x₂² + ... + xₙ²)
```

**Vấn đề:** Nếu chỉ dùng dot product:
- Vector lớn sẽ có dot product lớn hơn → không công bằng
- Cần normalize để chỉ so sánh HƯỚNG, không phải độ lớn

**Ví dụ:**
```python
Vector A = [2, 2]     # độ dài = √(4+4) = 2.83
Vector B = [1, 1]     # độ dài = √(1+1) = 1.41
Vector C = [4, 4]     # độ dài = √(16+16) = 5.66

# Dot product
dot(A, B) = 2*1 + 2*1 = 4
dot(A, C) = 2*4 + 2*4 = 16  # Lớn hơn, nhưng cùng hướng!

# Cosine similarity
cos(A, B) = 4 / (2.83 * 1.41) = 1.0   # GIỐNG NHAU 100%
cos(A, C) = 16 / (2.83 * 5.66) = 1.0  # CŨNG GIỐNG NHAU 100%
```

→ Chia cho norm để **chuẩn hóa**, chỉ so sánh hướng!

## 3. Từng Phần Của Công Thức

### A. Dot Product (Tích Vô Hướng)

```python
dot_product = e1[0]*e2[0] + e1[1]*e2[1] + ... + e1[n]*e2[n]
```

**Ý nghĩa:** Tổng tích các phần tử tương ứng

**Ví dụ:**
```python
e1 = [0.5, 0.3, 0.8]
e2 = [0.4, 0.2, 0.9]

dot_product = (0.5 * 0.4) + (0.3 * 0.2) + (0.8 * 0.9)
            = 0.2 + 0.06 + 0.72
            = 0.98
```

**Khi nào dot product lớn?**
- Khi các phần tử cùng dấu (cùng âm hoặc cùng dương)
- Khi các phần tử đều lớn và cùng chiều

### B. Norm (Độ Dài Vector)

```python
norm = sqrt(e1[0]² + e1[1]² + ... + e1[n]²)
```

**Ý nghĩa:** Khoảng cách từ gốc tọa độ đến điểm cuối vector (độ dài)

**Ví dụ:**
```python
e1 = [3, 4]
norm = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
```

### C. Chia Cho Norm × Norm

```python
similarity = dot_product / (norm(e1) * norm(e2))
```

**Mục đích:** 
- Chuẩn hóa về khoảng [-1, 1]
- Chỉ so sánh HƯỚNG, không phải độ lớn
- Giống như "projection" của 1 vector lên vector kia, rồi normalize

## 4. Ví Dụ Cụ Thể

### Ví Dụ 1: Embeddings Giống Nhau

```python
# "tomato" embedding
e1 = [0.8, 0.6, 0.0]

# "tomatoes" embedding (số nhiều, nghĩa giống)
e2 = [0.75, 0.65, 0.05]

# Tính toán
dot = (0.8*0.75) + (0.6*0.65) + (0.0*0.05) = 0.99
norm1 = sqrt(0.8² + 0.6² + 0²) = 1.0
norm2 = sqrt(0.75² + 0.65² + 0.05²) ≈ 0.995

similarity = 0.99 / (1.0 * 0.995) ≈ 0.995  # RẤT GIỐNG!
```

### Ví Dụ 2: Embeddings Khác Nhau

```python
# "tomato" embedding
e1 = [0.8, 0.6, 0.0]

# "car" embedding (hoàn toàn khác)
e2 = [0.1, 0.2, 0.9]

# Tính toán
dot = (0.8*0.1) + (0.6*0.2) + (0.0*0.9) = 0.08 + 0.12 + 0 = 0.20
norm1 = sqrt(0.8² + 0.6² + 0²) = 1.0
norm2 = sqrt(0.1² + 0.2² + 0.9²) ≈ 0.935

similarity = 0.20 / (1.0 * 0.935) ≈ 0.214  # KHÁC NHAU!
```

### Ví Dụ 3: Embeddings Trực Giao (Không Liên Quan)

```python
# Vector hướng X
e1 = [1.0, 0.0, 0.0]

# Vector hướng Y
e2 = [0.0, 1.0, 0.0]

# Tính toán
dot = (1.0*0.0) + (0.0*1.0) + (0.0*0.0) = 0.0
norm1 = 1.0
norm2 = 1.0

similarity = 0.0 / (1.0 * 1.0) = 0.0  # KHÔNG LIÊN QUAN
```

## 5. Tại Sao Dùng Cosine Similarity Cho Embeddings?

### Ưu Điểm:

1. **Bỏ qua độ lớn, chỉ quan tâm hướng:**
   - "tomato" và "tomatoes" có thể có embedding khác độ lớn
   - Nhưng hướng giống nhau → similarity cao

2. **Chuẩn hóa về [0, 1]:**
   - Dễ so sánh và đặt threshold
   - 1.0 = giống hệt, 0.0 = không liên quan

3. **Phù hợp với semantic embeddings:**
   - Embeddings đã capture nghĩa vào các chiều không gian
   - Góc nhỏ = nghĩa gần nhau

4. **Hiệu quả tính toán:**
   - Dot product nhanh (O(n) với n là dimensions)
   - Có thể tối ưu bằng vectorized operations

### So Sánh Với Các Cách Khác:

| Metric | Công Thức | Ưu Điểm | Nhược Điểm |
|--------|-----------|---------|------------|
| **Cosine Similarity** | `dot / (norm1 * norm2)` | Bỏ qua độ lớn, chuẩn hóa | Không quan tâm khoảng cách tuyệt đối |
| **Euclidean Distance** | `sqrt(sum((e1[i] - e2[i])²))` | Đo khoảng cách thực | Phụ thuộc vào độ lớn vector |
| **Dot Product** | `sum(e1[i] * e2[i])` | Nhanh, đơn giản | Phụ thuộc vào độ lớn |

## 6. Ý Nghĩa Trong Thực Tế

### Với Ingredient Embeddings:

```
"tomato" → embedding = [0.8, 0.6, 0.0, ...]
"tomatoes" → embedding = [0.75, 0.65, 0.05, ...]
similarity = 0.95

"tomato" → embedding = [0.8, 0.6, 0.0, ...]
"car" → embedding = [0.1, 0.2, 0.9, ...]
similarity = 0.21
```

**Embedding model học được:**
- Các ingredients liên quan sẽ có embeddings cùng "hướng"
- Ví dụ: vegetables có chiều dương ở một số dimensions
- Fruits có chiều dương ở dimensions khác
- Cosine similarity đo xem chúng có "hướng" giống nhau không

## 7. Kết Luận

**Cosine similarity = đo góc giữa 2 vectors**

- **Góc nhỏ (0°)** → vectors cùng hướng → **giống nhau** → score ≈ 1.0
- **Góc vuông (90°)** → vectors vuông góc → **không liên quan** → score = 0.0
- **Góc lớn (180°)** → vectors ngược hướng → **đối nghịch** → score ≈ -1.0

**Chia cho norm × norm để:**
- Chuẩn hóa về [-1, 1]
- Chỉ so sánh hướng, không phải độ lớn
- Làm cho metric công bằng với mọi vector

