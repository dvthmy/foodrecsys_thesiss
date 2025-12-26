# Công Thức Collaborative Filtering (CF)

## 1. Mean-Centered Cosine Similarity

### Công thức tính độ tương đồng giữa 2 users:

```
Sim(u, v) = ( Σ_{i ∈ I} (r_{u,i} - r̄_u)(r_{v,i} - r̄_v) ) / ( √( Σ_{i ∈ I} (r_{u,i} - r̄_u)² ) × √( Σ_{i ∈ I} (r_{v,i} - r̄_v)² ) )
```

### Trong đó:

**Ký hiệu:**
- `u, v`: User u và User v
- `I`: Tập hợp các món cả 2 user đều đã đánh giá (intersection)
- `r_{u,i}`: Rating của user u cho món i
- `r_{v,i}`: Rating của user v cho món i
- `r̄_u`: Trung bình ratings của user u trên các món chung
- `r̄_v`: Trung bình ratings của user v trên các món chung

**Giải thích:**
- **Tử số:** Tổng tích của độ lệch ratings (so với trung bình) của 2 users trên các món chung
- **Mẫu số:** Tích của độ dài (L2 norm) của 2 vector đã mean-centered

### Công thức tương đương (dạng vector):

```
Sim(u, v) = (v_u' · v_v') / (||v_u'|| × ||v_v'||)
```

Trong đó:
- `v_u' = [r_{u,i} - r̄_u]_{i ∈ I}` (vector đã mean-centered của user u)
- `v_v' = [r_{v,i} - r̄_v]_{i ∈ I}` (vector đã mean-centered của user v)

---

## 2. Weighted Prediction Score

### Công thức dự đoán rating của user u cho món i:

```
pred(u, i) = (Σⱼ∈N sim(u, uⱼ) × rⱼᵢ) / (Σⱼ∈N sim(u, uⱼ))
```

### Trong đó:
- `u`: User cần dự đoán
- `i`: Món cần dự đoán rating
- `N`: Tập hợp các similar users đã đánh giá món i
- `uⱼ`: User j trong tập N
- `sim(u, uⱼ)`: Độ tương đồng giữa user u và user j (Mean-Centered Cosine Similarity)
- `rⱼᵢ`: Rating của user j cho món i

### Điều kiện:
- Chỉ tính với các user có `sim(u, uⱼ) > threshold` (threshold = 0.5)
- Nếu `Σⱼ∈N sim(u, uⱼ) = 0`, fallback về `avg_score` của món i

---

## 3. Tóm tắt quy trình CF

```
1. Tính similarity matrix: sim(uᵢ, uⱼ) cho mọi cặp users
2. Với user u cần recommend:
   a. Tìm top-k similar users: N = {uⱼ | sim(u, uⱼ) > 0.5}
   b. Lấy các món mà N đã đánh giá nhưng u chưa đánh giá
   c. Với mỗi món i:
      pred(u, i) = weighted average ratings từ N
   d. Sắp xếp theo pred(u, i) giảm dần
   e. Trả về top-k món
```

