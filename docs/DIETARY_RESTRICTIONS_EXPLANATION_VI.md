# Giải Thích: Dietary Restrictions và Suited Ingredients

## Tổng Quan

Hệ thống quản lý dietary restrictions (hạn chế ăn uống) để filter recommendations và đảm bảo an toàn cho users.

---

## Cấu Trúc Dietary Restriction

Mỗi dietary restriction có 2 loại ingredients:

### 1. `not_suited_ingredients` (KHÔNG phù hợp)
- **Mục đích**: Ingredients phải TRÁNH
- **Dùng để**: Filter OUT dishes (loại bỏ món có ingredients này)
- **Ví dụ**: 
  - `dairy-allergy`: ["milk", "cheese", "butter", ...]
  - `vegetarian`: ["beef", "pork", "chicken", "fish", ...]

### 2. `suited_ingredients` (PHÙ HỢP)
- **Mục đích**: Ingredients PHÙ HỢP, có thể dùng
- **Dùng để**: 
  - Gợi ý alternatives (thay thế)
  - Hiển thị thông tin cho user
  - **Có thể** boost recommendations (chưa implement)
- **Ví dụ**:
  - `dairy-allergy`: ["coconut milk", "almond milk", "soy milk", ...]
  - `vegetarian`: ["tofu", "tempeh", "seitan", "egg", "cheese", ...]

---

## Ví Dụ Cụ Thể

### Example 1: Dairy Allergy

```python
{
    "name": "dairy-allergy",
    "description": "Avoids all dairy products including milk, cheese, and butter",
    "not_suited_ingredients": [
        "milk",        # ❌ Phải tránh
        "cheese",      # ❌ Phải tránh
        "butter",      # ❌ Phải tránh
        "yogurt",      # ❌ Phải tránh
        ...
    ],
    "suited_ingredients": [
        "coconut milk",    # ✅ Có thể dùng thay cho milk
        "almond milk",    # ✅ Có thể dùng thay cho milk
        "soy milk",       # ✅ Có thể dùng thay cho milk
        "oat milk",       # ✅ Có thể dùng thay cho milk
        "coconut cream",  # ✅ Có thể dùng thay cho cream
    ],
}
```

**Ý nghĩa**:
- User bị dị ứng sữa → không thể ăn món có "milk", "cheese", "butter"
- Nhưng có thể dùng alternatives: "coconut milk", "almond milk", ...

### Example 2: Vegetarian

```python
{
    "name": "vegetarian",
    "description": "No meat or fish, but eggs and dairy are allowed",
    "not_suited_ingredients": [
        "beef",        # ❌ Thịt bò
        "pork",        # ❌ Thịt heo
        "chicken",     # ❌ Thịt gà
        "fish",        # ❌ Cá
        "shrimp",      # ❌ Tôm
        ...
    ],
    "suited_ingredients": [
        "tofu",        # ✅ Đậu phụ
        "tempeh",      # ✅ Tempeh
        "seitan",      # ✅ Seitan
        "egg",         # ✅ Trứng (vegetarian ăn được)
        "cheese",      # ✅ Phô mai (vegetarian ăn được)
        "milk",        # ✅ Sữa (vegetarian ăn được)
        "legume",      # ✅ Đậu
        "lentil",      # ✅ Đậu lăng
        "chickpea",    # ✅ Đậu gà
        "bean",        # ✅ Đậu
    ],
}
```

**Ý nghĩa**:
- Vegetarian không ăn thịt/cá → loại bỏ món có "beef", "pork", "fish", ...
- Nhưng có thể ăn: "tofu", "egg", "cheese", "legume", ...

---

## Cách Hệ Thống Sử Dụng

### 1. Filtering Logic (Hiện Tại)

**Code**: `filter_dishes_by_restrictions()` trong `neo4j_service.py`

```python
def filter_dishes_by_restrictions(
    self,
    dish_ids: list[str],
    restriction_names: list[str],
) -> list[str]:
    # Get ingredients that are NOT suited for the restrictions
    MATCH (bad_ing:Ingredient)-[:NOT_SUITED_FOR]->(r:DietaryRestriction)
    WHERE r.name IN $restriction_names
    WITH collect(DISTINCT bad_ing.name) AS bad_ingredients
    
    # Check each dish
    UNWIND $dish_ids AS did
    MATCH (d:Dish {dish_id: did})
    OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
    WITH d, collect(DISTINCT i.name) AS ingredients, bad_ingredients
    WHERE NONE(ing IN ingredients WHERE ing IN bad_ingredients)
    RETURN d.dish_id AS dish_id
```

**Hoạt động**:
1. Lấy tất cả `not_suited_ingredients` từ restrictions
2. Check từng dish: có chứa `not_suited_ingredients` không?
3. Nếu có → **LOẠI BỎ**
4. Nếu không → **GIỮ LẠI**

**Ví dụ**:
```python
# User có restriction: "dairy-allergy"
# Dish A: ["pasta", "tomato sauce", "cheese"] → ❌ LOẠI BỎ (có "cheese")
# Dish B: ["pasta", "tomato sauce", "basil"] → ✅ GIỮ LẠI (không có dairy)
```

### 2. Suited Ingredients (Hiện Tại - CHƯA DÙNG)

**Hiện tại**: `suited_ingredients` được lưu trong database (tạo `SUITED_FOR` relationships), nhưng **KHÔNG được dùng trong filtering logic**.

**Có thể dùng để**:

#### A. Suggest Alternatives (Gợi ý thay thế)
```python
# User muốn làm món cần "milk" nhưng bị dairy-allergy
# → Gợi ý: "Bạn có thể dùng 'coconut milk' hoặc 'almond milk' thay cho 'milk'"
```

#### B. Boost Recommendations (Ưu tiên món có suited ingredients)
```python
# Khi recommend, ưu tiên dishes có "coconut milk" thay vì chỉ filter out "milk"
# → Dishes với alternatives được boost lên top
```

#### C. Display Information (Hiển thị thông tin)
```python
# Hiển thị cho user: "Với dairy-allergy, bạn có thể dùng: coconut milk, almond milk, ..."
```

---

## Workflow Trong Recommendation

### Step 1: Get User Restrictions

```python
restrictions = neo4j.get_user_dietary_restrictions(user_id)
# → ["dairy-allergy", "vegetarian"]
```

### Step 2: Generate Recommendations (Normal)

```python
recommendations = recommend_content_based(user_id, k=10)
# → [dish1, dish2, dish3, ..., dish10]
```

### Step 3: Apply Dietary Filter

```python
if apply_dietary_filter:
    restrictions = neo4j.get_user_dietary_restrictions(user_id)
    if restrictions:
        candidate_dish_ids = [d["dish_id"] for d in recommendations]
        safe_dish_ids = neo4j.filter_dishes_by_restrictions(
            dish_ids=candidate_dish_ids,
            restriction_names=restrictions,
        )
        # → Chỉ giữ dishes KHÔNG có not_suited_ingredients
        recommendations = [d for d in recommendations if d["dish_id"] in safe_dish_ids]
```

**Kết quả**: Recommendations đã được filter, chỉ còn dishes an toàn cho user.

---

## Ví Dụ Thực Tế

### Scenario: User với Dairy Allergy

**User restrictions**: `["dairy-allergy"]`

**Dishes trong database**:
1. **Pasta Carbonara**: ["pasta", "eggs", "bacon", "parmesan cheese"]
   - ❌ **LOẠI BỎ**: Có "parmesan cheese" (not_suited)

2. **Pasta Marinara**: ["pasta", "tomato sauce", "basil", "garlic"]
   - ✅ **GIỮ LẠI**: Không có dairy ingredients

3. **Vegan Pasta**: ["pasta", "tomato sauce", "coconut milk", "basil"]
   - ✅ **GIỮ LẠI**: Không có dairy, có "coconut milk" (suited ingredient)

**Recommendations sau filter**:
- ✅ Pasta Marinara
- ✅ Vegan Pasta
- ❌ Pasta Carbonara (đã bị loại)

---

## Potential Improvements (Chưa Implement)

### 1. Boost Dishes với Suited Ingredients

```python
# Thay vì chỉ filter, có thể boost dishes có suited ingredients
def recommend_with_dietary_boost(user_id, k=10):
    recommendations = get_recommendations(user_id, k=k*2)
    
    restrictions = get_user_restrictions(user_id)
    suited_ingredients = get_suited_ingredients(restrictions)
    
    # Boost dishes có suited ingredients
    for rec in recommendations:
        if has_any_ingredient(rec, suited_ingredients):
            rec.score *= 1.2  # Boost 20%
    
    return sorted(recommendations, key=lambda x: x.score, reverse=True)[:k]
```

**Lợi ích**:
- Không chỉ filter, mà còn ưu tiên dishes tốt cho dietary restriction
- User thấy alternatives được recommend nhiều hơn

### 2. Suggest Alternatives

```python
def suggest_alternatives(ingredient, restriction):
    suited = get_suited_ingredients_for_restriction(restriction)
    if ingredient in get_not_suited_ingredients(restriction):
        return suited  # Return alternatives
    return []
```

**Ví dụ**:
```python
alternatives = suggest_alternatives("milk", "dairy-allergy")
# → ["coconut milk", "almond milk", "soy milk", "oat milk"]
```

### 3. Display Dietary Info

```python
def get_dietary_info(restriction):
    return {
        "avoid": get_not_suited_ingredients(restriction),
        "alternatives": get_suited_ingredients(restriction),
    }
```

**UI Display**:
```
Dietary Restriction: Dairy Allergy
❌ Avoid: milk, cheese, butter, yogurt, ...
✅ Alternatives: coconut milk, almond milk, soy milk, ...
```

---

## Database Schema

### Relationships

```
(Ingredient)-[:NOT_SUITED_FOR]->(DietaryRestriction)
(Ingredient)-[:SUITED_FOR]->(DietaryRestriction)
(User)-[:HAS_RESTRICTION]->(DietaryRestriction)
```

**Ví dụ**:
```
(milk)-[:NOT_SUITED_FOR]->(dairy-allergy)
(coconut_milk)-[:SUITED_FOR]->(dairy-allergy)
(user123)-[:HAS_RESTRICTION]->(dairy-allergy)
```

---

## Tóm Tắt

### `not_suited_ingredients`:
- ✅ **Đang dùng**: Filter OUT dishes
- ✅ **Hoạt động**: Loại bỏ món có ingredients này
- ✅ **Quan trọng**: Đảm bảo an toàn cho user

### `suited_ingredients`:
- ⚠️ **Chưa dùng**: Chỉ lưu trong database
- 💡 **Có thể dùng**:
  - Suggest alternatives
  - Boost recommendations
  - Display information
- 📝 **Future work**: Implement các tính năng trên

---

## Kết Luận

**`suited_ingredients`** hiện tại được định nghĩa và lưu trong database, nhưng **chưa được sử dụng trong filtering logic**.

**Mục đích chính**:
1. **Documentation**: Ghi lại alternatives cho mỗi restriction
2. **Future features**: Có thể dùng để boost recommendations, suggest alternatives
3. **User information**: Hiển thị cho user biết có thể dùng gì

**Hiện tại**: Hệ thống chỉ dùng `not_suited_ingredients` để filter, đảm bảo an toàn cho users với dietary restrictions.

