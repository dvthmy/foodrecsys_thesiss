# Giải Thích Chi Tiết: Gemma Extractor - Toàn Bộ Code

## Tổng Quan

File `gemma_extractor.py` chứa class `GemmaExtractor` - service để trích xuất nguyên liệu từ mô tả văn bản sử dụng Google's Gemma 3 1B Instruct model.

**Mục đích chính**: Text description → Structured JSON với ingredients, dish name, cuisine

---

## Phần 1: Imports và Setup

```python
import json
import logging
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
```

**Giải thích**:
- `json`: Parse JSON response từ model
- `logging`: Logging cho debugging
- `re`: Regular expressions để clean text
- `typing.Any`: Type hints (vì transformers typing stubs thay đổi theo version)
- `torch`: PyTorch cho tensor operations
- `transformers`: Hugging Face library cho model loading

---

## Phần 2: Class Definition và Constants

### Class: GemmaExtractor

```python
class GemmaExtractor:
    """Service for extracting ingredients from text descriptions using local Gemma model."""
```

### Constants

#### 1. MODEL_ID
```python
MODEL_ID = "google/gemma-3-1b-it"
```

**Giải thích**:
- Model ID từ Hugging Face
- `gemma-3-1b-it`: Gemma 3, 1B parameters, instruction-tuned
- Instruction-tuned: Được fine-tune để follow instructions tốt hơn

#### 2. TEXT_PROMPT
```python
TEXT_PROMPT = """Extract all ingredients mentioned in this food dish description.

Description: {description}

Return a JSON object with the following structure:
{{
    "dish_name": "Name of the dish if mentioned, otherwise 'Unknown Dish'",
    "ingredients": ["ingredient1", "ingredient2", ...],
    "cuisine": "Type of cuisine if identifiable, otherwise null",
    "confidence": "high" | "medium" | "low"
}}

Guidelines:
- Extract all explicitly mentioned ingredients
- Infer common ingredients if the dish type is mentioned (e.g., "pizza" implies dough, tomato sauce, cheese)
- Use lowercase for ingredient names
- Be specific with ingredient names

Respond with ONLY the JSON object, no additional text."""
```

**Giải thích**:
- **Prompt Engineering**: Template để guide model
- `{description}`: Placeholder sẽ được thay thế
- `{{` và `}}`: Double braces để escape trong f-string
- **Guidelines**: Hướng dẫn model cách extract:
  - Extract tất cả ingredients được mention
  - Infer ingredients từ dish type (ví dụ: "pizza" → dough, tomato sauce, cheese)
  - Lowercase cho consistency
  - Specific names (không generic như "spices")

**Tại sao quan trọng**:
- Prompt tốt → Output tốt
- Structured format (JSON) → Dễ parse
- Clear guidelines → Consistent results

---

## Phần 3: Initialization (`__init__`)

```python
def __init__(
    self,
    model_id: str | None = None,
    device: str | None = None,
    torch_dtype: torch.dtype | None = None,
    preload: bool = False,
):
```

### Parameters

1. **`model_id`** (str | None):
   - Hugging Face model ID
   - Default: `None` → dùng `MODEL_ID` constant
   - Có thể override để dùng model khác

2. **`device`** (str | None):
   - Device để chạy model: `'cuda'`, `'cpu'`, `'mps'`, hoặc `'auto'`
   - Default: `None` → auto-detect bằng `_get_default_device()`

3. **`torch_dtype`** (torch.dtype | None):
   - Data type cho model weights
   - Default: `None` → auto-select:
     - GPU/MPS: `bfloat16` (tiết kiệm memory, nhanh hơn)
     - CPU: `float32` (chính xác hơn)

4. **`preload`** (bool):
   - Có load model ngay khi init không?
   - Default: `False` (lazy loading)
   - `True`: Load ngay (dùng khi muốn preload ở startup)

### Code Flow

```python
# 1. Set model ID
self._model_id = model_id or self.MODEL_ID

# 2. Auto-detect device
self._device = device or self._get_default_device()

# 3. Auto-select dtype
self._torch_dtype = torch_dtype or self._get_default_dtype()

# 4. Initialize as None (lazy loading)
self._model: Any | None = None
self._tokenizer: Any | None = None
self._pipeline: Any = None

# 5. Log initialization
logger.info("Initializing GemmaExtractor with model=%s, device=%s, dtype=%s", ...)

# 6. Preload nếu được yêu cầu
if preload:
    self._load_model()
```

**Lưu ý**:
- Model chưa được load ngay (lazy loading)
- Chỉ load khi thực sự cần dùng
- Tiết kiệm memory và startup time

---

## Phần 4: Helper Methods

### 1. `_get_default_device() -> str`

```python
def _get_default_device(self) -> str:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"           # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return "mps"            # Apple Silicon (M1/M2/M3)
    return "cpu"                 # Fallback
```

**Mục đích**: Tự động chọn device tốt nhất

**Priority**:
1. CUDA (NVIDIA GPU) - nhanh nhất
2. MPS (Apple Silicon) - tốt cho Mac
3. CPU - fallback

**Tại sao cần**:
- GPU nhanh hơn CPU rất nhiều (10-100x)
- Tự động chọn → không cần config thủ công

---

### 2. `_get_default_dtype() -> torch.dtype`

```python
def _get_default_dtype(self) -> torch.dtype:
    """Determine the best dtype based on device."""
    if self._device in ("cuda", "mps"):
        return torch.bfloat16    # 16-bit, nhanh hơn, tiết kiệm memory
    return torch.float32         # 32-bit, chính xác hơn (CPU)
```

**Giải thích**:
- **bfloat16** (GPU/MPS):
  - 16-bit floating point
  - Tiết kiệm memory (50% so với float32)
  - Nhanh hơn trên GPU
  - Đủ chính xác cho inference

- **float32** (CPU):
  - 32-bit floating point
  - Chính xác hơn
  - CPU không optimize tốt cho bfloat16

**Trade-off**:
- bfloat16: Nhanh hơn, ít memory, đủ chính xác
- float32: Chính xác hơn, nhiều memory, chậm hơn

---

## Phần 5: Model Loading (`_load_model`)

```python
def _load_model(self) -> None:
    """Load the model and tokenizer lazily."""
    if self._model is not None:
        return  # Đã load rồi → return ngay
```

**Early return**: Tránh load nhiều lần

### Step 1: Load Tokenizer

```python
logger.info("Loading Gemma model: %s", self._model_id)
self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
```

**Tokenizer làm gì**:
- Convert text → token IDs
- Handle special tokens (BOS, EOS, PAD)
- Truncate/pad sequences

**Ví dụ**:
```python
text = "Extract ingredients from pasta"
tokens = tokenizer(text)
# → [101, 2345, 6789, ..., 102]  # Token IDs
```

### Step 2: Setup Device Map

```python
device_map: str | None
if self._device == "cuda":
    device_map = "auto"  # Hugging Face tự động phân bổ layers
else:
    device_map = None    # Manual move sau
```

**Giải thích**:
- `device_map="auto"`: Hugging Face tự động phân bổ model layers lên GPU
  - Hữu ích khi model lớn, có thể split across multiple GPUs
- `None`: Sẽ move model manually sau

### Step 3: Load Model

```python
self._model = AutoModelForCausalLM.from_pretrained(
    self._model_id,
    torch_dtype=self._torch_dtype,  # bfloat16 hoặc float32
    device_map=device_map,          # "auto" hoặc None
)
```

**AutoModelForCausalLM**:
- Causal Language Model (decoder-only)
- Generate text từ trái sang phải
- Phù hợp cho text generation tasks

**Parameters**:
- `torch_dtype`: Data type cho weights
- `device_map`: Auto-distribute hoặc None

### Step 4: Manual Device Move (nếu cần)

```python
if self._device in ("cpu", "mps"):
    self._model = self._model.to(self._device)
```

**Tại sao cần**:
- `device_map="auto"` chỉ work với CUDA
- CPU/MPS cần move manually

### Step 5: Create Pipeline

```python
self._pipeline = pipeline(
    "text-generation",
    model=self._model,
    tokenizer=self._tokenizer,
    device_map=device_map,
)
```

**Pipeline**:
- High-level API từ Hugging Face
- Wrapper cho model + tokenizer
- Dễ sử dụng hơn raw model

**"text-generation"**:
- Task type: Generate text
- Tự động handle:
  - Tokenization
  - Model forward pass
  - Decoding tokens → text

### Step 6: Log Success

```python
logger.info("Gemma model loaded successfully on device: %s", self._device)
```

---

## Phần 6: Properties (Getters)

### 1. `@property model(self) -> Any`

```python
@property
def model(self) -> Any:
    """Get or create the Gemma model instance."""
    self._load_model()      # Load nếu chưa có
    assert self._model is not None  # Type safety
    return self._model
```

**Mục đích**: Lazy loading getter

**Usage**:
```python
extractor = GemmaExtractor()
model = extractor.model  # Model tự động load ở đây
```

**Lợi ích**:
- Không cần gọi `_load_model()` thủ công
- Model chỉ load khi thực sự cần

---

### 2. `@property tokenizer(self) -> Any`

```python
@property
def tokenizer(self) -> Any:
    """Get or create the tokenizer instance."""
    self._load_model()      # Load nếu chưa có
    assert self._tokenizer is not None
    return self._tokenizer
```

**Tương tự như `model` property**

---

## Phần 7: JSON Parsing (`_parse_json_response`)

```python
def _parse_json_response(self, text: str) -> dict[str, Any]:
    """Parse JSON from model response, handling markdown code blocks."""
```

**Mục đích**: Parse JSON từ model response, handle các edge cases

### Step 1: Remove Markdown Code Blocks

```python
text = text.strip()
if text.startswith("```"):
    # Remove ```json or ``` at start
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    # Remove ``` at end
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
```

**Vấn đề**: Model có thể trả về JSON trong markdown code blocks:
```
```json
{"ingredients": ["pasta", "tomato"]}
```
```

**Giải pháp**: Remove markdown markers

**Regex giải thích**:
- `^```(?:json)?\s*\n?`: Match ` ```json` hoặc ` ```` ở đầu
- `\n?```\s*$`: Match ` ```` ở cuối

### Step 2: Extract JSON Object

```python
json_match = re.search(r"\{.*\}", text, re.DOTALL)
if json_match:
    text = json_match.group(0)
```

**Vấn đề**: Model có thể thêm text thừa:
```
Here is the JSON: {"ingredients": ["pasta"]} That's all!
```

**Giải pháp**: Tìm JSON object đầu tiên (từ `{` đến `}`)

**Regex**:
- `\{.*\}`: Match từ `{` đến `}`
- `re.DOTALL`: `.` match cả newline

### Step 3: Parse JSON

```python
try:
    return json.loads(text)
except json.JSONDecodeError as e:
    logger.error("Failed to parse model response as JSON: %s", e)
    logger.error("Full raw response:\n%s", text)
    raise ValueError(
        f"Failed to parse model response as JSON: {e}\nResponse text: {text[:200]}"
    )
```

**Error handling**:
- Log error để debug
- Log full response
- Raise ValueError với context

**Tại sao quan trọng**:
- Model có thể generate invalid JSON
- Cần handle gracefully
- Provide context để debug

---

## Phần 8: Main Extraction Method (`extract_from_description`)

```python
def extract_from_description(
    self,
    description: str,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
```

**Mục đích**: Extract ingredients từ text description

### Parameters

1. **`description`** (str):
   - Text mô tả món ăn
   - Ví dụ: "Italian pasta with tomato sauce and basil"

2. **`max_new_tokens`** (int):
   - Số tokens tối đa để generate
   - Default: 512
   - Càng nhiều → response càng dài (nhưng chậm hơn)

### Step 1: Validate Input

```python
if not description or not description.strip():
    raise ValueError("Description cannot be empty")
```

**Validation**: Đảm bảo input không rỗng

### Step 2: Load Model

```python
self._load_model()
assert self._pipeline is not None
assert self._tokenizer is not None
```

**Lazy loading**: Model chỉ load khi cần

### Step 3: Format Prompt

```python
prompt = self.TEXT_PROMPT.format(description=description)
logger.info("Sending prompt to Gemma for description: %s...", description[:100])
```

**Prompt engineering**: Format template với description

**Ví dụ**:
```
Extract all ingredients mentioned in this food dish description.

Description: Italian pasta with tomato sauce and basil

Return a JSON object with the following structure:
...
```

### Step 4: Format as Chat Messages

```python
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": "{"}
]
```

**Chat format**: Gemma 3 là instruction-tuned model, expect chat format

**Trick**: `"content": "{"` - Start với `{` để guide model generate JSON

**Tại sao**:
- Model sẽ tiếp tục từ `{`
- Tăng khả năng generate valid JSON

### Step 5: Generate Response

```python
outputs = self._pipeline(
    messages,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
)
```

**Parameters giải thích**:

1. **`messages`**: Chat messages format
2. **`max_new_tokens`**: Max tokens to generate
3. **`do_sample=True`**: Sampling thay vì greedy decoding
   - Greedy: Luôn chọn token có probability cao nhất
   - Sampling: Chọn ngẫu nhiên theo distribution
   - Sampling → đa dạng hơn, tự nhiên hơn
4. **`temperature=0.7`**: Control randomness
   - 0.0: Deterministic (greedy)
   - 1.0: Very random
   - 0.7: Balance giữa quality và diversity
5. **`top_p=0.9`**: Nucleus sampling
   - Chỉ sample từ top 90% probability mass
   - Loại bỏ tokens có probability quá thấp
6. **`pad_token_id`**: Token để pad sequences
   - Lấy từ tokenizer, fallback None nếu không có

### Step 6: Extract Generated Text

```python
response_text = outputs[0]["generated_text"]
```

**Pipeline output**:
- List of outputs
- `outputs[0]`: First (và thường là duy nhất) output
- `["generated_text"]`: Generated text string

### Step 7: Handle List Response

```python
if isinstance(response_text, list):
    # Find the assistant's response
    for msg in response_text:
        if msg.get("role") == "assistant":
            response_text = msg.get("content", "")
            break
    else:
        # If no assistant message, use the last message content
        response_text = response_text[-1].get("content", "") if response_text else ""
```

**Edge case**: Pipeline có thể trả về list of messages

**Giải pháp**:
- Tìm message có `role == "assistant"`
- Nếu không có → dùng message cuối cùng

### Step 8: Log Raw Response

```python
logger.info("Gemma raw response text:\n%s", response_text)
```

**Debugging**: Log để kiểm tra model output

### Step 9: Parse JSON

```python
result = self._parse_json_response(response_text)
```

**Parse**: Convert text → dict

### Step 10: Validate Result

```python
if not isinstance(result, dict):
    raise ValueError(
        f"Expected dictionary from Gemma, got {type(result).__name__}"
    )
```

**Type safety**: Đảm bảo result là dict

### Step 11: Extract Fields Safely

```python
dish_name = (
    result.get("dish_name")
    if isinstance(result.get("dish_name"), str)
    else "Unknown Dish"
)
ingredients = (
    result.get("ingredients")
    if isinstance(result.get("ingredients"), list)
    else []
)
cuisine = (
    result.get("cuisine") if isinstance(result.get("cuisine"), str) else None
)
confidence = (
    result.get("confidence")
    if isinstance(result.get("confidence"), str)
    else "medium"
)
```

**Safe access**: 
- Dùng `.get()` để tránh KeyError
- Type checking để đảm bảo đúng type
- Default values nếu thiếu hoặc sai type

**Tại sao cần**:
- Model có thể generate fields sai type
- Model có thể thiếu fields
- Cần handle gracefully

### Step 12: Return Structured Result

```python
return {
    "dish_name": dish_name or "Unknown Dish",
    "ingredients": ingredients,
    "cuisine": cuisine,
    "confidence": confidence,
    "source": "description",
}
```

**Structured output**: Luôn return cùng format

**Fields**:
- `dish_name`: Tên món ăn
- `ingredients`: List ingredients
- `cuisine`: Loại ẩm thực
- `confidence`: Độ tin cậy
- `source`: Luôn "description" (để track source)

---

## Phần 9: Singleton Pattern

### Global Variable

```python
_gemma_extractor: GemmaExtractor | None = None
```

**Singleton instance**: Chỉ 1 instance trong toàn bộ app

### Factory Function

```python
def get_gemma_extractor(
    model_id: str | None = None,
    device: str | None = None,
    preload: bool = False,
) -> GemmaExtractor:
```

**Mục đích**: Get hoặc create singleton instance

### Implementation

```python
global _gemma_extractor
if _gemma_extractor is None:
    _gemma_extractor = GemmaExtractor(model_id=model_id, device=device, preload=preload)
elif preload:
    _gemma_extractor._load_model()
return _gemma_extractor
```

**Logic**:
1. Nếu chưa có instance → tạo mới
2. Nếu đã có và `preload=True` → load model
3. Return instance

**Lợi ích**:
- **Memory efficient**: Chỉ load model 1 lần
- **Fast**: Reuse model đã load
- **Consistent**: Cùng config cho toàn bộ app

**Usage**:
```python
# Lần 1: Tạo mới
extractor1 = get_gemma_extractor()

# Lần 2: Reuse
extractor2 = get_gemma_extractor()

# extractor1 và extractor2 là cùng 1 object!
assert extractor1 is extractor2  # True
```

---

## Workflow Tổng Quan

### Khi extract ingredients:

```
1. User gọi: extractor.extract_from_description("Italian pasta with tomato")
   ↓
2. Validate input (không rỗng)
   ↓
3. Load model (nếu chưa load)
   ├── Load tokenizer
   ├── Load model
   └── Create pipeline
   ↓
4. Format prompt với description
   ↓
5. Format as chat messages (user + assistant start với "{")
   ↓
6. Generate response với pipeline
   ├── Tokenize input
   ├── Forward pass qua model
   ├── Decode tokens → text
   └── Return generated text
   ↓
7. Extract generated text từ output
   ↓
8. Parse JSON (remove markdown, extract JSON object)
   ↓
9. Validate và extract fields safely
   ↓
10. Return structured dict
    {
        "dish_name": "Italian Pasta",
        "ingredients": ["pasta", "tomato", "basil"],
        "cuisine": "Italian",
        "confidence": "high",
        "source": "description"
    }
```

---

## Best Practices

### 1. **Lazy Loading**
```python
# ✅ Tốt: Model chỉ load khi cần
extractor = GemmaExtractor()
result = extractor.extract_from_description("pasta")  # Model load ở đây

# ❌ Không cần: Load ngay khi init
extractor = GemmaExtractor()
extractor._load_model()  # Không cần thiết
```

### 2. **Singleton Pattern**
```python
# ✅ Tốt: Dùng get_gemma_extractor()
extractor = get_gemma_extractor()

# ❌ Không tốt: Tạo nhiều instances
extractor1 = GemmaExtractor()
extractor2 = GemmaExtractor()  # Load model 2 lần!
```

### 3. **Error Handling**
```python
# ✅ Tốt: Handle exceptions
try:
    result = extractor.extract_from_description("pasta")
except ValueError as e:
    print(f"Extraction error: {e}")
```

### 4. **Preload ở Startup**
```python
# ✅ Tốt: Preload ở app startup
extractor = get_gemma_extractor(preload=True)
# Model đã sẵn sàng, không delay ở lần gọi đầu tiên
```

---

## So Sánh với CLIP và EmbeddingGemma

| Đặc điểm | GemmaExtractor | CLIPEmbedder | IngredientEmbedder |
|----------|----------------|--------------|-------------------|
| **Mục đích** | Extract (text → JSON) | Embed (image → vector) | Embed (text → vector) |
| **Input** | Text description | Image file | Ingredient name |
| **Output** | Structured dict | 512-dim vector | 512-dim vector |
| **Model type** | Generative LM | Vision-Language | Embedding model |
| **Use case** | Parse ingredients | Visual similarity | Semantic similarity |

---

## Kết Luận

`GemmaExtractor` là một service hoàn chỉnh để:
- ✅ Extract ingredients từ text descriptions
- ✅ Parse JSON response từ model
- ✅ Handle edge cases (markdown, invalid JSON)
- ✅ Lazy loading cho memory efficiency
- ✅ Singleton pattern cho consistency
- ✅ Error handling và validation
- ✅ Structured output format

Mỗi phần của code đều có vai trò quan trọng và được thiết kế để robust và dễ sử dụng!

