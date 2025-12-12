# API Reference

Base URL: `http://localhost:8000`

Interactive documentation available at `/docs` (Swagger UI) or `/redoc`.

## Endpoints

### Health Check

```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "food-recsys"
}
```

---

### Upload Dish Images

```http
POST /api/v1/dishes/upload
Content-Type: multipart/form-data
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `images` | file[] | Yes | One or more image files (png, jpg, jpeg, webp, gif) |
| `descriptions` | string[] | No | Descriptions for each image (matching order) |

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/dishes/upload" \
  -F "images=@pizza.jpg" \
  -F "images=@sushi.jpg" \
  -F "descriptions=Margherita pizza" \
  -F "descriptions=Salmon nigiri"
```

**Response:**
```json
{
  "uploaded": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "original_name": "pizza.jpg",
      "temp_path": "/tmp/food-recsys/uploads/550e8400-e29b-41d4-a716-446655440000.jpg",
      "description": "Margherita pizza"
    }
  ],
  "errors": [],
  "total": 2,
  "failed": 0
}
```

---

### Process Uploaded Images

```http
POST /api/v1/dishes/process
Content-Type: application/json
```

**Request Body:**
```json
{
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "temp_path": "/tmp/food-recsys/uploads/550e8400.jpg",
      "description": "Optional description"
    }
  ]
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "660e8400-e29b-41d4-a716-446655440001",
  "status": "processing",
  "total_items": 1,
  "status_url": "/api/v1/jobs/660e8400-e29b-41d4-a716-446655440001/status"
}
```

---

### Upload and Process (Combined)

```http
POST /api/v1/dishes/upload-and-process
Content-Type: multipart/form-data
```

Combines upload and process in a single request.

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/dishes/upload-and-process" \
  -F "images=@dish.jpg" \
  -F "descriptions=Homemade pasta with tomato sauce"
```

**Response (202 Accepted):**
```json
{
  "job_id": "770e8400-e29b-41d4-a716-446655440002",
  "status": "processing",
  "total_items": 1,
  "upload_errors": null,
  "status_url": "/api/v1/jobs/770e8400-e29b-41d4-a716-446655440002/status"
}
```

---

### Get Job Status

```http
GET /api/v1/jobs/{job_id}/status
```

Poll this endpoint to track processing progress.

**Response:**
```json
{
  "job_id": "770e8400-e29b-41d4-a716-446655440002",
  "status": "completed",
  "progress": 100.0,
  "total": 1,
  "completed": 1,
  "failed": 0,
  "created_at": "2025-12-12T10:00:00",
  "started_at": "2025-12-12T10:00:01",
  "finished_at": "2025-12-12T10:00:05",
  "results": {
    "550e8400-e29b-41d4-a716-446655440000": {
      "item_id": "550e8400-e29b-41d4-a716-446655440000",
      "success": true,
      "dish_id": "880e8400-e29b-41d4-a716-446655440003",
      "dish_name": "Pasta with Tomato Sauce",
      "ingredients": ["pasta", "tomato sauce", "olive oil", "garlic", "basil"],
      "error": null
    }
  }
}
```

**Status Values:**
| Status | Description |
|--------|-------------|
| `pending` | Job created, not yet started |
| `processing` | Currently processing items |
| `completed` | All items processed successfully |
| `failed` | All items failed |
| `partial` | Some items succeeded, some failed |

---

### Get Dish by ID

```http
GET /api/v1/dishes/{dish_id}
```

**Response:**
```json
{
  "dish_id": "880e8400-e29b-41d4-a716-446655440003",
  "name": "Pasta with Tomato Sauce",
  "description": "Homemade pasta with tomato sauce",
  "image_url": "/tmp/food-recsys/uploads/550e8400.jpg",
  "ingredients": ["pasta", "tomato sauce", "olive oil", "garlic", "basil"],
  "country": "Italian"
}
```

---

### List All Ingredients

```http
GET /api/v1/ingredients
```

**Response:**
```json
{
  "ingredients": ["basil", "garlic", "olive oil", "pasta", "tomato sauce"],
  "count": 5
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": {
    "error": "Error message",
    "code": "ERROR_CODE"
  }
}
```

**Common Error Codes:**
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `NO_IMAGES` | 400 | No images provided in request |
| `EMPTY_FILES` | 400 | Files array is empty |
| `NO_VALID_FILES` | 400 | No files with valid extensions |
| `EMPTY_ITEMS` | 400 | Items array is empty |
| `NO_VALID_ITEMS` | 400 | No items with existing temp files |
| `JOB_NOT_FOUND` | 404 | Job ID does not exist |
| `DISH_NOT_FOUND` | 404 | Dish ID does not exist |
