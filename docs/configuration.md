# Configuration

All configuration is managed through environment variables, loaded from a `.env` file.

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key for ingredient extraction | `AIza...` |
| `NEO4J_PASSWORD` | Neo4j database password | `food-recsys-password` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `APP_ENV` | `development` | Environment (development/production) |
| `APP_DEBUG` | `0` | Enable debug mode (1=enabled) |
| `APP_HOST` | `0.0.0.0` | Server bind host |
| `APP_PORT` | `8000` | Server bind port |
| `MAX_CONTENT_LENGTH` | `16777216` | Max upload size in bytes (16MB) |
| `TEMP_UPLOAD_DIR` | `/tmp/food-recsys/uploads` | Temporary file storage path |

## Setup

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your values:
   ```env
   # Required
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Neo4j (matches docker-compose.yml defaults)
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=food-recsys-password
   
   # Server
   APP_ENV=development
   APP_DEBUG=1
   APP_HOST=0.0.0.0
   APP_PORT=8000
   ```

## Getting API Keys

### Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy the key to your `.env` file

**Note:** The free tier has rate limits. For production, consider upgrading to a paid plan.

## Docker Compose Configuration

The `docker-compose.yml` sets up Neo4j with these defaults:

| Setting | Value |
|---------|-------|
| Image | `neo4j:5.26-community` |
| HTTP Port | `7474` (Neo4j Browser) |
| Bolt Port | `7687` (Database connections) |
| Username | `neo4j` |
| Password | `food-recsys-password` |
| Plugins | APOC (enabled) |

### Customizing Neo4j

To change the password, update both:
1. `docker-compose.yml`: `NEO4J_AUTH=neo4j/your-new-password`
2. `.env`: `NEO4J_PASSWORD=your-new-password`

Then restart:
```bash
docker compose down -v  # Remove old data
docker compose up -d
python main.py --init-db  # Recreate constraints
```

## CLI Options

The server supports command-line overrides:

```bash
# Custom host and port
python main.py --host 127.0.0.1 --port 3000

# Enable auto-reload
python main.py --reload

# Initialize database only
python main.py --init-db
```

## File Upload Limits

Default max upload size is 16MB. To change:

```env
MAX_CONTENT_LENGTH=33554432  # 32MB
```

Supported image formats: `png`, `jpg`, `jpeg`, `webp`, `gif`
