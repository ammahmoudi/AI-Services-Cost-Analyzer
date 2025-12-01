# AI Services Cost Analyzer

A comprehensive tool for tracking, analyzing, and comparing pricing across multiple AI service providers. Extract real-time costs from various sources, compare model pricing, and maintain a centralized database of AI service costs.

## Features

- **Multi-Source Cost Extraction**: Support for multiple AI providers including:
  - fal.ai (with authenticated playground pricing)
  - Together AI
  - AvalAI (Iranian AI aggregator with transparent pricing)
  - MetisAI (comprehensive Iranian AI platform)
  - Runware (multi-provider image/video generation API)
  - More sources coming soon

- **Intelligent LLM-Based Extraction**: Uses OpenRouter LLMs to intelligently parse and extract pricing data from various formats (HTML, JSON, Markdown)

- **Flexible Storage**: 
  - Hybrid cache system (database or file-based)
  - SQLite for development, PostgreSQL for production
  - Automatic data persistence and versioning

- **Authentication Management**: Secure storage of API keys and session cookies for authenticated endpoints

- **Web Interface**: Clean, intuitive UI for:
  - Adding and managing AI service sources
  - Configuring extractors and API keys
  - Viewing and comparing model costs
  - Managing cache and database

- **Docker-Ready**: Complete containerization with docker-compose for easy deployment

## Quick Start (Docker - Recommended)

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ai-costs
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start the stack**:
```bash
docker compose up -d
```

4. **Access the application**:
- Main app: http://localhost:5000
- pgAdmin (optional): http://localhost:5050
  ```bash
  docker compose --profile admin up -d
  ```

5. **Stop the stack**:
```bash
docker compose down
```

## Manual Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ (optional, SQLite used by default)

### Steps

1. **Create virtual environment**:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
playwright install chromium
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env as needed
```

4. **Initialize database**:
```bash
python -c "from ai_cost_manager.database import init_db; init_db()"
```

5. **Run the application**:
```bash
python app.py
```

6. **Open browser**: Navigate to http://localhost:5000

## Configuration

### Database Backend

**SQLite (Default)**:
```env
DATABASE_URL=sqlite:///ai_costs.db
```

**PostgreSQL**:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/ai_costs
```

### Cache Backend

**Database Cache (Recommended)**:
```env
CACHE_BACKEND=database
```

**File Cache (Development)**:
```env
CACHE_BACKEND=file
CACHE_DIR=cache
```

### Security

Generate a secure Flask secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Add to `.env`:
```env
FLASK_SECRET_KEY=<generated-key>
```

For complete configuration options, see [docs/CONFIGURATION.md](docs/CONFIGURATION.md)

## Usage

### Adding a New AI Service Source

1. Navigate to **Add Source** in the web interface
2. Fill in:
   - **Name**: Display name (e.g., "fal.ai")
   - **URL**: Documentation or pricing page URL
   - **Extractor**: Select from available extractors (fal, together, etc.)
3. Click **Add Source**

### Configuring Extractor API Keys

1. Go to **Settings** → **Extractor API Keys**
2. For Together AI:
   - Enter your API key
   - Toggle "Active" to enable
3. For fal.ai:
   - Enter your `wos-session` cookie (see help section for extraction steps)
   - Toggle "Active" to enable
4. Click **Save Changes**

### Extracting Model Costs

1. From the **Sources** page, click on a source
2. Click **Fetch Models** to scrape the source
3. For HTML sources, the system will:
   - Extract raw HTML
   - Generate a schema using LLM
   - Parse models based on the schema
4. View extracted models and their pricing

### Viewing Model Data

- **Models Page**: Browse all extracted models across sources
- Click on a model to view:
  - Detailed pricing information
  - Source URL
  - Cached data (raw, schema, playground)
  - Extraction timestamp

## Architecture

```
ai-costs/
├── ai_cost_manager/
│   ├── extractors/         # Source-specific extraction logic
│   │   ├── fal.py         # fal.ai extractor with auth
│   │   └── together.py    # Together AI extractor
│   ├── models.py          # SQLAlchemy database models
│   ├── database.py        # Database initialization and config
│   ├── cache.py           # Hybrid cache system
│   └── openrouter.py      # LLM integration for extraction
├── templates/             # Jinja2 HTML templates
├── static/               # CSS, JS, images
├── tests/                # Unit tests
├── docs/                 # Documentation
├── cache/                # File-based cache (if enabled)
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Multi-container orchestration
├── requirements.txt      # Python dependencies
├── app.py               # Flask application entry point
└── .env.example         # Environment template
```

## Default Sources

The system automatically seeds four default sources on first initialization:

1. **fal.ai** (`extractor: fal`)
   - URL: https://fal.ai/models
   - Supports authenticated playground pricing

2. **Together AI** (`extractor: together`)
   - URL: https://api.together.xyz/models
   - JSON API-based extraction

3. **AvalAI** (`extractor: avalai`)
   - URL: https://docs.avalai.ir/fa/pricing.md
   - Iranian AI aggregator with transparent pricing (100% aligned with base provider rates)
   - Supports OpenAI, Google, Anthropic, XAI, Meta, Mistral, Alibaba, DeepSeek, and more
   - New users get 20,000 IRR free credit

4. **MetisAI** (`extractor: metisai`)
   - URL: https://api.metisai.ir/api/v1/meta/providers/pricing
   - Iranian AI aggregator with comprehensive pricing API
   - Supports OpenAI, Google, Anthropic, Cohere, Meta, Mistral, DeepSeek, Grok, and more
   - Includes LLM, embedding, image generation, video, audio, and search models

5. **Runware** (`extractor: runware`)
   - URL: https://runware.ai/pricing
   - Multi-provider image and video generation API via WebSocket
   - Supports FLUX, Midjourney, Stable Diffusion, Sora, Kling, Google Veo, and 40+ more models
   - Includes background removal, upscaling, and video processing tools
   - Pricing for image generation (per image), video generation (per video/second), and tools

## Development

### Running Tests

```bash
pytest tests/
```

### Database Migrations

When modifying models, recreate the database:

```bash
# Backup first!
python -c "from ai_cost_manager.database import init_db; init_db()"
```

### Adding a New Extractor

1. Create `ai_cost_manager/extractors/your_source.py`
2. Implement the extractor class with required methods
3. Register in `ai_cost_manager/extractors/__init__.py`
4. Add configuration in Settings UI if needed

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for detailed development guides.

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///ai_costs.db` |
| `CACHE_BACKEND` | Cache storage type (database/file) | `database` |
| `CACHE_DIR` | Directory for file cache | `cache` |
| `FLASK_SECRET_KEY` | Flask session encryption key | (generated) |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM extraction | (none) |

## Troubleshooting

### "Playwright browser not found"

```bash
playwright install chromium
```

### "Database connection failed"

Check `DATABASE_URL` in `.env`:
- Ensure PostgreSQL is running (if used)
- Verify connection credentials
- For SQLite, check file permissions

### "Cache backend error"

Switch cache backend in `.env`:
```env
CACHE_BACKEND=file  # or database
```

For more troubleshooting, see [docs/CONFIGURATION.md](docs/CONFIGURATION.md#troubleshooting)

## Docker Services

The docker-compose stack includes:

- **postgres**: PostgreSQL 15 database with persistent storage
- **app**: Flask application with Playwright browser
- **pgadmin** (optional): Database administration tool

### Docker Commands

```bash
# Build and start
docker compose up -d

# View logs
docker compose logs -f app

# Access database
docker compose exec postgres psql -U postgres -d ai_costs

# Restart services
docker compose restart

# Clean shutdown
docker compose down

# With pgAdmin
docker compose --profile admin up -d
```

## Performance Considerations

- **Database Cache**: Recommended for production, supports concurrent access
- **File Cache**: Useful for debugging, not ideal for multi-user scenarios
- **PostgreSQL**: Better for >10k models, supports connection pooling
- **SQLite**: Simple setup, good for <10k models

## Security Notes

- Never commit `.env` files with real credentials
- Generate unique `FLASK_SECRET_KEY` for each deployment
- Use strong PostgreSQL passwords in production
- fal.ai cookies expire; rotate them periodically
- API keys stored encrypted in database

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License



## Acknowledgments

- Built with Flask, SQLAlchemy, and Playwright
- LLM extraction powered by OpenRouter
- UI inspired by modern dashboard designs

## Support

For issues, questions, or contributions, please open an issue on GitHub.
