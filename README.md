# AI Cost Manager

A dynamic system to manage and track AI model API costs from various sources.

## Features

- 🔌 **Modular Extractor System**: Easy to add new API sources
- 💰 **Cost Tracking**: Track pricing for AI models across platforms
- 📋 **Schema Fetching**: Extract OpenAPI schemas for input/output definitions
- 🔄 **Auto-Update**: Automatically fetch latest pricing data
- 🎯 **Extensible**: Simple interface to add custom extractors
- 🔍 **Search & Filter**: Find models by type, name, or tags
- 📊 **Web Dashboard**: Beautiful UI to browse and compare models

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment:
```bash
cp .env.example .env
```

3. Initialize database:
```bash
python manage.py init-db
```

4. Add a source:
```bash
python manage.py add-source --name "Fal.ai" --url "https://fal.ai/api/trpc/models.list" --extractor "fal"
```

5. Extract data:
```bash
# Fast extraction (metadata only)
python manage.py extract

# Full extraction with OpenAPI schemas (slower but complete)
python manage.py extract --fetch-schemas
```

6. Run the web interface:
```bash
python app.py
```

## Supported Sources

- **Fal.ai**: Text-to-image, video generation, and more

## Adding Custom Extractors

Create a new file in `extractors/` directory:

```python
from extractors.base import BaseExtractor

class MyCustomExtractor(BaseExtractor):
    def extract(self, response_data):
        # Your extraction logic here
        pass
```

Register it in `extractors/__init__.py`.
