# Quick Start Guide

Get up and running with AI Cost Manager in 5 minutes!

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Set Up Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env if needed (optional for basic usage)
```

## 3. Initialize Database

```bash
python manage.py init-db
```

## 4. Add Fal.ai as a Source

```bash
python manage.py add-source --name "Fal.ai" --url "https://fal.ai/api/trpc/models.list" --extractor "fal"
```

## 5. Extract Model Data

```bash
python manage.py extract
```

This will fetch all models from Fal.ai and store them in the database.

## 6. View the Results

### Option A: Command Line

```bash
# List all sources
python manage.py list-sources

# List all models
python manage.py list-models

# Filter by type
python manage.py list-models --type text-to-image
```

### Option B: Web Interface

```bash
# Start the web server
python app.py
```

Then open your browser to: http://localhost:5000

## Available Commands

```bash
# Initialize database
python manage.py init-db

# Add a new source
python manage.py add-source --name "Name" --url "URL" --extractor "extractor_name"

# List all sources
python manage.py list-sources

# Extract data from all active sources
python manage.py extract

# Extract from specific source
python manage.py extract --source-id 1

# List all models
python manage.py list-models

# Filter models by source
python manage.py list-models --source-id 1

# Filter models by type
python manage.py list-models --type text-to-image

# List available extractors
python manage.py list-extractors
```

## Adding More Sources

To add support for other APIs (like OpenAI, Anthropic, etc.), you need to:

1. Create a new extractor in `extractors/` directory
2. Register it in `extractors/__init__.py`
3. Add the source using `python manage.py add-source`

See the README for more details on creating custom extractors.

## Example: Using the POST Endpoint

If you mentioned `POST https://fal.ai/explore/search`, you can create a custom extractor for it:

```python
# extractors/fal_search_extractor.py
from extractors.base import BaseExtractor

class FalSearchExtractor(BaseExtractor):
    def extract(self):
        # POST request with search parameters
        data = self.fetch_data(
            method='POST',
            json={
                'query': '',
                'limit': 100
            }
        )
        
        # Process and return results
        return [self._normalize_model(m) for m in data.get('results', [])]
```

Then register it and add as a source!

## Troubleshooting

**Database errors?**
```bash
# Delete and recreate the database
rm ai_costs.db
python manage.py init-db
```

**Import errors?**
Make sure you're in the project root directory and have installed all dependencies.

**Web interface not loading?**
Check that port 5000 is available. You can change it in `app.py` if needed.
