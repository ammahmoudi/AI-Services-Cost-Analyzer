# Schema Fetching Guide

## Overview

The AI Cost Manager now supports fetching OpenAPI schemas for AI models, which provides detailed information about:
- **Input parameters**: What inputs the model accepts
- **Output format**: What the model returns
- **Field types**: Data types, constraints, defaults
- **Required fields**: Which parameters are mandatory

## How to Use

### Basic Extraction (Fast - No Schemas)
```bash
python manage.py extract
```

### Full Extraction with Schemas (Slower but Complete)
```bash
python manage.py extract --fetch-schemas
```

## Schema API Endpoints

Fal.ai provides OpenAPI schemas at these endpoints:
- `https://fal.run/{model_id}/api/schema`
- `https://queue.fal.run/{model_id}/api/schema`

Example:
```
https://fal.run/fal-ai/flux/schnell/api/schema
```

## What Gets Extracted

### Input Schema
```json
{
  "inputs": [
    {
      "name": "prompt",
      "type": "string",
      "description": "The text prompt for image generation",
      "required": true
    },
    {
      "name": "image_size",
      "type": "string",
      "enum": ["square", "landscape", "portrait"],
      "default": "square",
      "required": false
    }
  ]
}
```

### Output Schema
```json
{
  "inputs": [
    {
      "name": "image",
      "type": "object",
      "description": "Generated image output",
      "required": true
    }
  ]
}
```

## Database Storage

Schemas are stored in the `AIModel` table:
- `input_schema`: JSON field with input parameters
- `output_schema`: JSON field with output format
- `raw_metadata['openapi_schema']`: Full OpenAPI specification

## Viewing Schemas

### Via Web Interface
1. Go to http://localhost:5000/models
2. Click on any model
3. Scroll to "Raw Metadata" section
4. Expand to see full schemas

### Via CLI
```bash
python -c "from ai_cost_manager import get_session, AIModel; s = get_session(); m = s.query(AIModel).first(); print(m.input_schema)"
```

### Via Python
```python
from ai_cost_manager import get_session, AIModel

session = get_session()
model = session.query(AIModel).filter_by(model_id='fal-ai/flux-pro').first()

print("Input Schema:")
for field in model.input_schema.get('inputs', []):
    print(f"  - {field['name']}: {field['type']} {'(required)' if field.get('required') else ''}")

print("\nOutput Schema:")
for field in model.output_schema.get('inputs', []):
    print(f"  - {field['name']}: {field['type']}")
```

## Performance Considerations

**Without Schemas:**
- Extracts ~880 models in ~30 seconds
- Only fetches model list and metadata

**With Schemas:**
- Extracts ~880 models in ~15-30 minutes
- Makes additional API call for each model
- More complete data but slower

**Recommendation:**
1. Initial run: Don't use `--fetch-schemas` (fast)
2. Periodic updates: Use `--fetch-schemas` once a week
3. Specific models: Fetch schemas only when needed

## Troubleshooting

### SSL Errors
If you see SSL errors when fetching schemas:
1. Check your internet connection
2. Try again later (API might be temporarily down)
3. Use extraction without schemas for now

### Empty Schemas
Some models might not have public schemas:
- The API endpoint might not exist for that model
- Model might be private/enterprise only
- Schema will be left empty (`{}`)

### Rate Limiting
If you hit rate limits:
- Add delays between requests
- Extract in smaller batches
- Contact fal.ai for API key with higher limits

## Custom Extractor Example

If you want to add schema fetching to other extractors:

```python
from extractors.base import BaseExtractor

class MyExtractor(BaseExtractor):
    def __init__(self, source_url: str, fetch_schemas: bool = False):
        super().__init__(source_url)
        self.fetch_schemas = fetch_schemas
    
    def extract(self):
        models = []
        for model_data in self.fetch_models():
            normalized = self.normalize(model_data)
            
            if self.fetch_schemas:
                schema = self.fetch_schema(model_data['id'])
                normalized['input_schema'] = self.parse_input(schema)
                normalized['output_schema'] = self.parse_output(schema)
            
            models.append(normalized)
        return models
```

## Future Enhancements

Potential improvements:
1. **Caching**: Cache schemas locally to avoid re-fetching
2. **Batch API**: Use batch endpoints if available
3. **Async Fetching**: Parallel schema fetching for speed
4. **Schema Diff**: Track schema changes over time
5. **Validation**: Validate input/output against schemas
