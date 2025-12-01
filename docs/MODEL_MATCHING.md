# Model Matching Feature

## Overview

The Model Matching feature identifies equivalent AI models across different providers, enabling you to:
- **Compare prices** for the same model on different platforms
- **Find alternatives** to save costs
- **View unified models** grouped by canonical name instead of provider

## How It Works

### 1. LLM-Based Matching

The system uses GPT-4o-mini to intelligently identify equivalent models by analyzing:
- Model names (e.g., "FLUX.1 Dev" = "flux-dev-1.0")
- Provider information
- Model descriptions
- Tags and categories
- Version indicators

### 2. Confidence Scoring

Each match is assigned a confidence score (0.0 - 1.0) indicating how certain the system is that the models are equivalent:
- **0.9 - 1.0**: Very high confidence (same model, minor naming differences)
- **0.7 - 0.9**: High confidence (likely same model)
- **0.5 - 0.7**: Medium confidence (similar models)
- **< 0.5**: Low confidence (may not be exact match)

### 3. Canonical Models

Each group of equivalent models gets a **canonical model** which represents the unified identity:
- `canonical_name`: Normalized identifier (e.g., "flux-dev-1.0")
- `display_name`: Human-readable name (e.g., "FLUX.1 Dev")
- Provider-agnostic metadata

## Usage

### Web Interface

#### Run Model Matching

1. Navigate to **🔄 Unified View** in the navigation
2. Click **🤖 Run Model Matching** button
3. Wait for the LLM to analyze all models
4. View results in the unified view

#### View Model Alternatives

1. Go to any model's detail page
2. Scroll to **🔄 Alternative Providers** section
3. See price comparisons and potential savings
4. Click **View Details** to switch to a different provider

#### Unified View

- Shows all unique models grouped by canonical name
- Displays price ranges across providers
- Highlights the cheapest option
- Filter by model type
- Expand to see all providers

### API Endpoints

#### `POST /api/match-models`

Run model matching on all models.

**Request Body:**
```json
{
  "force_refresh": false
}
```

**Response:**
```json
{
  "status": "success",
  "canonical_models_created": 45,
  "model_matches_created": 120,
  "total_models_processed": 120,
  "match_groups": 45
}
```

#### `GET /api/models/{id}/alternatives`

Get alternative providers for a specific model.

**Response:**
```json
{
  "model_id": 123,
  "alternatives": [
    {
      "model_id": 456,
      "name": "FLUX.1 Dev",
      "provider": "replicate",
      "cost_per_call": 0.003,
      "confidence": 0.95
    }
  ]
}
```

#### `GET /api/models/{id}/with-alternatives`

Get a model with all alternatives and pricing comparison.

**Response:**
```json
{
  "model": {
    "id": 123,
    "name": "flux-dev-1.0",
    "provider": "fal",
    "cost_per_call": 0.005
  },
  "alternatives": [...],
  "best_alternative": {...},
  "potential_savings": 0.002
}
```

#### `GET /api/canonical-models`

Get all canonical models with provider details.

**Query Parameters:**
- `model_type`: Filter by type (image, text, video, audio)

**Response:**
```json
{
  "total": 45,
  "models": [
    {
      "canonical_id": 1,
      "canonical_name": "flux-dev-1.0",
      "display_name": "FLUX.1 Dev",
      "model_type": "image",
      "provider_count": 3,
      "providers": [...],
      "best_price": {...}
    }
  ]
}
```

#### `GET /api/canonical-models/{id}`

Get details of a specific canonical model.

### Command Line

```bash
# Run matching
python run_matching.py

# Force re-match
python run_matching.py --force
```

## Database Schema

### CanonicalModel Table

Stores unified model identities:
- `id`: Primary key
- `canonical_name`: Normalized identifier
- `display_name`: Human-readable name
- `description`: Model description
- `model_type`: image, text, video, audio
- `tags`: List of tags

### ModelMatch Table

Links AIModel instances to canonical models:
- `id`: Primary key
- `canonical_model_id`: Reference to canonical model
- `ai_model_id`: Reference to specific provider's model
- `confidence`: Match confidence (0.0 - 1.0)
- `matched_by`: 'llm', 'manual', or 'rule'
- `matched_at`: Timestamp

## Examples

### Example 1: FLUX.1 Dev

**Provider Models:**
- FAL: "flux-dev-1.0" @ $0.005
- Replicate: "FLUX.1 Dev" @ $0.003
- Runware: "bfl:4@1" @ $0.004

**Canonical Model:**
- Name: "flux-dev-1.0"
- Display: "FLUX.1 Dev"
- Best Price: Replicate @ $0.003
- Savings: Up to $0.002 per call

### Example 2: Stable Diffusion XL

**Provider Models:**
- Replicate: "sdxl-1.0" @ $0.002
- FAL: "stable-diffusion-xl" @ $0.003
- Runware: "stabilityai:1@0" @ $0.0025

**Canonical Model:**
- Name: "sdxl-1.0"
- Display: "Stable Diffusion XL"
- Best Price: Replicate @ $0.002
- Savings: Up to $0.001 per call

## Configuration

The matching system uses your configured LLM (OpenAI) from the Settings page. Make sure you have:
1. OpenAI API key configured
2. Model set to "gpt-4o-mini" or similar
3. Sufficient API credits

## Tips

- **Run matching after adding new extractors** to find cross-provider matches
- **Check confidence scores** - higher is better
- **Review manual matches** for low confidence results
- **Re-run periodically** as new models are added
- **Use unified view** to quickly find the cheapest option

## Limitations

- Requires LLM API access (OpenAI)
- Matching accuracy depends on model metadata quality
- Some models may be incorrectly matched (check confidence)
- Version differences may not be detected perfectly
- Manual review recommended for critical decisions

## Future Enhancements

- [ ] Manual match editing
- [ ] Bulk approve/reject matches
- [ ] Historical price tracking
- [ ] Automated re-matching schedule
- [ ] Export comparison reports
- [ ] Custom matching rules
- [ ] Model quality comparisons
