# Cache System Documentation

## Overview
The AI Cost Manager now includes a comprehensive caching system to avoid repeated API calls during extraction. This significantly reduces extraction time and API costs by storing raw data, OpenAPI schemas, and LLM extractions locally.

## Features

### 1. **Three-Level Caching**
- **Raw Data**: API responses from sources (e.g., fal.ai model list)
- **Schemas**: OpenAPI schemas fetched from model endpoints
- **LLM Extractions**: Pricing data extracted using LLM analysis

### 2. **Cache Storage Structure**
```
cache/
  └── {source_name}/          # e.g., "fal.ai"
      └── {model_id}/         # e.g., "fal-ai_flux-pro"
          ├── raw.json        # Raw API response
          ├── schema.json     # OpenAPI schema
          └── llm_extraction.json  # LLM-extracted pricing
```

### 3. **Force Refresh Option**
- CLI: `--force-refresh` flag
- UI: "Force refresh (bypass cache)" checkbox
- Fetches fresh data even if cache exists

### 4. **Cache Management UI**
- **Statistics Dashboard**: View cache size, model counts, schemas, LLM extractions
- **Per-Source View**: See cache stats for each API source
- **Selective Clearing**: 
  - Clear cache for specific sources
  - Clear only LLM extractions
  - Clear only schemas
  - Clear all cache

## Usage

### CLI Commands

#### Extract with Cache (default)
```bash
python manage.py extract
```
This will use cached data when available, only fetching new/changed data.

#### Extract with Force Refresh
```bash
python manage.py extract --force-refresh
```
Bypasses cache and fetches fresh data from all APIs.

#### Extract with LLM and Schemas
```bash
python manage.py extract --use-llm --fetch-schemas
```
Uses cache for LLM and schema data when available.

### Web UI

#### Extraction Dialog
1. Navigate to **Sources** page
2. Click **Extract** button on a source
3. Choose options:
   - ✅ **Use LLM for pricing extraction**
   - ✅ **Fetch OpenAPI schemas**
   - ✅ **Force refresh (bypass cache)**
4. Click **Start Extraction**

#### Cache Management
1. Navigate to **📦 Cache** in the navigation menu
2. View cache statistics:
   - Total cached models
   - Number of schemas
   - Number of LLM extractions
   - Total cache size in MB
3. Clear cache:
   - **Per Source**: Clear cache for specific API source
   - **Clear All Cache**: Delete all cached data
   - **Clear LLM Extractions Only**: Remove only LLM analysis results
   - **Clear Schemas Only**: Remove only OpenAPI schemas

## Cache Behavior

### First Extraction
```
1. Fetch raw data from API → Save to cache/source/model/raw.json
2. (If enabled) Fetch schema → Save to cache/source/model/schema.json
3. (If enabled) Extract with LLM → Save to cache/source/model/llm_extraction.json
4. Save to database
```

### Subsequent Extractions (with cache)
```
1. Check cache/source/model/raw.json → Use if exists
2. (If enabled) Check cache/source/model/schema.json → Use if exists
3. (If enabled) Check cache/source/model/llm_extraction.json → Use if exists
4. Update database
```

### With Force Refresh
```
1. Fetch fresh data from API → Overwrite cache
2. Fetch fresh schemas → Overwrite cache
3. Re-extract with LLM → Overwrite cache
4. Update database
```

## Benefits

### Performance
- **First extraction** (880 fal.ai models):
  - Without cache: ~15-30 minutes (depending on schema/LLM options)
  - With cache: ~2-5 minutes
  
### Cost Savings
- **Schema fetching**: Saves 880 HTTP requests (some timeout at 5s)
- **LLM extractions**: Saves 880 OpenRouter API calls (most expensive)
- **API rate limits**: Reduces load on fal.ai API

### Incremental Updates
- Only new/changed models need fresh fetching
- Existing models load instantly from cache
- LLM extractions preserved across runs

## Cache File Format

### raw.json
```json
{
  "cached_at": "2025-01-15T10:30:00.000000",
  "source": "fal.ai",
  "model_id": "fal-ai/flux-pro",
  "data": {
    "id": "fal-ai/flux-pro",
    "title": "FLUX Pro",
    "description": "...",
    "pricingInfo": "...",
    ...
  }
}
```

### schema.json
```json
{
  "cached_at": "2025-01-15T10:30:05.000000",
  "source": "fal.ai",
  "model_id": "fal-ai/flux-pro",
  "schema": {
    "openapi": "3.0.0",
    "paths": { ... },
    "components": { ... }
  }
}
```

### llm_extraction.json
```json
{
  "cached_at": "2025-01-15T10:30:10.000000",
  "source": "fal.ai",
  "model_id": "fal-ai/flux-pro",
  "extraction": {
    "pricing_type": "per-generation",
    "pricing_formula": "0.05 per image",
    "pricing_variables": "image_count",
    "input_cost_per_unit": 0.05,
    "output_cost_per_unit": null,
    "cost_unit": "generation"
  }
}
```

## Implementation Details

### CacheManager Class
- **Location**: `ai_cost_manager/cache.py`
- **Methods**:
  - `save_raw_data(source, model_id, data)`
  - `load_raw_data(source, model_id)`
  - `save_schema(source, model_id, schema)`
  - `load_schema(source, model_id)`
  - `save_llm_extraction(source, model_id, llm_data)`
  - `load_llm_extraction(source, model_id)`
  - `clear_cache(source, model_id)`
  - `clear_all_cache()`
  - `clear_llm_cache()`
  - `clear_schema_cache()`
  - `get_cache_stats()`

### Integration Points
- **FalAIExtractor**: Checks cache before API calls, saves at each step
- **CLI**: `--force-refresh` flag bypasses cache
- **Web UI**: Force refresh checkbox in extraction dialog
- **Flask Routes**: `/cache`, `/cache/clear/*` endpoints

## Best Practices

1. **First Run**: Use `--force-refresh` to populate cache
2. **Regular Updates**: Run without `--force-refresh` to use cache
3. **API Changes**: Use `--force-refresh` when API structure changes
4. **LLM Improvements**: Clear LLM cache only to re-extract pricing
5. **Disk Space**: Monitor cache size, clear old data periodically

## Troubleshooting

### Cache Not Loading
- Check file permissions in `cache/` directory
- Verify JSON files are valid (not corrupted)
- Check console for cache loading errors

### Stale Data
- Use `--force-refresh` to update
- Clear specific source cache
- Check `cached_at` timestamp in JSON files

### Large Cache Size
- Clear schema cache if not needed
- Clear LLM cache if pricing changes
- Use per-source clearing for specific cleanup

## Future Enhancements
- Cache expiration based on age
- Automatic cache invalidation on API changes
- Cache compression for large datasets
- Distributed cache for multi-instance deployments
