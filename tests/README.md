# Tests

Test files for the AI Cost Manager project.

## Test Categories

### Extraction Tests
- `test_extraction_improved.py` - Full extraction with progress bar and timestamps
- `test_refactored.py` - Test refactored fal.ai extractor
- `test_quick.py` - Quick extraction test (limited models)
- `test_together.py` - Together.ai extractor tests

### Playground & Pricing Tests
- `test_playground.py` - Playground data extraction integration
- `test_playground_direct.py` - Direct POST to playground endpoint
- `test_playground_method.py` - Playground method testing
- `test_parse_playground.py` - HTML parsing from playground
- `test_pricing_text.py` - Pricing text extraction
- `test_pricing_complex.py` - Complex multi-tier pricing
- `test_extract_billing.py` - Billing data extraction
- `test_extract_cost.py` - Cost extraction with regex

### Cache Tests
- `test_cache_indicators.py` - Cache and error indicators in progress bar
- `test_with_schemas.py` - Schema caching with fresh vs cached runs

### Database Tests
- `test_save_db.py` - Model saving to database with timestamps

### Debug Tests
- `test_debug_playground.py` - Debug playground parsing issues
- `test_debug_pricing.py` - Debug pricing text extraction
- `test_parse_rsc.py` - Parse React Server Components
- `test_parse_pricing.py` - Parse pricing patterns
- `test_simple_parse.py` - Simple parsing tests

## Running Tests

### Individual Test
```bash
python tests/test_quick.py
```

### With UTF-8 Encoding (Windows)
```powershell
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
python tests/test_extraction_improved.py
```

## Test Data

Tests use:
- Live API calls to fal.ai (rate-limited)
- Cached data from `cache/` directory when available
- Test database: `ai_costs.db`

## Notes

- Most tests fetch real data and may take time
- Cache significantly speeds up subsequent runs
- Some tests are for debugging specific issues (can be removed after fixes)
