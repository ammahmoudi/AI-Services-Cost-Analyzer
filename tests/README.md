# Test Suite Organization

This directory contains all test and verification scripts for the AI Services Cost Analyzer project.

## Directory Structure

### 📁 `database/`
Database-related tests and verification scripts:
- `check_model_79.py` - Check specific model (ID 79) data
- `check_cache_316.py` - Check cache entry 316
- `check_cache_db.py` - Verify cache database entries
- `check_db.py` - General database checks
- `check_llm_extracted.py` - Verify LLM extraction data in database
- `check_model_data.py` - Check model data integrity
- `test_direct_update.py` - Test direct database updates
- `verify_db_updates.py` - Verify database update operations

### 📁 `models/`
Model-related tests and inspections:
- `check_extraction_filter.py` - Test extraction filtering logic
- `check_types.py` - Verify model type normalization
- `inspect_models.py` - Inspect model data
- `inspect_models_detailed.py` - Detailed model inspection
- `test_model_79.py` - Specific tests for model 79

### 📁 `llm/`
LLM extraction and processing tests:
- `compare_llm_modeltype.py` - Compare LLM model type detection
- `test_llm_extraction.py` - Test LLM extraction functionality

### 📁 `extractors/`
Third-party API extractor tests:
- `test_avalai.py` - Avail AI extractor tests
- `test_avalai_debug.py` - Debug version
- `test_avalai_quick.py` - Quick test version
- `test_metisai.py` - Metis AI extractor tests
- `test_runware.py` - Runware extractor tests
- `test_together.py` - Together AI extractor tests
- `test_together_schema.py` - Together AI schema tests

### 📁 `playground/`
Playground data extraction tests:
- `test_debug_playground.py` - Debug playground extraction
- `test_playground.py` - Main playground tests
- `test_playground_direct.py` - Direct playground access tests
- `test_playground_method.py` - Playground method tests
- `test_parse_playground.py` - Playground parsing tests

### 📁 `pricing/`
Pricing extraction and parsing tests:
- `test_debug_pricing.py` - Debug pricing extraction
- `test_playwright_pricing.py` - Playwright-based pricing tests
- `test_pricing_complex.py` - Complex pricing scenarios
- `test_pricing_text.py` - Text-based pricing parsing
- `test_parse_pricing.py` - Pricing parsing tests
- `test_extract_billing.py` - Billing info extraction
- `test_extract_cost.py` - Cost extraction tests

### 📁 `extraction/`
General extraction workflow tests:
- `test_extraction_improved.py` - Improved extraction logic tests
- `test_single_model_extraction.py` - Single model extraction
- `test_with_schemas.py` - Schema extraction tests
- `test_parse_rsc.py` - RSC parsing tests

### 📁 `misc/`
Miscellaneous tests and utilities:
- `test_auth.py` - Authentication tests
- `test_cache_indicators.py` - Cache indicator tests
- `test_quick.py` - Quick smoke tests
- `test_refactored.py` - Refactored code tests
- `test_save_db.py` - Database save operations
- `test_simple_parse.py` - Simple parsing tests
- `setup_runware_auth.py` - Runware authentication setup

## Running Tests

### Run specific test file:
```bash
python tests/database/check_db.py
python tests/extractors/test_avalai.py
```

### Run all tests in a category:
```bash
python -m pytest tests/database/
python -m pytest tests/extractors/
```

### Run all tests:
```bash
python -m pytest tests/
```

### With UTF-8 Encoding (Windows):
```powershell
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
python tests/extraction/test_extraction_improved.py
```

## Notes

- Most test files are standalone scripts that can be run directly
- Database tests require proper `.env` configuration with database credentials
- Extractor tests may require API keys in environment variables
- Some tests use Playwright for browser automation

## Test Data

Tests use:
- Live API calls to fal.ai (rate-limited)
- Cached data from `cache/` directory when available
- Test database: `ai_costs.db`

## Notes

- Most tests fetch real data and may take time
- Cache significantly speeds up subsequent runs
- Some tests are for debugging specific issues (can be removed after fixes)
