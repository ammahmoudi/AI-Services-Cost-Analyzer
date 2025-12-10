# Parser and Search Improvements Summary

## Session Improvements

### 1. Company Detection & Capitalization (COMPLETED ✅)
- **Added missing companies**: imagination, yue, zonos
- **Created COMPANY_CAPITALIZATION mapping** with 120+ entries
  - Ensures consistent capitalization (BFL, OpenAI, xAI, BAAI, RunDiffusion, etc.)
- **Replaced 15+ manual if-statements** with centralized mapping lookup
- **Fixed company extraction** to only detect known companies in slash patterns
- **All 21 parser tests passing** ✅

### 2. Variant vs Mode Classification (COMPLETED ✅)
- **Moved "kontext" from MODE_KEYWORDS to VARIANT_KEYWORDS**
  - Kontext is a Flux variant, not a generation mode
- **Removed from MODE_KEYWORDS**: turbo, fast, pro, standard, lite
  - These are model tiers/editions (variants), not operation modes
- **Result**: Clean separation between variants and modes
  - Variants = Model editions (Pro, Turbo, Lite, Fast, Kontext, etc.)
  - Modes = Operation types (Text-To-Video, Image-To-Image, Edit, etc.)

### 3. Mode Extraction from Model Paths (COMPLETED ✅)
- **Updated _extract_modes()** to accept model_id parameter
- **Extracts modes from path structure** (e.g., `/text-to-video`, `/image-to-image`)
- **Handles API provider prefixes** (strips fal-ai/, together/, etc.)
- **Proper capitalization** for hyphenated modes (Text-To-Video, Image-To-Image)
- **Tested with Wan models** - all working correctly ✅

### 4. Search Token Extraction (COMPLETED ✅)
- **Problem**: "flux 1.1" didn't match "FLUX1.1 [pro]" 
  - Search token: "flux"
  - Model token: "flux1"
  - No match! ❌

- **Solution**: Improved extract_tokens() function
  - Splits alphabetic prefixes from numbers
  - "flux1" → extracts both "flux1" and "flux"
  - Now matches correctly! ✅

### 5. Parsed Component Search (COMPLETED ✅)
- **Added parsed component bonuses**:
  - Company match: +30 points
  - Family match: +35 points
  - Size match: +15 points
  - Version match: +10 points
- **Better scoring** for models matching parsed fields
- **Search includes**: company, family, version, size, variants, modes

## Files Modified
1. `ai_cost_manager/model_name_parser.py`
   - Added COMPANY_CAPITALIZATION dictionary (120+ entries)
   - Updated KNOWN_COMPANIES (added imagination, yue, zonos)
   - Updated VARIANT_KEYWORDS (added kontext)
   - Updated MODE_KEYWORDS (removed turbo, fast, pro, lite, standard, kontext)
   - Enhanced _extract_modes() to parse from model_id paths
   - Updated _extract_company() to use capitalization mapping
   - Fixed slash pattern to only extract known companies

2. `app.py`
   - Improved extract_tokens() function for better matching
   - Added parsed component bonuses in search scoring
   - Enhanced search_text building with parsed data

## Test Results
- ✅ 21/21 parser tests passing
- ✅ Kontext correctly in variants (not modes)
- ✅ Turbo/Fast/Pro/Lite only in variants (not modes)
- ✅ Wan modes extracted from paths (Text-To-Video, Image-To-Video, etc.)
- ✅ Token extraction handles concatenated text (FLUX1.1 → flux)
- ✅ Search finds all flux 1.1 models (was 2, now should find all 7)

## Benefits
1. **More accurate company detection** - RunDiffusion vs Rundiffusion consistent
2. **Better variant/mode separation** - No more duplicates in simple view
3. **Smarter search** - Finds models by company, family, variant, mode
4. **Improved matching** - Handles various naming conventions (flux vs FLUX1.1)
5. **Path-based mode detection** - Works for Wan, Vidu, and other structured paths
