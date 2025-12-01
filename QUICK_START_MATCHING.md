# Quick Start Guide - Model Matching Feature

## 🎉 Feature Complete!

The cross-provider model matching system is now fully implemented and ready to use.

## What You Can Do Now

✅ **Compare Prices** - See the same model across different providers  
✅ **Find Alternatives** - Discover cheaper options automatically  
✅ **Unified View** - Browse models by name instead of provider  
✅ **Save Money** - Identify cost savings opportunities  

## Setup Instructions

### Step 1: Install Dependencies (if not already done)

```bash
cd C:\Users\ammah\Documents\GitHub\ai-costs
pip install -r requirements.txt
```

### Step 2: Run Database Migration

```bash
python migrations/add_model_matching.py
```

This creates two new tables:
- `canonical_models` - Unified model identities
- `model_matches` - Links between provider models

### Step 3: Configure LLM (if not already done)

1. Open http://localhost:5000/settings
2. Add your OpenAI API key
3. Select model: "gpt-4o-mini"
4. Save configuration

### Step 4: Extract Models from Multiple Providers

Make sure you have models from at least 2-3 providers:
- FAL
- Replicate
- Runware
- Together AI
- AvalAI

### Step 5: Run Model Matching

**Option A - Web UI:**
1. Visit http://localhost:5000/canonical-models
2. Click "🤖 Run Model Matching"
3. Wait for analysis (30-60 seconds)
4. Explore results!

**Option B - Command Line:**
```bash
python run_matching.py
```

## Usage Examples

### Example 1: Find Cheaper Alternative

1. Go to any model detail page
2. Scroll to "🔄 Alternative Providers"
3. See price comparison
4. Click "View Details" on cheaper option

### Example 2: Browse Unified Models

1. Click "🔄 Unified View" in navigation
2. See models grouped by canonical name
3. Filter by type (image, text, video, audio)
4. Expand to see all providers and prices
5. ✓ Best price is highlighted

## What Was Built

### Backend
- `model_matcher.py` - LLM-based matching engine
- `model_matching_service.py` - Business logic layer
- 5 new API endpoints for matching operations
- Database tables for canonical models

### Frontend
- **Unified View** page - Compare all models
- **Alternative Providers** section on model pages
- Navigation link to unified view
- Price comparison UI with savings indicators

### Tools
- CLI tool for running matching
- Database migration script
- Complete documentation

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/match-models` | POST | Run matching on all models |
| `/api/models/{id}/alternatives` | GET | Get alternative providers |
| `/api/models/{id}/with-alternatives` | GET | Full comparison data |
| `/api/canonical-models` | GET | List unified models |
| `/api/canonical-models/{id}` | GET | Canonical model details |

## Files Created/Modified

### New Files (6)
- `ai_cost_manager/model_matcher.py`
- `ai_cost_manager/model_matching_service.py`
- `templates/canonical_models.html`
- `migrations/add_model_matching.py`
- `run_matching.py`
- `docs/MODEL_MATCHING.md`

### Modified Files (4)
- `ai_cost_manager/models.py` (added tables)
- `app.py` (added endpoints)
- `templates/model_detail.html` (added alternatives)
- `templates/base.html` (added nav link)

## How Matching Works

1. **Extracts features** from all models (name, provider, description, tags)
2. **Groups by type** (image, text, video, audio)
3. **Uses GPT-4o-mini** to identify equivalent models
4. **Assigns confidence scores** (0.0 - 1.0)
5. **Creates canonical models** for each unique model
6. **Links provider models** to canonical models
7. **Displays comparisons** in the UI

## Example Match

**FLUX.1 Dev across providers:**

| Provider | Model Name | Price | Status |
|----------|------------|-------|--------|
| Replicate | "FLUX.1 Dev" | $0.003 | ✓ Best |
| Runware | "bfl:4@1" | $0.004 | |
| FAL | "flux-dev-1.0" | $0.005 | |

**Savings**: Switch to Replicate and save $0.002 per call!

## Troubleshooting

### "No models found"
→ Extract models from sources first

### "Models already matched"
→ Use `--force` flag to re-match

### "LLM error"
→ Check OpenAI API key in settings

### "Low confidence matches"
→ Review manually, some may be incorrect

## Next Steps

1. ✅ Run migration
2. ✅ Extract models from multiple providers
3. ✅ Configure OpenAI API key
4. ✅ Run model matching
5. ✅ Explore unified view
6. ✅ Find cost savings!

## Documentation

- **Full Feature Guide**: `docs/MODEL_MATCHING.md`
- **Implementation Details**: `docs/MODEL_MATCHING_IMPLEMENTATION.md`
- **Runware Auth**: `docs/RUNWARE_AUTH.md`

## Support

The model matching system is production-ready and fully functional. If you encounter issues:

1. Check that OpenAI API key is configured
2. Ensure models exist in database
3. Review confidence scores for match quality
4. Check browser console for errors
5. Review server logs for API errors

---

**Enjoy finding the best prices across AI providers!** 🚀💰
