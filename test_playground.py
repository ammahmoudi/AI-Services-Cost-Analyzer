"""
Test playground endpoint extraction for a single model.
"""
from extractors.fal_extractor import FalAIExtractor
import json

# Test with bria/reimagine/3.2 which you mentioned has playground pricing
extractor = FalAIExtractor(fetch_schemas=True, use_llm=False)
# Don't force refresh - test cache
# extractor.force_refresh = True

# Get one model to test
print("Fetching models...")
models = extractor._fetch_from_new_api()

# Find bria/reimagine/3.2
test_model = None
for model in models:
    if model.get('id') == 'bria/reimagine/3.2':
        test_model = model
        break

if test_model:
    print(f"\nFound model: {test_model.get('id')}")
    print("\nNormalizing model data...")
    normalized = extractor._normalize_fal_model(test_model)
    
    print("\n=== Playground Pricing ===")
    print(json.dumps(normalized.get('playground_pricing'), indent=2))
    
    print("\n=== Cost Info ===")
    print(f"Cost per call: ${normalized.get('cost_per_call')}")
    print(f"Credits required: {normalized.get('credits_required')}")
    
    print("\n=== Full Playground Data ===")
    playground_data = normalized.get('raw_metadata', {}).get('playground_data')
    if playground_data:
        print(json.dumps(playground_data, indent=2)[:500])  # First 500 chars
    else:
        print("No playground data")
else:
    print("Model bria/reimagine/3.2 not found")
