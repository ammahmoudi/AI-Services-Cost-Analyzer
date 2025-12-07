"""
Test extraction of a single model to debug pricing
"""
from extractors.together_extractor import TogetherAIExtractor
import json

print("Testing Together.ai extractor for FLUX.1 Kontext...")

# Initialize extractor
extractor = TogetherAIExtractor(use_llm=False, fetch_schemas=False)

# Test data (simulating API response)
test_model = {
    "id": "flux-1-kontext-[dev]",
    "display_name": "FLUX.1 Kontext [dev]",
    "type": "image",
    "pricing": {
        "per_mp": 0.025
    },
    "link": "https://www.together.ai/models/flux-1-kontext-dev",
    "images_per_dollar": 40.0,
    "default_steps": 28,
    "supports_batch": False
}

print("\n=== Input Data ===")
print(json.dumps(test_model, indent=2))

# Normalize
normalized = extractor._normalize_together_model(test_model)

print("\n=== Normalized Output ===")
print(f"Model ID: {normalized['model_id']}")
print(f"Name: {normalized['name']}")
print(f"Type: {normalized['model_type']}")
print(f"Cost per call: ${normalized['cost_per_call']}")
print(f"Pricing type: {normalized['pricing_type']}")
print(f"Cost unit: {normalized['cost_unit']}")
print(f"\nPricing variables:")
print(json.dumps(normalized['pricing_variables'], indent=2))

# Validation
print("\n=== Validation ===")
if normalized['cost_per_call'] == 0.025:
    print("✓ cost_per_call is correct: $0.025")
else:
    print(f"✗ cost_per_call is WRONG: ${normalized['cost_per_call']} (should be $0.025)")

if normalized['pricing_type'] == 'per_image':
    print("✓ pricing_type is correct: per_image")
else:
    print(f"✗ pricing_type is WRONG: {normalized['pricing_type']} (should be per_image)")

if normalized['cost_unit'] in ['megapixel', 'image']:
    print(f"✓ cost_unit is correct: {normalized['cost_unit']}")
else:
    print(f"✗ cost_unit is WRONG: {normalized['cost_unit']} (should be megapixel or image)")
