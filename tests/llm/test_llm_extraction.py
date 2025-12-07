"""
Test LLM extraction with a specific pricing example
"""
from ai_cost_manager.llm_extractor import extract_pricing_with_llm

# Test with the FLUX.1 Kontext model
test_data = {
    'name': 'FLUX.1 Kontext [dev]',
    'model_type': 'image-generation',
    'pricing_info': '$0.025/MP | 40.0 img/$1 | 28 steps',
    'creditsRequired': None
}

print("Testing LLM extraction...")
print(f"Model: {test_data['name']}")
print(f"Pricing Info: {test_data['pricing_info']}")
print("\nExtracting...")

result = extract_pricing_with_llm(test_data)

print("\n=== LLM Extraction Result ===")
import json
print(json.dumps(result, indent=2))

print("\n=== Validation ===")
print(f"✓ pricing_type: {result.get('pricing_type')} (should be 'per_image' or 'per_megapixel')")
print(f"✓ cost_unit: {result.get('cost_unit')} (should be 'megapixel' or 'image')")
print(f"✓ cost_per_call: ${result.get('cost_per_call', 0)} (should be around $0.025)")

# Check if pricing_variables are correctly extracted
if result.get('pricing_variables'):
    print(f"✓ pricing_variables: {result['pricing_variables']}")
    vars = result['pricing_variables']
    if isinstance(vars.get('price_per_mp'), (int, float)):
        print(f"  ✓ price_per_mp is numeric: {vars['price_per_mp']}")
    else:
        print(f"  ✗ price_per_mp is NOT numeric: {vars.get('price_per_mp')}")
    
    if isinstance(vars.get('images_per_dollar'), (int, float)):
        print(f"  ✓ images_per_dollar is numeric: {vars['images_per_dollar']}")
    else:
        print(f"  ✗ images_per_dollar is NOT numeric: {vars.get('images_per_dollar')}")
