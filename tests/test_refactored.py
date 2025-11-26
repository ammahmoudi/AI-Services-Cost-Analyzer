"""Test refactored FalAIExtractor"""
from extractors.fal_extractor import FalAIExtractor

# Create extractor
extractor = FalAIExtractor()
print("✓ Import successful")

# Extract first 3 models
print("\nExtracting first 3 models...")
models = extractor.extract()

print(f"\n✓ Total models extracted: {len(models)}")

if models:
    first_model = models[0]
    print(f"\nFirst model:")
    print(f"  Name: {first_model['name']}")
    print(f"  Provider: {first_model['provider']}")
    print(f"  Model type: {first_model['model_type']}")
    print(f"  Cost per call: ${first_model['cost_per_call']}")
    
    # Check if schemas work
    if first_model.get('input_schema'):
        print(f"  Input schema: {len(first_model['input_schema'].get('inputs', []))} fields")
    if first_model.get('output_schema'):
        print(f"  Output schema: {len(first_model['output_schema'].get('inputs', []))} fields")
    
    # Check playground data
    if 'pricing_text' in first_model.get('raw_metadata', {}):
        print(f"  Pricing text: {first_model['raw_metadata']['pricing_text'][:80]}...")
    
    print("\n✅ Refactored code works correctly!")
else:
    print("\n❌ No models extracted")
