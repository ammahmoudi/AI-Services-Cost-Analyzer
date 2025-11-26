"""Quick test of extraction with progress bar showing model names"""
from extractors.fal_extractor import FalAIExtractor

print("Testing extraction with progress bar...\n")

# Create extractor
extractor = FalAIExtractor(fetch_schemas=False, use_llm=False)

# Monkey-patch to limit extraction for testing
original_extract = extractor.extract

def limited_extract():
    models = original_extract()
    return models[:20]  # Only first 20 models for quick test

extractor.extract = limited_extract

# Extract
models = extractor.extract()

print(f"\n{'='*80}")
print("EXTRACTED MODELS:")
print(f"{'='*80}\n")

for i, model in enumerate(models[:10], 1):
    print(f"{i}. {model['name']}")
    print(f"   ID: {model['model_id']}")
    print(f"   Type: {model['model_type']}")
    print(f"   Cost: ${model['cost_per_call']:.4f}")
    
    # Show what data we have
    has_data = []
    if model.get('last_raw_fetched'):
        has_data.append("✓ Raw")
    if model.get('input_schema'):
        has_data.append("✓ Schema")
    if model.get('last_playground_fetched'):
        has_data.append("✓ Playground")
    if model.get('last_llm_fetched'):
        has_data.append("✓ LLM")
    
    if has_data:
        print(f"   Data: {', '.join(has_data)}")
    print()

print("✅ Test completed!")
