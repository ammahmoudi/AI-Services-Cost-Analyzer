"""
Test script for Together AI extractor
"""
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from extractors.together_extractor import TogetherAIExtractor


def test_together_extractor():
    """Test the Together AI extractor."""
    print("=" * 60)
    print("Testing Together AI Extractor")
    print("=" * 60)
    
    # Get API key from environment if available
    api_key = os.environ.get('TOGETHER_API_KEY')
    if not api_key:
        print("WARNING: No TOGETHER_API_KEY found in environment")
        print("   Will use web scraping fallback (public pricing page)\n")
    else:
        print("OK: Using API key for authentication\n")
    
    # Create extractor (works with or without API key)
    extractor = TogetherAIExtractor(api_key=api_key, fetch_schemas=True)
    
    # Get source info
    print("\n Source Info:")
    source_info = extractor.get_source_info()
    for key, value in source_info.items():
        print(f"   {key}: {value}")
    
    # Extract models
    print("\n" + "=" * 60)
    models = extractor.extract()
    
    if not models:
        print("\n No models extracted")
        return
    
    print("\n" + "=" * 60)
    print(f" Summary Statistics")
    print("=" * 60)
    print(f"Total models: {len(models)}")
    
    # Count models with batch API support
    batch_models = [m for m in models if m.get('batch_pricing') and m['batch_pricing'].get('supported')]
    print(f"Batch API supported: {len(batch_models)}")
    
    # Count models with schema data
    schema_models = [m for m in models if m.get('schema_data')]
    print(f"Schema data fetched: {len(schema_models)}")
    
    # Count by model type
    types = {}
    for model in models:
        model_type = model.get('model_type', 'unknown')
        types[model_type] = types.get(model_type, 0) + 1
    
    print(f"\nModels by type:")
    for model_type, count in sorted(types.items()):
        print(f"   {model_type}: {count}")
    
    # Show sample models
    print("\n" + "=" * 60)
    print(" Sample Models (first 5)")
    print("=" * 60)
    
    for i, model in enumerate(models[:5], 1):
        print(f"\n{i}. {model['name']}")
        print(f"   ID: {model['model_id']}")
        print(f"   Type: {model['model_type']}")
        print(f"   Pricing: {model['pricing_info']}")
        
        # Show schema data if available
        if model.get('schema_data'):
            schema = model['schema_data']
            if schema.get('pricing'):
                print(f"   Schema Pricing: {schema['pricing']}")
            elif schema.get('description'):
                desc = schema['description'][:80]
                print(f"   Schema Desc: {desc}...")
        
        if model.get('batch_pricing') and model['batch_pricing'].get('supported'):
            batch = model['batch_pricing']
            print(f"   Batch API:  50% discount")
            if batch.get('input_price_per_million'):
                print(f"      Input: ${batch['input_price_per_million']}/M tokens (was ${model['raw_metadata']['pricing']['input']}/M)")
            if batch.get('output_price_per_million'):
                print(f"      Output: ${batch['output_price_per_million']}/M tokens (was ${model['raw_metadata']['pricing']['output']}/M)")
        else:
            print(f"   Batch API:  Not supported")
        
        if model.get('tags'):
            print(f"   Tags: {', '.join(model['tags'][:5])}")
    
    # Show batch API models
    if batch_models:
        print("\n" + "=" * 60)
        print(" Batch API Supported Models (sample)")
        print("=" * 60)
        
        for i, model in enumerate(batch_models[:10], 1):
            batch = model['batch_pricing']
            input_price = model['raw_metadata']['pricing']['input']
            batch_input = batch['input_price_per_million']
            savings = input_price - batch_input if input_price and batch_input else 0
            
            print(f"{i}. {model['name']}")
            print(f"   Standard: ${input_price}/M tokens | Batch: ${batch_input}/M tokens | Saves: ${savings}/M")
    
    print("\n" + "=" * 60)
    print(" Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_together_extractor()

