"""
Test MetisAI extractor

Tests the MetisAI pricing API extraction.
"""
from extractors.metisai_extractor import MetisAIExtractor


def main():
    """Test MetisAI extraction"""
    print("Testing MetisAI extractor...")
    
    # Create extractor
    extractor = MetisAIExtractor(use_llm=False, fetch_schemas=False)
    
    # Extract models
    models = extractor.extract()
    
    print(f"✅ Successfully extracted {len(models)} models")
    
    # Show sample models
    print("\nSample models:")
    for i, model in enumerate(models[:5], 1):
        model_id = model.get('model_id', 'Unknown')
        cost = model.get('cost_per_call', 0)
        pricing_type = model.get('pricing_type', 'unknown')
        model_type = model.get('model_type', 'unknown')
        print(f"{i}. {model_id} (${cost:.6f} {pricing_type}) - {model_type}")
    
    # Show model distribution by type
    type_counts = {}
    for model in models:
        model_type = model.get('model_type', 'unknown')
        type_counts[model_type] = type_counts.get(model_type, 0) + 1
    
    print("\nModel distribution by type:")
    for model_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_type}: {count}")
    
    # Show model distribution by provider
    provider_counts = {}
    for model in models:
        provider = model.get('pricing_variables', {}).get('provider', 'unknown')
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
    
    print("\nModel distribution by provider:")
    for provider, count in sorted(provider_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {provider}: {count}")
    
    # Show API type distribution
    api_type_counts = {}
    for model in models:
        api_type = model.get('category', 'unknown')
        api_type_counts[api_type] = api_type_counts.get(api_type, 0) + 1
    
    print("\nModel distribution by API type:")
    for api_type, count in sorted(api_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {api_type}: {count}")


if __name__ == '__main__':
    main()
