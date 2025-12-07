"""
Test AvalAI Extractor

Quick test to verify the AvalAI extractor can fetch and parse pricing data.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractors.avalai_extractor import AvalAIExtractor


def test_avalai_basic():
    """Test basic AvalAI extraction"""
    print("Testing AvalAI extractor...")
    
    extractor = AvalAIExtractor()
    
    # Get a small sample (first 5 models)
    print("\nFetching models from AvalAI...")
    models = extractor.extract()
    
    if not models:
        print("ERROR: No models extracted!")
        return
    
    print(f"\n✓ Successfully extracted {len(models)} models")
    
    # Show first 3 models as examples
    print("\nSample models:")
    for i, model in enumerate(models[:3], 1):
        print(f"\n{i}. {model['name']}")
        print(f"   Model ID: {model['model_id']}")
        print(f"   Provider: {model.get('raw_metadata', {}).get('provider', 'N/A')}")
        print(f"   Type: {model['model_type']}")
        print(f"   Category: {model['category']}")
        print(f"   Cost per call: ${model['cost_per_call']:.6f}")
        print(f"   Pricing: {model['pricing_info']}")
        if model.get('pricing_formula'):
            print(f"   Formula: {model['pricing_formula']}")
    
    # Show statistics
    categories = {}
    for model in models:
        cat = model['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n\nModel distribution by category:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} models")
    
    print("\n✅ AvalAI extractor test completed successfully!")


if __name__ == '__main__':
    test_avalai_basic()
