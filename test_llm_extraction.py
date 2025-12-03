#!/usr/bin/env python3
"""Test LLM extraction with specific model examples"""

from ai_cost_manager.llm_extractor import LLMPricingExtractor

# Test cases
test_models = [
    {
        'name': 'Hyper3d (V2)',
        'model_id': 'hyper3d-v2',
        'description': '3D model generation',
        'pricing_info': '$0.05 per generation',
        'creditsRequired': 5,
        'model_type': 'other',
        'category': None,
        'tags': [],
        'raw_metadata': {},
        'input_schema': {},
        'output_schema': {},
    },
    {
        'name': 'SAM (Segment Anything Model)',
        'model_id': 'sam',
        'description': 'Object detection and segmentation',
        'pricing_info': '$0.01 per image',
        'creditsRequired': 1,
        'model_type': 'other',
        'category': None,
        'tags': [],
        'raw_metadata': {},
        'input_schema': {},
        'output_schema': {},
    },
    {
        'name': 'LoRA Training',
        'model_id': 'lora-training',
        'description': 'Train custom LoRA models',
        'pricing_info': '$2.00 per training run',
        'creditsRequired': 200,
        'model_type': 'other',
        'category': None,
        'tags': [],
        'raw_metadata': {},
        'input_schema': {},
        'output_schema': {},
    },
]

print("\n🧪 Testing LLM Extraction Type Detection\n")
print("=" * 60)

extractor = LLMPricingExtractor()

if not extractor.config:
    print("\n❌ No LLM configuration found!")
    print("Please configure an LLM in Settings → LLM Configuration")
    exit(1)

for test_model in test_models:
    print(f"\n📝 Testing: {test_model['name']}")
    print(f"   Current type: {test_model['model_type']}")
    
    try:
        result = extractor.extract_pricing(test_model)
        
        extracted_type = result.get('model_type', 'unknown')
        extracted_category = result.get('category', 'none')
        
        print(f"   ✅ Extracted type: {extracted_type}")
        print(f"   ✅ Extracted category: {extracted_category}")
        
        if extracted_type == 'other':
            print(f"   ⚠️  WARNING: Still classified as 'other'!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("\n✅ Test completed\n")
