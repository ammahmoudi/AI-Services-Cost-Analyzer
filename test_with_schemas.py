"""Test extraction with schemas enabled to see more cache indicators"""
from extractors.fal_extractor import FalAIExtractor
from ai_cost_manager.cache import cache_manager

print("Testing with schemas enabled...\n")

# Clear cache first
print("Clearing cache...")
cache_manager.clear_cache("fal.ai")

print("\nFirst run: Fresh data (with schemas)")
print("="*80)

# First extraction with schemas - fresh data
extractor1 = FalAIExtractor(fetch_schemas=True, use_llm=False)
models1 = extractor1.extract()
models1 = models1[:5]  # Only 5 models for faster test

print("\n\nSecond run: Cached data (with schemas)")
print("="*80)

# Second extraction with schemas - should use cache
extractor2 = FalAIExtractor(fetch_schemas=True, use_llm=False)
models2 = extractor2.extract()
models2 = models2[:5]

print("\n\n" + "="*80)
print("RESULTS")
print("="*80 + "\n")

for i, (m1, m2) in enumerate(zip(models1, models2), 1):
    print(f"{i}. {m1['name']} ({m1['model_id']})")
    
    print(f"   First run (fresh):")
    cache1 = m1.get('_cache_used', [])
    errors1 = m1.get('_errors', [])
    print(f"      Cache: {cache1 if cache1 else 'None - all fresh'}")
    if errors1:
        print(f"      Errors: {errors1}")
    
    print(f"   Second run (cached):")
    cache2 = m2.get('_cache_used', [])
    errors2 = m2.get('_errors', [])
    print(f"      Cache: {cache2 if cache2 else 'None'}")
    if errors2:
        print(f"      Errors: {errors2}")
    
    # Show what was fetched
    has_schema = 'Yes' if m2.get('input_schema') else 'No'
    has_playground = 'Yes' if m2.get('last_playground_fetched') else 'No'
    print(f"   Data available: Schema={has_schema}, Playground={has_playground}")
    print()

print("\nLegend for progress bar:")
print("  📦 = Using cached data")
print("  ⚠️ = Error occurred during fetch")
print("\n✅ Test completed!")
