"""Test extraction with cache and error indicators"""
from extractors.fal_extractor import FalAIExtractor

print("Testing extraction with cache/error indicators...\n")
print("First run: Will fetch fresh data")
print("="*80)

# First extraction - fresh data
extractor1 = FalAIExtractor(fetch_schemas=False, use_llm=False)
models1 = extractor1.extract()
models1 = models1[:10]

print("\n\nSecond run: Will use cached data")
print("="*80)

# Second extraction - should use cache
extractor2 = FalAIExtractor(fetch_schemas=False, use_llm=False)
models2 = extractor2.extract()
models2 = models2[:10]

print("\n\n" + "="*80)
print("COMPARISON: Fresh vs Cached")
print("="*80 + "\n")

for i, (m1, m2) in enumerate(zip(models1[:5], models2[:5]), 1):
    print(f"{i}. {m1['name']}")
    print(f"   First run:")
    print(f"      Cache used: {m1.get('_cache_used', [])}")
    print(f"      Errors: {m1.get('_errors', [])}")
    print(f"   Second run:")
    print(f"      Cache used: {m2.get('_cache_used', [])}")
    print(f"      Errors: {m2.get('_errors', [])}")
    print()

print("Legend:")
print("  📦 = Using cached data")
print("  ⚠️ = Error occurred")
print("\n✅ Test completed!")
