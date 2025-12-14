import re

# Test the new version matching logic
def test_version_match(search_versions, model_versions):
    """Test if search version matches any model version"""
    version_match = False
    for search_ver in search_versions:
        # Extract major version (e.g., "2.0" -> "2", "1.5" -> "1")
        search_major = search_ver.split('.')[0]
        
        for model_ver in model_versions:
            model_major = model_ver.split('.')[0] if '.' in model_ver else model_ver
            
            # Match if:
            # 1. Exact match (2.0 == 2.0)
            # 2. Major version match (2.0 matches 2, 2.1, 2.5)
            if search_ver == model_ver or search_major == model_major:
                version_match = True
                break
        
        if version_match:
            break
    
    return version_match

# Test cases
test_cases = [
    (["2.0"], ["2.1"], True, "2.0 should match 2.1 (same major version)"),
    (["2.0"], ["2"], True, "2.0 should match 2 (major version match)"),
    (["2.0"], ["3"], False, "2.0 should NOT match 3 (different major version)"),
    (["2.0"], ["1.5"], False, "2.0 should NOT match 1.5 (different major version)"),
    (["1.1"], ["1.1"], True, "1.1 should match 1.1 (exact match)"),
    (["3.5"], ["3", "2"], True, "3.5 should match 3 (major version match)"),
]

print("Testing version matching logic:")
print("="*80)

for search_vers, model_vers, expected, description in test_cases:
    result = test_version_match(search_vers, model_vers)
    status = "✅" if result == expected else "❌"
    print(f"{status} {description}")
    print(f"   Search: {search_vers}, Model: {model_vers}, Result: {result}, Expected: {expected}")
    print()
