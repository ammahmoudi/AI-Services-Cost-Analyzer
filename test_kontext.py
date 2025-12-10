from ai_cost_manager.model_name_parser import ModelNameParser

parser = ModelNameParser()

# Test Flux Kontext models
test_cases = [
    ("BFL flux.1-kontext-pro", "flux.1-kontext-pro"),
    ("black-forest-labs flux-kontext-max", "black-forest-labs/flux-kontext-max"),
    ("black-forest-labs flux-kontext-pro", "black-forest-labs/flux-kontext-pro"),
]

print("KONTEXT VARIANT TEST")
print("=" * 60)

for name, model_id in test_cases:
    result = parser.parse(name, model_id)
    print(f"\nModel: {name}")
    print(f"  Company: {result.company}")
    print(f"  Family: {result.model_family}")
    print(f"  Variants: {result.variants}")
    print(f"  Modes: {result.modes}")
    
    # Check that kontext is in variants, not modes
    has_kontext_variant = result.variants and 'Kontext' in result.variants
    has_kontext_mode = result.modes and 'Kontext' in result.modes
    
    if has_kontext_variant and not has_kontext_mode:
        print("  ✅ PASS: Kontext correctly in variants")
    elif has_kontext_mode:
        print("  ❌ FAIL: Kontext incorrectly in modes")
    elif not has_kontext_variant:
        print("  ⚠️  WARNING: Kontext not detected")
